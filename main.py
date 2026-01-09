from __future__ import annotations

import argparse
import copy
import random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, average_precision_score
import os
import time

from data_loader import build_standard_loaders
from model import QuantumAnsatz, HybridQuantumClassifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantum patch model training entrypoint")
    default_binary_root = Path(__file__).resolve().parent / "data" / "Binary_dataset"
    default_multi_root = Path(__file__).resolve().parent / "data" / "Multi_dataset"

    # Data
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument(
        "--dataset-choice",
        type=str,
        default="cifar10",
        choices=["mnist", "fmnist", "cifar10", "pcam"],
        help="Choose dataset source.",
    )
    parser.add_argument(
        "--classification-task",
        type=str,
        default="multi",
        choices=["binary", "multi"],
        help="Classification task: binary or multi.",
    )
    parser.add_argument(
        "--binary-data-root",
        type=Path,
        default=default_binary_root,
        help="Root directory for Binary_dataset splits.",
    )
    parser.add_argument(
        "--multi-data-root",
        type=Path,
        default=default_multi_root,
        help="Root directory for Multi_dataset splits.",
    )
    parser.add_argument(
        "--dataset-labels",
        type=int,
        nargs="*",
        help="Optional list of labels to keep (applies to MNIST/FMNIST/CIFAR10).",
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        help="Optional per-label cap for MNIST/FMNIST/CIFAR10; used as per-class cap for PCam.",
    )
    # Quantum ansatz
    parser.add_argument("--num-qubits", type=int, default=8)
    parser.add_argument("--vqc-layers", type=int, default=4)
    parser.add_argument("--reuploading", type=int, default=2, help="Number of times to repeat data encoding + VQC block.")
    parser.add_argument("--backend-device", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--use-torch-autograd", action="store_true", default=True)

    # Data split (counts only)
    parser.add_argument("--train-count", type=int, default=320, help="Number of training samples.")
    parser.add_argument("--val-count", type=int, default=0, help="Number of validation samples.")
    parser.add_argument("--test-count", type=int, default=80, help="Number of test samples.")

    # Attention
    parser.add_argument("--attn-layers", type=int, default=1)
    parser.add_argument("--rbf-gamma", type=float, default=1.0)

    # Classifier
    parser.add_argument(
        "--debug-logs",
        action="store_true",
        help="Print tensor statistics for the first batch of each split in every epoch.",
    )
    parser.add_argument(
        "--save-statevector",
        action="store_true",
        help="Save per-layer Q/K/V statevectors before measurement for selected epochs.",
    )
    parser.add_argument(
        "--save-statevector-epoch",
        type=int,
        default=1,
        help="Save statevectors every N epochs (1 means every epoch).",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda:0"])
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0001)
    parser.add_argument("--no-pos-weight", action="store_true", help="Disable class-balanced pos_weight in BCE loss.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num-classes", type=int, default=10, help="If >0, override inferred class count.")

    # Logging / checkpoints
    parser.add_argument("--log-dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--model-dir", type=Path, default=Path("results/models"))
    parser.add_argument("--run-name", type=str, default="auto", help="If 'auto', name will encode dataset, count, and timestamp.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    best_overall = {"auroc": -1.0, "run_name": None}

    def run_once(cfg: argparse.Namespace) -> dict:
        # Reproducibility
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        balance_sampler = False
        debug_logs = cfg.debug_logs
        (train_loader, train_set), (val_loader, val_set), (test_loader, test_set) = build_standard_loaders(
            dataset_choice=cfg.dataset_choice,
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True if "cuda" in cfg.device else False,
            train_count=cfg.train_count,
            val_count=cfg.val_count,
            test_count=cfg.test_count,
            seed=cfg.seed,
            dataset_labels=cfg.dataset_labels,
            samples_per_label=cfg.samples_per_label,
            balance_sampler=balance_sampler,
            binary_root=cfg.binary_data_root,
            multi_root=cfg.multi_data_root,
            classification_task=cfg.classification_task,
        )

        def collect_labels(ds: Dataset) -> list[int]:
            if hasattr(ds, "indices") and hasattr(ds, "dataset") and hasattr(ds.dataset, "labels"):
                return [int(ds.dataset.labels[i]) for i in ds.indices]
            labels = getattr(ds, "labels", None)
            if labels is None:
                labels = getattr(ds, "targets", None)
            if labels is None:
                return []
            if torch.is_tensor(labels):
                return labels.detach().cpu().numpy().astype(int).tolist()
            if isinstance(labels, np.ndarray):
                return labels.astype(int).tolist()
            return [int(v) for v in labels]

        train_labels = collect_labels(train_set)
        val_labels = collect_labels(val_set)
        test_labels = collect_labels(test_set)
        all_labels = train_labels + val_labels + test_labels
        total_samples = len(all_labels)
        label_counts = Counter(all_labels)
        inferred_classes = len(label_counts) if label_counts else 0
        if cfg.classification_task == "binary":
            num_classes = 2
            is_binary = True
        else:
            num_classes = cfg.num_classes if cfg.num_classes > 0 else inferred_classes
            if num_classes < 2:
                raise ValueError(
                    f"num_classes must be >=2 (inferred {inferred_classes}, override with --num-classes)."
                )
            is_binary = False
        dataset_tag = cfg.dataset_choice
        if cfg.dataset_labels:
            label_tag = "-".join(str(l) for l in cfg.dataset_labels)
            dataset_tag = f"{dataset_tag}[{label_tag}]"
        if cfg.samples_per_label:
            dataset_tag = f"{dataset_tag}-n{cfg.samples_per_label}"
        model_out_dim = 1 if is_binary else num_classes
        if cfg.run_name == "auto":
            ts = time.strftime("%Y%m%d-%H%M%S")
            cfg.run_name = f"{dataset_tag}-N{total_samples}-{ts}"
        print(
            f"[run {cfg.run_name}] Data splits -> train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)} | "
            f"batch_size: {cfg.batch_size}, image_size: {cfg.image_size}, patch_size: {cfg.patch_size}, device: {device}"
        )
        print(
            f"[run {cfg.run_name}] Dataset={dataset_tag}, task={cfg.classification_task}, total={total_samples}, "
            f"label_counts={dict(label_counts)}"
        )

        class_counts = Counter(train_labels)
        pos_weight_tensor = None
        pos_weight_val = None
        if is_binary:
            pos_count = class_counts.get(1, 0)
            neg_count = class_counts.get(0, 0)
            if balance_sampler:
                print(f"[run {cfg.run_name}] Balance sampler ON -> disabling pos_weight to avoid double correction.")
            elif not cfg.no_pos_weight and pos_count > 0 and neg_count > 0:
                pos_weight_val = neg_count / pos_count
                pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
            print(
                f"[run {cfg.run_name}] Class balance -> pos: {pos_count}, neg: {neg_count}"
                + (f", pos_weight: {pos_weight_val:.3f}" if pos_weight_val is not None else " (pos_weight disabled)")
            )

        def tensor_stats(x: torch.Tensor) -> str:
            x_det = x.detach()
            if x_det.is_complex():
                x_det = x_det.abs()
            flat = x_det.float().reshape(-1)
            return (
                f"shape={list(x_det.shape)} "
                f"mean={flat.mean().item():.4f} std={flat.std().item():.4f} "
                f"min={flat.min().item():.4f} max={flat.max().item():.4f}"
            )

        def log_debug(phase: str, images: torch.Tensor, labels: torch.Tensor, outputs: torch.Tensor, loss: torch.Tensor, attn_stats):
            if is_binary:
                probs = torch.sigmoid(outputs.detach())
            else:
                probs = torch.softmax(outputs.detach(), dim=1)
            print(f"[debug:{phase}] images {tensor_stats(images)} device={images.device} dtype={images.dtype}")
            print(f"[debug:{phase}] labels {tensor_stats(labels)} pos_frac={labels.float().mean().item():.4f}")
            print(f"[debug:{phase}] logits {tensor_stats(outputs)}")
            print(f"[debug:{phase}] probs {tensor_stats(probs)}")
            print(f"[debug:{phase}] loss={loss.item():.4f}")
            if attn_stats is not None:
                print(f"[debug:{phase}] attn entropy={attn_stats['entropy']:.4f} max_w={attn_stats['max_weight']:.4f}")

        def report_grads() -> None:
            missing: list[str] = []
            zero: list[str] = []
            max_vals: list[float] = []
            mean_vals: list[float] = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                g = param.grad
                if g is None:
                    missing.append(name)
                else:
                    if g.is_sparse:
                        vals = g.coalesce().values()
                    else:
                        vals = g
                    max_abs = vals.abs().max().item()
                    mean_abs = vals.abs().mean().item()
                    max_vals.append(max_abs)
                    mean_vals.append(mean_abs)
                    if max_abs == 0:
                        zero.append(name)
            msg = f"[grad] missing={len(missing)} zero={len(zero)}"
            if max_vals:
                msg += f" max={max(max_vals):.3e} mean={sum(mean_vals)/len(mean_vals):.3e}"
            if missing:
                msg += f" missing_examples={missing[:3]}"
            elif zero:
                msg += f" zero_examples={zero[:3]}"
            print(msg)

        def infer_channels(ds: Dataset) -> int:
            sample = ds[0][0]
            if sample.dim() != 3:
                raise ValueError(f"Expected sample with 3 dims [C,H,W], got {sample.shape}")
            return sample.shape[0]

        # Support variable channel counts for standard datasets
        channel_count = infer_channels(train_set)
        patch_area = cfg.patch_size * cfg.patch_size

        ansatz = QuantumAnsatz(
            data_dim=channel_count * patch_area,
            num_qubits=cfg.num_qubits,
            vqc_layers=cfg.vqc_layers,
            reuploading=cfg.reuploading,
            backend_device=cfg.backend_device,
            use_torch_autograd=cfg.use_torch_autograd,
        )
        ansatz_k = QuantumAnsatz(
            data_dim=channel_count * patch_area,
            num_qubits=cfg.num_qubits,
            vqc_layers=cfg.vqc_layers,
            reuploading=cfg.reuploading,
            backend_device=cfg.backend_device,
            use_torch_autograd=cfg.use_torch_autograd,
        )
        ansatz_v = QuantumAnsatz(
            data_dim=channel_count * patch_area,
            num_qubits=cfg.num_qubits,
            vqc_layers=cfg.vqc_layers,
            reuploading=cfg.reuploading,
            backend_device=cfg.backend_device,
            use_torch_autograd=cfg.use_torch_autograd,
        )
        model = HybridQuantumClassifier(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            ansatz_q=ansatz,
            ansatz_k=ansatz_k,
            ansatz_v=ansatz_v,
            attn_layers=cfg.attn_layers,
            rbf_gamma=cfg.rbf_gamma,
            device=device,
            save_statevector=cfg.save_statevector,
            save_statevector_epoch=cfg.save_statevector_epoch,
            num_classes=model_out_dim,
            classification_task=cfg.classification_task,
        )
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[run {cfg.run_name}] Trainable parameters: {param_count:,}")
        if is_binary:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        def run_epoch(loader, train: bool, phase: str):
            nonlocal best_val_loss, patience_counter
            if train:
                model.train()
            else:
                model.eval()
            total_loss = 0.0
            total = 0
            all_labels = []
            all_outputs = []
            grad_checked = False
            debug_logged = False
            for images, labels, _ in loader:
                images = images.to(device)
                if is_binary:
                    labels = labels.float().to(device)
                else:
                    labels = labels.long().to(device)
                if train:
                    optimizer.zero_grad()
                with torch.set_grad_enabled(train):
                    out = model(images)
                    if isinstance(out, tuple):
                        outputs, attn_stats = out
                    else:
                        outputs, attn_stats = out, None
                    loss = criterion(outputs, labels)
                    if train:
                        loss.backward()
                        if not grad_checked:
                            report_grads()
                            if attn_stats is not None:
                                print(f"[attn] entropy={attn_stats['entropy']:.4f} max_w={attn_stats['max_weight']:.4f}")
                            grad_checked = True
                        optimizer.step()
                if debug_logs and not debug_logged:
                    log_debug(phase, images, labels, outputs, loss, attn_stats)
                    debug_logged = True
                total_loss += loss.item() * labels.size(0)
                total += labels.size(0)
                all_labels.append(labels.detach().cpu())
                all_outputs.append(outputs.detach().cpu())
            if all_labels:
                labels_cat = torch.cat(all_labels)
                outputs_cat = torch.cat(all_outputs)
                if is_binary:
                    probs = torch.sigmoid(outputs_cat)
                    preds = (probs >= 0.5).int()
                else:
                    probs = torch.softmax(outputs_cat, dim=1)
                    preds = probs.argmax(dim=1)
                acc = accuracy_score(labels_cat.numpy(), preds.numpy())
            else:
                acc = 0.0
            return total_loss / max(1, total), acc

        for epoch in range(1, cfg.epochs + 1):
            if hasattr(model, "configure_statevector_saving"):
                model.configure_statevector_saving(epoch, active=True, reset_storage=True)
            train_loss, train_acc = run_epoch(train_loader, train=True, phase=f"train/epoch{epoch}")
            if hasattr(model, "configure_statevector_saving"):
                model.configure_statevector_saving(epoch, active=False)
            val_loss, val_acc = run_epoch(val_loader, train=False, phase=f"val/epoch{epoch}")
            print(
                f"[{cfg.run_name}] Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            if cfg.early_stop:
                if val_loss + cfg.early_stop_min_delta < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model based on validation loss
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.early_stop_patience:
                        print(f"[{cfg.run_name}] Early stopping triggered.")
                        break

        # Restore best model if early stopping was used
        if cfg.early_stop and best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"[{cfg.run_name}] Restored best model from epoch with lowest validation loss.")

        # Test evaluation
        model.eval()
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(device)
                labels = labels.float().to(device)
                out = model(images)
                outputs = out[0] if isinstance(out, tuple) else out
                all_labels.append(labels.cpu())
                all_outputs.append(outputs.cpu())
        metrics = {}
        if all_labels:
            y_true_t = torch.cat(all_labels)
            y_scores_t = torch.cat(all_outputs)
            if is_binary:
                probs = torch.sigmoid(y_scores_t)
                y_pred = (probs >= 0.5).int()
                metrics["acc"] = accuracy_score(y_true_t.numpy(), y_pred.numpy())
                try:
                    metrics["auroc"] = roc_auc_score(y_true_t.numpy(), y_scores_t.numpy())
                except ValueError:
                    metrics["auroc"] = float("nan")
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true_t.numpy(), y_pred.numpy(), average="binary", zero_division=0
                )
                try:
                    metrics["auprc"] = average_precision_score(y_true_t.numpy(), probs.numpy())
                except ValueError:
                    metrics["auprc"] = float("nan")
            else:
                probs = torch.softmax(y_scores_t, dim=1)
                y_pred = probs.argmax(dim=1)
                metrics["acc"] = accuracy_score(y_true_t.numpy(), y_pred.numpy())
                try:
                    y_true_onehot = torch.nn.functional.one_hot(y_true_t.long(), num_classes=num_classes).numpy()
                    metrics["auroc"] = roc_auc_score(y_true_onehot, probs.numpy(), multi_class="ovr")
                except ValueError:
                    metrics["auroc"] = float("nan")
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true_t.numpy(), y_pred.numpy(), average="macro", zero_division=0
                )
                try:
                    y_true_onehot = torch.nn.functional.one_hot(y_true_t.long(), num_classes=num_classes).numpy()
                    metrics["auprc"] = average_precision_score(y_true_onehot, probs.numpy(), average="macro")
                except ValueError:
                    metrics["auprc"] = float("nan")
            metrics.update({"precision": precision, "recall": recall, "f1": f1})
            print(
                f"[{cfg.run_name}] Test -> acc: {metrics['acc']:.4f} auroc: {metrics['auroc']:.4f} "
                f"precision: {metrics['precision']:.4f} recall: {metrics['recall']:.4f} f1: {metrics['f1']:.4f} "
                f"auprc: {metrics.get('auprc', float('nan')):.4f}"
            )
            if not (metrics["auroc"] != metrics["auroc"]):  # not NaN
                if metrics["auroc"] > best_overall["auroc"]:
                    best_overall["auroc"] = metrics["auroc"]
                    best_overall["run_name"] = cfg.run_name
                    os.makedirs(cfg.model_dir, exist_ok=True)
                    save_path = Path(cfg.model_dir) / f"{cfg.run_name}_best.pt"
                    torch.save(model.state_dict(), save_path)
                    print(f"[best] Updated best AUROC: {metrics['auroc']:.4f} | saved model to {save_path}")
        return metrics

    run_once(args)


if __name__ == "__main__":
    main()
