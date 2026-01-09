from __future__ import annotations

import argparse
import copy
import math
import os
import signal
import subprocess
import atexit
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from pathlib import Path
from typing import List, Sequence, Tuple

import pennylane as qml

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from data_loader import build_standard_loaders

# Repro

def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


class Tee:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")

    def write(self, data):
        for s in self.streams:
            s.write(data)
        self.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


def setup_logging(prefix: str) -> Path:
    env_prefix = os.environ.get("QSAN_LOG_PREFIX")
    if env_prefix:
        prefix = env_prefix
    log_dir = Path(os.environ.get("QSAN_LOG_DIR", Path(__file__).resolve().parents[1] / "results" / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("QSAN_LOG_TAG") or time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{prefix}_{tag}.log"
    log_fh = log_path.open("w", encoding="utf-8")

    sys.stdout = Tee(sys.stdout, log_fh)
    sys.stderr = Tee(sys.stderr, log_fh)

    def _close():
        try:
            log_fh.close()
        except Exception:
            pass

    atexit.register(_close)
    return log_path


# Helper Functions from data_loader.py

def _normalize_image_size(image_size: Sequence[int] | int) -> Tuple[int, int]:
    if isinstance(image_size, int):
        return (image_size, image_size)
    if len(image_size) != 2:
        raise ValueError("image_size must be an int or a length-2 sequence")
    return (int(image_size[0]), int(image_size[1]))


# PCam Dataset from data_loader.py (Modified for QSAN)

class PCamDataset(torch.utils.data.Dataset):
    """
    PatchCamelyon (Camelyon16) patches stored as .npy under class0/class1 folders.
    Modified to be compatible with QSAN's filter_remap_dataset logic.
    """

    CLASS_MAP = [("class0", 0), ("class1", 1)]

    def __init__(
        self,
        root: str | Path,
        image_size: Sequence[int] | int,
        samples_per_class: int | None = None,
        seed: int = 42,
        force_grayscale: bool = True
    ) -> None:
        self.root = Path(root)
        self.image_size = _normalize_image_size(image_size)
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.force_grayscale = force_grayscale
        
        self.items = self._gather_items()
        self.labels = [lbl for _, lbl in self.items]
        
        self.targets = self.labels 
        
        self.num_channels = 1 if force_grayscale else self._infer_channels()

    def _gather_items(self) -> List[Tuple[Path, int]]:
        rng = np.random.default_rng(self.seed)
        items: List[Tuple[Path, int]] = []
        for cname, label in self.CLASS_MAP:
            cdir = self.root / cname
            if not cdir.is_dir():
                continue
            paths = sorted(cdir.rglob("*.npy"))
            if self.samples_per_class is not None and len(paths) > self.samples_per_class:
                rng.shuffle(paths)
                paths = paths[: self.samples_per_class]
            for p in paths:
                items.append((p, label))
        
        if not items:
            pass 
            
        rng.shuffle(items)
        return items

    def _infer_channels(self) -> int:
        if not self.items: return 3
        sample = np.load(self.items[0][0])
        if sample.ndim == 2:
            return 1
        if sample.ndim == 3:
            return sample.shape[-1]
        return 3

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        arr = np.load(path)
        
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        
        img = torch.from_numpy(arr).permute(2, 0, 1).float()
        
        if img.max() > 1.0:
            img = img / 255.0
            
        img = TF.resize(img, self.image_size, antialias=True)

        if self.force_grayscale and img.shape[0] == 3:
            img = TF.rgb_to_grayscale(img)

        return img, label



# Dataset filtering + remap

def filter_remap_dataset(ds, classes: list[int], per_class_limit: int | None, seed: int):
    """
    Keep only samples with label in `classes` and remap labels to 0..K-1.
    Handles both standard torchvision datasets and the custom PCamDataset.
    """
    rng = np.random.default_rng(seed)


    if isinstance(ds.targets, list):
        y = np.array(ds.targets, dtype=np.int64)
    else:

        if torch.is_tensor(ds.targets):
            y = ds.targets.detach().cpu().numpy().astype(np.int64)
        else:
             y = np.array(ds.targets, dtype=np.int64)

    classes = list(classes)
    keep = np.isin(y, classes)
    idx_all = np.where(keep)[0]

    if per_class_limit is None:
        idx_keep = idx_all
    else:
        idx_keep = []
        for c in classes:
            idx_c = idx_all[y[idx_all] == c]
            rng.shuffle(idx_c)
            idx_keep.extend(idx_c[:per_class_limit].tolist())
        idx_keep = np.array(idx_keep, dtype=np.int64)

    c2n = {c: i for i, c in enumerate(classes)}
    y_new = np.array([c2n[int(y[i])] for i in idx_keep], dtype=np.int64)

    if hasattr(ds, "data") and not isinstance(ds, PCamDataset):
        ds.data = ds.data[idx_keep]
        if isinstance(ds.targets, list):
            ds.targets = y_new.tolist()
        else:
            ds.targets = torch.tensor(y_new, dtype=torch.long)
    
    elif isinstance(ds, PCamDataset):
        ds.items = [ds.items[i] for i in idx_keep]
        ds.labels = y_new.tolist()
        ds.targets = ds.labels 
    
    else:
        pass

    return ds



# Metrics

def compute_metrics_binary(y_true: np.ndarray, prob_pos: np.ndarray) -> dict:
    y_pred = (prob_pos >= 0.5).astype(np.int64)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    try:
        auroc = roc_auc_score(y_true, prob_pos)
    except ValueError:
        auroc = float("nan")

    try:
        auprc = average_precision_score(y_true, prob_pos)
    except ValueError:
        auprc = float("nan")

    return {
        "Accuracy": float(acc),
        "AUROC": float(auroc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1 score": float(f1),
        "AUPRC": float(auprc),
    }


def compute_metrics_multiclass(y_true: np.ndarray, prob: np.ndarray) -> dict:
    K = prob.shape[1]
    y_pred = prob.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    y_oh = label_binarize(y_true, classes=list(range(K)))
    try:
        auroc = roc_auc_score(y_oh, prob, multi_class="ovr", average="macro")
    except ValueError:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_oh, prob, average="macro")
    except ValueError:
        auprc = float("nan")

    return {
        "Accuracy": float(acc),
        "AUROC": float(auroc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1 score": float(f1),
        "AUPRC": float(auprc),
    }


def _to_grayscale(img: torch.Tensor) -> torch.Tensor:
    if img.dim() != 3:
        return img
    if img.shape[0] == 1:
        return img
    if img.shape[0] == 3:
        return TF.rgb_to_grayscale(img)
    return img.mean(dim=0, keepdim=True)


def collate_flatten(batch):
    xs = []
    ys = []
    for item in batch:
        if len(item) == 3:
            img, y, _ = item
        else:
            img, y = item
        img = _to_grayscale(img)
        if torch.is_tensor(y):
            y = int(y.view(-1)[0].item())
        else:
            y = int(y)
        xs.append(img.view(-1))
        ys.append(y)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


@torch.no_grad()
def compute_loss(model: nn.Module, loader, device: torch.device, loss_fn):
    if len(loader.dataset) == 0:
        return float("nan")
    model.eval()
    losses = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def resolve_split_counts(args, num_classes: int):
    if any(c is not None for c in (args.train_count, args.val_count, args.test_count)):
        if not all(c is not None for c in (args.train_count, args.val_count, args.test_count)):
            raise ValueError("train_count, val_count, and test_count must all be set when using count-based split.")
        return args.train_count, args.val_count, args.test_count, args.samples_per_label

    train_count = args.train_per_class * num_classes
    val_count = args.val_per_class * num_classes
    test_count = args.test_per_class * num_classes
    samples_per_label = args.samples_per_label
    if samples_per_label is None:
        samples_per_label = args.train_per_class + args.val_per_class + args.test_per_class
    return train_count, val_count, test_count, samples_per_label



# QSAN Model

class QSANModel(nn.Module):
    def __init__(self, n_qubits: int, n_layers: int, num_classes: int, shots: int = 0):
        super().__init__()
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.shots = None if shots == 0 else int(shots)
        self.q_wires = list(range(0, n_qubits))
        self.k_wires = list(range(n_qubits, 2 * n_qubits))
        self.work = 2 * n_qubits
        self.out = 2 * n_qubits + 1
        self.total_wires = 2 * n_qubits + 2

        self.dev_qls = qml.device("lightning.qubit", wires=self.total_wires, shots=self.shots)
        self.dev_val = qml.device("lightning.qubit", wires=n_qubits, shots=self.shots)

        # Trainable circuit weights (torch parameters)
        wshape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.w_q = nn.Parameter(0.01 * torch.randn(*wshape))
        self.w_k = nn.Parameter(0.01 * torch.randn(*wshape))
        self.w_v = nn.Parameter(0.01 * torch.randn(*wshape))

        # Classification head
        self.cls = nn.Linear(n_qubits, num_classes)

        @qml.qnode(self.dev_qls, interface="torch", diff_method="adjoint")
        def qls_prob(xq, xk, wq, wk):
            qml.AmplitudeEmbedding(xq, wires=self.q_wires, pad_with=0.0, normalize=True)
            qml.StronglyEntanglingLayers(wq, wires=self.q_wires)

            qml.AmplitudeEmbedding(xk, wires=self.k_wires, pad_with=0.0, normalize=True)
            qml.StronglyEntanglingLayers(wk, wires=self.k_wires)

            for qw, kw in zip(self.q_wires, self.k_wires):
                qml.Toffoli(wires=[qw, kw, self.work])
                qml.CNOT(wires=[self.work, self.out])
                qml.Toffoli(wires=[qw, kw, self.work])

            return qml.expval(qml.PauliZ(self.out))

        @qml.qnode(self.dev_val, interface="torch", diff_method="adjoint")
        def value_expvals(x, wv):
            qml.AmplitudeEmbedding(x, wires=range(n_qubits), pad_with=0.0, normalize=True)
            qml.StronglyEntanglingLayers(wv, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qls_prob = qls_prob
        self.value_expvals = value_expvals

    def forward(self, x_vec: torch.Tensor) -> torch.Tensor:
        B, D = x_vec.shape
        x_vec = x_vec.float()

        v_list = []
        p_list = []
        for b in range(B):
            vb = self.value_expvals(x_vec[b], self.w_v)
            vb = torch.stack(vb).float()
            v_list.append(vb)

            ez = self.qls_prob(x_vec[b], x_vec[b], self.w_q, self.w_k)
            pb = 0.5 * (1.0 - ez)
            p_list.append(pb.float())

        V = torch.stack(v_list, dim=0)       # [B, n_qubits]
        P = torch.stack(p_list, dim=0)       # [B]
        gated = V * P.unsqueeze(-1)          # [B, n_qubits]

        logits = self.cls(gated)             # [B, C]
        return logits



# Train/Eval

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, num_classes: int):
    model.eval()
    ys = []
    probs = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pr = F.softmax(logits, dim=1)
        ys.append(yb.detach().cpu().numpy())
        probs.append(pr.detach().cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    prob = np.concatenate(probs, axis=0)

    if num_classes == 2:
        return compute_metrics_binary(y_true, prob[:, 1])
    return compute_metrics_multiclass(y_true, prob)


def run_experiment(args, dataset_choice: str, lr: float, batch_size: int):
    seed_all(args.seed)

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    K = len(args.classes)
    if K < 2:
        raise ValueError("Need at least 2 classes.")

    train_count, val_count, test_count, samples_per_label = resolve_split_counts(args, K)

    (_, train_set), (_, val_set), (_, test_set) = build_standard_loaders(
        dataset_choice=dataset_choice,
        image_size=args.img_side,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        seed=args.seed,
        dataset_labels=args.classes,
        samples_per_label=samples_per_label,
        binary_root=args.binary_data_root,
        multi_root=args.multi_data_root,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_flatten,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_flatten,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_flatten,
    )

    feature_dim = args.img_side * args.img_side
    n_qubits = int(round(math.log2(feature_dim)))
    if 2 ** n_qubits != feature_dim:
        raise ValueError(
            f"img_side^2 ({feature_dim}) must be power of two for amplitude embedding (e.g. 4x4 -> 16)."
        )

    model = QSANModel(n_qubits=n_qubits, n_layers=args.q_layers, num_classes=K, shots=args.shots).to(device)

    opt = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    use_val = len(val_set) > 0
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    print(f"[QSAN] dataset={dataset_choice} classes={args.classes} K={K}")
    print(f"[QSAN] train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    print(f"[QSAN] resize={args.img_side}x{args.img_side} -> dim={feature_dim} qubits={n_qubits}")
    print(f"[QSAN] epochs={args.epochs} batch={batch_size} lr={lr} nesterov(m={args.momentum}) device={device}")

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val_loss = compute_loss(model, val_loader, device, loss_fn) if use_val else float("nan")
        if use_val:
            if val_loss + args.early_stop_delta < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= args.early_stop_patience:
                print(f"[early-stop] no val improvement for {args.early_stop_patience} epochs.")
                break

        if ep == 1 or ep % args.eval_interval == 0 or ep == args.epochs:
            tr_m = evaluate(model, train_loader, device, num_classes=K)
            te_m = evaluate(model, test_loader, device, num_classes=K)
            print(f"[epoch {ep:03d}] loss={float(np.mean(losses)):.4f} val_loss={val_loss:.4f}")
            print("  train:", " | ".join(f"{k}={v:.4f}" for k, v in tr_m.items()))
            print("  test :", " | ".join(f"{k}={v:.4f}" for k, v in te_m.items()))

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    else:
        model.load_state_dict(best_state)

    print("\n[Final Test Metrics]")
    final = evaluate(model, test_loader, device, num_classes=K)
    for k, v in final.items():
        print(f"{k:<10}: {v:.6f}")

    return {
        "dataset": dataset_choice,
        "lr": lr,
        "batch_size": batch_size,
        "best_val_loss": best_val_loss if use_val else float("nan"),
        "test_metrics": final,
    }


def _cuda_visible_from_device(device: str) -> str | None:
    if not device:
        return None
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    if device.startswith("cuda"):
        return device.replace("cuda", "")
    if device.isdigit():
        return device
    return None


def _build_child_command(args, dataset_choice: str, lr: float, batch_size: int) -> list[str]:
    script = Path(__file__).resolve()
    cmd = [sys.executable, str(script)]
    cmd += ["--dataset", dataset_choice]
    labels = resolve_classes(dataset_choice, args.classes)
    if labels:
        cmd += ["--classes"] + [str(x) for x in labels]
    if args.samples_per_label is not None:
        cmd += ["--samples-per-label", str(args.samples_per_label)]
    cmd += ["--img-side", str(args.img_side)]
    if args.train_count is not None:
        cmd += ["--train-count", str(args.train_count)]
    if args.val_count is not None:
        cmd += ["--val-count", str(args.val_count)]
    if args.test_count is not None:
        cmd += ["--test-count", str(args.test_count)]
    cmd += ["--seed", str(args.seed)]
    cmd += ["--q-layers", str(args.q_layers)]
    cmd += ["--shots", str(args.shots)]
    cmd += ["--epochs", str(args.epochs)]
    cmd += ["--batch-size", str(batch_size)]
    cmd += ["--lr", str(lr)]
    cmd += ["--eval-interval", str(args.eval_interval)]
    cmd += ["--early-stop-patience", str(args.early_stop_patience)]
    cmd += ["--early-stop-delta", str(args.early_stop_delta)]
    cmd += ["--momentum", str(args.momentum)]
    cmd += ["--weight-decay", str(args.weight_decay)]
    cmd += ["--binary-data-root", str(args.binary_data_root)]
    cmd += ["--multi-data-root", str(args.multi_data_root)]
    return cmd


def _terminate_procs(procs: list[tuple[subprocess.Popen, str, str, Path]]):
    for proc, _, _, _ in procs:
        try:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    proc.terminate()
        except Exception:
            pass


def run_grid_parallel(args):
    devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    if len(args.grid_datasets) > len(devices):
        raise ValueError(f"grid_datasets has {len(args.grid_datasets)} entries but only {len(devices)} devices.")

    log_dir = Path(os.environ.get("QSAN_LOG_DIR", Path(__file__).resolve().parents[1] / "results" / "logs"))
    for lr in args.grid_lrs:
        for bs in args.grid_batch_sizes:
            procs = []

            def _handle_signal(signum, frame):
                _terminate_procs(procs)
                raise SystemExit(1)

            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)
            atexit.register(_terminate_procs, procs)

            for dataset_choice, device in zip(args.grid_datasets, devices):
                cmd = _build_child_command(args, dataset_choice, lr, bs)
                env = os.environ.copy()
                visible = _cuda_visible_from_device(device)
                if visible is not None:
                    env["CUDA_VISIBLE_DEVICES"] = visible
                tag = time.strftime("%Y%m%d-%H%M%S")
                env["QSAN_LOG_TAG"] = tag
                prefix = f"qsan_{dataset_choice}_lr{lr}_bs{bs}"
                env["QSAN_LOG_PREFIX"] = prefix
                log_path = log_dir / f"{prefix}_{tag}.log"
                print(f"[grid-start] dataset={dataset_choice} device={device} lr={lr} batch={bs} log={log_path}")
                proc = subprocess.Popen(cmd, env=env, start_new_session=True)
                procs.append((proc, dataset_choice, device, log_path))

            failed = False
            for proc, dataset_choice, device, log_path in procs:
                code = proc.wait()
                print(f"[grid-end] dataset={dataset_choice} device={device} exit={code} log={log_path}")
                if code != 0:
                    failed = True
            if failed:
                print(f"[grid-warn] one or more runs failed for lr={lr} batch={bs}")


def resolve_classes(dataset_choice: str, classes: list[int] | None) -> list[int]:
    if classes is not None:
        return classes
    return [1, 7] if dataset_choice.lower() == "mnist" else [0, 1]


def main():
    p = argparse.ArgumentParser("QSAN IEEE-style (4x4 resize) simulator training")

    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fmnist", "cifar10", "pcam"])
    p.add_argument(
        "--binary-data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "Binary_dataset",
        help="Root directory for Binary_dataset splits",
    )
    p.add_argument(
        "--multi-data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "Multi_dataset",
        help="Root directory for Multi_dataset splits",
    )
    p.add_argument("--classes", type=int, nargs="+", default=None, help="labels to keep (remapped to 0..K-1)")
    p.add_argument("--samples-per-label", type=int, default=10000, help="cap samples per label before split")
    p.add_argument("--train-per-class", type=int, default=50)
    p.add_argument("--val-per-class", type=int, default=30)
    p.add_argument("--test-per-class", type=int, default=30)
    p.add_argument("--train-count", type=int, default=None)
    p.add_argument("--val-count", type=int, default=None)
    p.add_argument("--test-count", type=int, default=None)

    p.add_argument("--img-side", type=int, default=4, help="paper uses 4x4 resizing")
    p.add_argument("--batch-size", type=int, default=15)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=5)

    # quantum
    p.add_argument("--q-layers", type=int, default=1)
    p.add_argument("--shots", type=int, default=0, help="0=analytic, else finite shots")

    # optimizer (Nesterov)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--early-stop-patience", type=int, default=5)
    p.add_argument("--early-stop-delta", type=float, default=0.0)

    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")

    p.add_argument("--grid-search", action="store_true")
    p.add_argument("--grid-parallel", action="store_true", help="run grid datasets in parallel on cuda:0..3")
    p.add_argument("--grid-datasets", type=str, nargs="+", default=["mnist", "fmnist", "cifar10", "pcam"])
    p.add_argument("--grid-lrs", type=float, nargs="+", default=[0.01, 0.05])
    p.add_argument("--grid-batch-sizes", type=int, nargs="+", default=[16])

    args = p.parse_args()
    log_prefix = "qsan_grid" if args.grid_search else f"qsan_{args.dataset}"
    log_path = setup_logging(log_prefix)
    print(f"[log] saving to {log_path}")

    if args.grid_search and args.grid_parallel:
        run_grid_parallel(args)
    elif args.grid_search:
        results = []
        for dataset_choice in args.grid_datasets:
            for lr in args.grid_lrs:
                for bs in args.grid_batch_sizes:
                    print("\n==============================")
                    print(f"[grid] dataset={dataset_choice} lr={lr} batch={bs}")
                    args.classes = resolve_classes(dataset_choice, args.classes)
                    results.append(run_experiment(args, dataset_choice, lr, bs))
        print("\n==== Grid Summary (best val loss) ====")
        for r in results:
            acc = r["test_metrics"]["Accuracy"]
            print(
                f"{r['dataset']} lr={r['lr']} batch={r['batch_size']} "
                f"best_val_loss={r['best_val_loss']:.4f} test_acc={acc:.4f}"
            )
    else:
        args.classes = resolve_classes(args.dataset, args.classes)
        run_experiment(args, args.dataset, args.lr, args.batch_size)


if __name__ == "__main__":
    main()
