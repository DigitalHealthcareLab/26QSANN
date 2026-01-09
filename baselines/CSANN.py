"""
Simple self-attention classifier extracted from the JYJ CSANN notebook.
- Uses patch embeddings -> multihead self-attention -> flatten -> classifier.
- Data loading/splitting matches QKSAN_reference/GQHAN.py via build_standard_loaders.
- Hyperparameters mirror run.sh/main.py (dataset/labels/counts, image/patch sizes, attn layers, hidden dims, lr, early stop, logging).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import copy
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn, optim
from torch.utils.data import Dataset

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data_loader import build_standard_loaders


# Utilities

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu" or not torch.cuda.is_available():
        if requested != "cpu" and not torch.cuda.is_available():
            print(f"[device] CUDA unavailable; falling back to CPU (requested '{requested}').")
        return torch.device("cpu")
    try:
        idx = int(requested.split(":")[1]) if ":" in requested else 0
    except (IndexError, ValueError):
        idx = 0
    gpu_count = torch.cuda.device_count()
    if idx >= gpu_count:
        print(f"[device] Requested CUDA device '{requested}' not found (have {gpu_count}); using cuda:0.")
        return torch.device("cuda:0")
    return torch.device(requested)


def infer_num_classes(train_set: Dataset, override: int) -> int:
    base = train_set.dataset if hasattr(train_set, "dataset") else train_set
    labels = getattr(base, "labels", None)
    if labels is None or len(labels) == 0:
        raise ValueError("Dataset must expose labels for class inference.")
    inferred = len(set(int(l) for l in labels))
    if override > 0:
        return override
    return inferred


def infer_in_channels(train_set: Dataset) -> int:
    sample, _, _ = train_set[0]
    if sample.dim() != 3:
        raise ValueError(f"Expected sample with 3 dims [C,H,W]; got shape {sample.shape}")
    return sample.shape[0]


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def dataset_tag(args: argparse.Namespace) -> str:
    tag = args.dataset_choice
    if args.dataset_labels:
        label_tag = "-".join(str(l) for l in args.dataset_labels)
        tag = f"{tag}[{label_tag}]"
    if args.samples_per_label:
        tag = f"{tag}-n{args.samples_per_label}"
    return tag


def make_run_name(prefix: str, args: argparse.Namespace) -> str:
    if args.run_name != "auto":
        return args.run_name
    ts = time.strftime("%Y%m%d-%H%M%S")
    ds_tag = dataset_tag(args)
    return f"{prefix}-{ds_tag}-N{args.train_count or 'auto'}-{ts}"


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> Dict[str, float]:
    labels_np = labels.cpu().numpy()
    if num_classes == 2:
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(labels_np, preds)
        try:
            auroc = roc_auc_score(labels_np, probs)
        except ValueError:
            auroc = float("nan")
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np, preds, average="binary", zero_division=0
        )
        try:
            auprc = average_precision_score(labels_np, probs)
        except ValueError:
            auprc = float("nan")
    else:
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        acc = accuracy_score(labels_np, preds)
        try:
            onehot = torch.nn.functional.one_hot(labels.long(), num_classes=num_classes).cpu().numpy()
            auroc = roc_auc_score(onehot, probs, multi_class="ovr")
            auprc = average_precision_score(onehot, probs, average="macro")
        except ValueError:
            auroc = float("nan")
            auprc = float("nan")
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np, preds, average="macro", zero_division=0
        )
    return {
        "acc": acc,
        "auroc": auroc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auprc": auprc,
    }


def evaluate(model: nn.Module, loader, criterion, device: torch.device, num_classes: int) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.long().to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    avg_loss = total_loss / max(1, total)
    if all_logits:
        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        metrics = compute_metrics(logits_cat, labels_cat, num_classes)
    else:
        metrics = {k: float("nan") for k in ["acc", "auroc", "precision", "recall", "f1", "auprc"]}
    return avg_loss, metrics



# Model

class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"Image dims must be divisible by patch_size={self.patch_size} (got {h}x{w}).")
        ph = h // self.patch_size
        pw = w // self.patch_size
        patches = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .contiguous()
            .view(b, c, ph, pw, self.patch_size, self.patch_size)
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(b, ph * pw, -1)
        return self.proj(patches)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int, dropout: float) -> None:
        super().__init__()
        if embed_size % heads != 0:
            raise ValueError("embed_size must be divisible by heads.")
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)
        self.out_proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, seq_len, _ = x.shape
        q = self.q_proj(x).view(b, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, seq_len, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, seq_len, self.embed_size)
        out = self.out_proj(out)
        return out, attn


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_size: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_size, heads, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size * 2, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff = self.ffn(x)
        x = self.norm2(x + self.dropout(ff))
        return x, attn


class PatchSelfAttentionClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        embed_size: int,
        heads: int,
        attn_layers: int,
        hidden_dims: Sequence[int],
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size must be divisible by patch_size (got {image_size}, patch {patch_size}).")
        self.grid_size = image_size // patch_size
        self.seq_len = self.grid_size * self.grid_size
        self.embed = PatchEmbed(in_channels, patch_size, embed_size)
        self.blocks = nn.ModuleList(
            SelfAttentionBlock(embed_size, heads, dropout) for _ in range(attn_layers)
        )

        head_layers = []
        in_dim = embed_size * self.seq_len
        for hd in hidden_dims:
            head_layers.append(nn.Linear(in_dim, hd))
            head_layers.append(nn.ReLU())
            if dropout > 0:
                head_layers.append(nn.Dropout(dropout))
            in_dim = hd
        head_layers.append(nn.Linear(in_dim, num_classes))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x, _ = block(x)
        x = x.flatten(start_dim=1)
        return self.head(x)



# Training

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-attention baseline (CSANN notebook replica)")

    # Data / splits
    parser.add_argument("--dataset-choice", type=str, default="mnist", choices=["mnist", "fmnist", "cifar10", "pcam"])
    parser.add_argument("--dataset-labels", type=int, nargs="*", help="Optional list of labels to keep.")
    parser.add_argument("--samples-per-label", type=int, help="Per-label cap; also used as per-class cap for PCam.")
    parser.add_argument(
        "--binary-data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "Binary_dataset",
        help="Root directory for Binary_dataset splits.",
    )
    parser.add_argument(
        "--multi-data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "Multi_dataset",
        help="Root directory for Multi_dataset splits.",
    )
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--train-count", type=int, default=320, help="Absolute train size.")
    parser.add_argument("--val-count", type=int, default=0, help="Absolute val size.")
    parser.add_argument("--test-count", type=int, default=80, help="Absolute test size.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--balance-sampler", action="store_true", help="Enable class-balanced sampler for training loader.")

    # Model hyperparameters
    parser.add_argument("--embed-size", type=int, default=16)
    parser.add_argument("--attn-heads", type=int, default=2)
    parser.add_argument("--attn-layers", type=int, default=1)
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[], help="Classifier MLP hidden dims; empty for direct logits.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-classes", type=int, default=0, help="Override inferred class count (<=0 means infer).")

    # Optimization / training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)

    # Logging / checkpoints
    parser.add_argument("--log-dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--model-dir", type=Path, default=Path("results/models"))
    parser.add_argument("--run-name", type=str, default="auto", help="If 'auto', name encodes dataset/count/timestamp.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_labels is None and args.dataset_choice in ("mnist", "fmnist", "cifar10"):
        args.dataset_labels = list(range(10))

    set_all_seeds(args.seed)
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"

    (train_loader, train_set), (val_loader, val_set), (test_loader, test_set) = build_standard_loaders(
        dataset_choice=args.dataset_choice,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
        dataset_labels=args.dataset_labels,
        samples_per_label=args.samples_per_label,
        balance_sampler=args.balance_sampler,
        binary_root=args.binary_data_root,
        multi_root=args.multi_data_root,
    )

    base_dataset = train_set.dataset if hasattr(train_set, "dataset") else train_set
    label_counts = Counter(getattr(base_dataset, "labels", []))
    num_classes = infer_num_classes(train_set, args.num_classes)
    is_binary = num_classes == 2

    in_channels = infer_in_channels(train_set)
    model = PatchSelfAttentionClassifier(
        in_channels=in_channels,
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_size=args.embed_size,
        heads=args.attn_heads,
        attn_layers=args.attn_layers,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    run_name = make_run_name("selfattn", args)
    ensure_dirs(args.log_dir, args.model_dir)
    log_path = Path(args.log_dir) / f"{run_name}.log"

    def log(msg: str) -> None:
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"[run {run_name}] dataset={dataset_tag(args)} labels={dict(label_counts)}")
    log(
        f"[run {run_name}] image_size={args.image_size} patch_size={args.patch_size} "
        f"train/val/test={len(train_set)}/{len(val_set)}/{len(test_set)} "
        f"batch={args.batch_size} device={device}"
    )
    log(
        f"[run {run_name}] split_mode=count counts={args.train_count}/{args.val_count}/{args.test_count}"
    )
    log(
        f"[run {run_name}] model embed_size={args.embed_size} heads={args.attn_heads} "
        f"attn_layers={args.attn_layers} hidden_dims={args.hidden_dims} num_classes={num_classes}"
    )
    total_params, trainable_params = count_parameters(model)
    log(f"[run {run_name}] params total={total_params} trainable={trainable_params}")
    if is_binary:
        log(f"[run {run_name}] Binary mode enabled.")

    best_val = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
        log(
            f"[{run_name}] epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['acc']:.4f}"
        )

        if len(val_set) > 0:
            if val_loss + args.early_stop_min_delta < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
                save_path = Path(args.model_dir) / f"{run_name}_best.pt"
                torch.save(best_state, save_path)
                log(f"[{run_name}] New best val_loss={best_val:.4f}; saved to {save_path}")
            else:
                patience += 1
                if args.early_stop and patience >= args.early_stop_patience:
                    log(f"[{run_name}] Early stopping triggered.")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        log(f"[{run_name}] Restored best validation checkpoint.")

    test_loss, test_metrics = evaluate(model, test_loader, criterion, device, num_classes)
    log(
        f"[{run_name}] Test -> loss={test_loss:.4f} acc={test_metrics['acc']:.4f} "
        f"auroc={test_metrics['auroc']:.4f} precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f} "
        f"auprc={test_metrics['auprc']:.4f}"
    )

    final_path = Path(args.model_dir) / f"{run_name}_final.pt"
    torch.save(model.state_dict(), final_path)
    log(f"[{run_name}] Saved final model to {final_path}")


if __name__ == "__main__":
    main()
