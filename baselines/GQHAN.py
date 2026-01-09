
import os
import copy
import math
import time
import atexit
import subprocess
import signal
import random
import argparse
import sys
from pathlib import Path
from pathlib import Path
import sys
import numpy as np

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler  
torch.set_default_dtype(torch.float64)
import pennylane as qml

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data_loader import build_standard_loaders

# 0) Determinism / Seed
def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1) Data via QSANN_revision pipeline

def flatten_subset(subset):
    base = subset.dataset
    xs, ys = [], []
    for idx in subset.indices:
        img, lbl, _ = base[idx]
        if torch.is_tensor(img):
            arr = img.view(-1).numpy()
        else:
            arr = torch.tensor(img).view(-1).numpy()
        xs.append(arr.astype(np.float64))
        ys.append(int(lbl))
    return np.stack(xs), np.array(ys, dtype=np.int64)

def load_binary_from_pipeline(
    dataset_choice: str,
    dataset_labels,
    samples_per_label: int | None,
    image_size: int,
    train_count: int | None,
    val_count: int | None,
    test_count: int | None,
    seed: int,
    pca_dim: int = 8,
    binary_root: Path | None = None,
    multi_root: Path | None = None,
):
    (_, train_set), (_, val_set), (_, test_set) = build_standard_loaders(
        dataset_choice=dataset_choice,
        image_size=image_size,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        seed=seed,
        dataset_labels=dataset_labels,
        samples_per_label=samples_per_label,
        balance_sampler=False,
        binary_root=binary_root,
        multi_root=multi_root,
    )

    X_train_raw, y_train = flatten_subset(train_set)
    if len(val_set) > 0:
        X_val_raw, y_val = flatten_subset(val_set)
    else:
        X_val_raw = np.empty((0, X_train_raw.shape[1]), dtype=np.float64)
        y_val = np.empty((0,), dtype=np.int64)
    X_test_raw, y_test = flatten_subset(test_set)

    X_all_raw = np.concatenate([X_train_raw, X_val_raw, X_test_raw], axis=0)
    pca = PCA(n_components=pca_dim, random_state=seed)
    X_all_pca = pca.fit_transform(X_all_raw).astype(np.float64)

    scaler = MinMaxScaler(feature_range=(0.1, 1.0))
    X_all_scaled = scaler.fit_transform(X_all_pca)

    train_end = len(X_train_raw)
    val_end = train_end + len(X_val_raw)
    X_train = X_all_scaled[:train_end]
    X_val = X_all_scaled[train_end:val_end]
    X_test = X_all_scaled[val_end:]

    uniq = np.unique(np.concatenate([y_train, y_val, y_test]))
    if len(uniq) != 2:
        raise ValueError(f"Binary only; got labels {uniq}. Set --dataset-labels to two classes.")
    lo, hi = sorted(uniq.tolist())
    y_train_pm = np.where(y_train == lo, 1.0, -1.0).astype(np.float64)
    y_val_pm = np.where(y_val == lo, 1.0, -1.0).astype(np.float64)
    y_test_pm = np.where(y_test == lo, 1.0, -1.0).astype(np.float64)
    return X_train, y_train_pm, X_val, y_val_pm, X_test, y_test_pm

# 2) Quantum blocks: FO and ADO

def mcz_on_all_ones(controls, target):
    qml.ctrl(qml.PauliZ, control=controls)(wires=target)

def dp_controlled_phase_flip_for_bitstring(bitstring_int, data_wires, ancilla_wire):
    n = len(data_wires)
    bits = [(bitstring_int >> (n - 1 - i)) & 1 for i in range(n)]
    
    for w, b in zip(data_wires, bits):
        if b == 0: qml.PauliX(wires=w)
        
    controls = [ancilla_wire] + data_wires[:-1]
    target = data_wires[-1]
    mcz_on_all_ones(controls=controls, target=target)
    
    for w, b in zip(data_wires, bits):
        if b == 0: qml.PauliX(wires=w)

def flexible_oracle(theta_fo, ancilla_wire, data_wires):
    """ FO with Uncompute & RY """
    n = len(data_wires)
    for b in range(2 ** n):
        qml.RY(theta_fo[b], wires=ancilla_wire)
        dp_controlled_phase_flip_for_bitstring(b, data_wires=data_wires, ancilla_wire=ancilla_wire)
        qml.RY(-theta_fo[b], wires=ancilla_wire) 

def ado(theta_ado, data_wires):
    """ ADO with Pauli-X Sandwich """
    n = len(data_wires)
    
    # U1
    for w in data_wires: qml.Hadamard(wires=w)
    for i in range(n):
        qml.CRY(theta_ado[i], wires=[data_wires[i], data_wires[(i + 1) % n]])

    for w in data_wires: qml.PauliX(wires=w)
    mcz_on_all_ones(controls=data_wires[:-1], target=data_wires[-1])
    for w in data_wires: qml.PauliX(wires=w)

    for i in range(n):
        qml.CRY(theta_ado[i + n], wires=[data_wires[i], data_wires[(i + 1) % n]])
    for w in data_wires: qml.Hadamard(wires=w)


# 3) QNode / Model

class GQHAN(nn.Module):
    def __init__(self, seed=42, device_name="default.qubit"):
        super().__init__()
        self.seed = seed
        self.n = 3
        self.ancilla = 0
        self.data_wires = [1, 2, 3]

        gen = torch.Generator()
        gen.manual_seed(seed)

        self.theta_fo = nn.Parameter(
            (torch.rand(2 ** self.n, generator=gen) * 2 * np.pi) - np.pi
        )
        
        self.theta_ado = nn.Parameter(torch.zeros(2 * self.n))

        self.dev = qml.device(device_name, wires=1 + self.n)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x, theta_fo, theta_ado):
            qml.AmplitudeEmbedding(x, wires=self.data_wires, normalize=True)
            flexible_oracle(theta_fo, ancilla_wire=self.ancilla, data_wires=self.data_wires)
            ado(theta_ado, data_wires=self.data_wires)
            return qml.expval(qml.PauliZ(wires=self.data_wires[-1]))

        self.circuit = circuit

    def forward(self, x_batch):
        outs = []
        for i in range(x_batch.shape[0]):
            outs.append(self.circuit(x_batch[i], self.theta_fo, self.theta_ado))
        return torch.stack(outs, dim=0)


# 4) Train/Eval utilities

@torch.no_grad()
def accuracy_from_expval(expvals, y_pm):
    pred = torch.where(expvals >= 0, torch.tensor(1.0), torch.tensor(-1.0))
    return (pred == y_pm).float().mean().item()

def binary_metrics(expvals: torch.Tensor, y_pm: torch.Tensor):
    from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_recall_fscore_support
    y_true = (y_pm.cpu().numpy() > 0).astype(int)
    scores = expvals.cpu().numpy()
    y_pred = (scores >= 0).astype(int)
    metrics = {}
    metrics["acc"] = accuracy_score(y_true, y_pred)
    try: metrics["auroc"] = roc_auc_score(y_true, scores)
    except: metrics["auroc"] = float("nan")
    try: metrics["auprc"] = average_precision_score(y_true, scores)
    except: metrics["auprc"] = float("nan")
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    metrics.update({"precision": precision, "recall": recall, "f1": f1})
    return metrics

@torch.no_grad()
def eval_loss(model: nn.Module, x_t: torch.Tensor, y_t: torch.Tensor, loss_fn, batch_size: int):
    if x_t.numel() == 0:
        return float("nan")
    model.eval()
    losses = []
    for i in range(0, len(x_t), batch_size):
        xb = x_t[i : i + batch_size]
        yb = y_t[i : i + batch_size]
        out = model(xb)
        losses.append(loss_fn(out, yb).item())
    return float(np.mean(losses)) if losses else float("nan")

@torch.no_grad()
def eval_accuracy(model: nn.Module, x_t: torch.Tensor, y_t: torch.Tensor, batch_size: int):
    if x_t.numel() == 0:
        return float("nan")
    model.eval()
    outs = []
    for i in range(0, len(x_t), batch_size):
        xb = x_t[i : i + batch_size]
        outs.append(model(xb))
    expvals = torch.cat(outs, dim=0) if outs else torch.empty_like(y_t)
    return accuracy_from_expval(expvals, y_t)

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
    env_prefix = os.environ.get("GQHAN_LOG_PREFIX")
    if env_prefix:
        prefix = env_prefix
    log_dir = Path(os.environ.get("GQHAN_LOG_DIR", Path(__file__).resolve().parents[1] / "results" / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("GQHAN_LOG_TAG") or time.strftime("%Y%m%d-%H%M%S")
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
    cmd += ["--dataset-choice", dataset_choice]
    labels = args.dataset_labels
    if labels is None:
        labels = [1, 7] if dataset_choice.lower() == "mnist" else [0, 1]
    if labels:
        cmd += ["--dataset-labels"] + [str(x) for x in labels]
    if args.samples_per_label is not None:
        cmd += ["--samples-per-label", str(args.samples_per_label)]
    cmd += ["--image-size", str(args.image_size)]
    cmd += ["--train-count", str(args.train_count)]
    cmd += ["--val-count", str(args.val_count)]
    cmd += ["--test-count", str(args.test_count)]
    cmd += ["--seed", str(args.seed)]
    cmd += ["--pca-dim", str(args.pca_dim)]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    elif args.steps is not None:
        cmd += ["--steps", str(args.steps)]
    cmd += ["--batch-size", str(batch_size)]
    cmd += ["--lr", str(lr)]
    cmd += ["--eval-interval", str(args.eval_interval)]
    cmd += ["--early-stop-patience", str(args.early_stop_patience)]
    cmd += ["--early-stop-delta", str(args.early_stop_delta)]
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

    log_dir = Path(os.environ.get("GQHAN_LOG_DIR", Path(__file__).resolve().parents[1] / "results" / "logs"))
    for lr in args.grid_lrs:
        for bs in args.grid_batch_sizes:
            tag = time.strftime("%Y%m%d-%H%M%S")
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
                env["GQHAN_LOG_TAG"] = tag
                prefix = f"gqhan_{dataset_choice}_lr{lr}_bs{bs}"
                env["GQHAN_LOG_PREFIX"] = prefix
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

def qhas_mask(theta_fo):
    pi = np.pi
    t = theta_fo.detach().cpu().numpy()
    t_mod = (t + 2*pi) % (4*pi) - 2*pi
    dist = np.abs(t_mod - pi)
    mask = (dist < 0.2).astype(np.int32)
    return mask, t

def run_experiment(args, dataset_choice: str, lr: float, batch_size: int):
    if len(args.dataset_labels) != 2:
        raise ValueError("GQHAN supports binary classification only; set exactly two --dataset-labels.")

    set_all_seeds(args.seed)

    X_train, y_train, X_val, y_val, X_test, y_test = load_binary_from_pipeline(
        dataset_choice=dataset_choice,
        dataset_labels=args.dataset_labels,
        samples_per_label=args.samples_per_label,
        image_size=args.image_size,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
        pca_dim=args.pca_dim,
        binary_root=args.binary_data_root,
        multi_root=args.multi_data_root,
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float64)
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    X_val_t = torch.tensor(X_val, dtype=torch.float64)
    y_val_t = torch.tensor(y_val, dtype=torch.float64)
    X_test_t  = torch.tensor(X_test, dtype=torch.float64)
    y_test_t  = torch.tensor(y_test, dtype=torch.float64)

    if batch_size > len(X_train_t):
        raise ValueError(f"batch_size {batch_size} exceeds train size {len(X_train_t)}")

    model = GQHAN(seed=args.seed).double()

    # Nesterov momentum
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    loss_fn = nn.MSELoss()

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    steps_per_epoch = int(math.ceil(len(train_ds) / batch_size))
    if args.epochs is None:
        if args.steps is None:
            raise ValueError("Set --epochs or --steps.")
        epochs = max(1, int(math.ceil(args.steps / steps_per_epoch)))
    else:
        epochs = args.epochs

    eval_interval = max(1, args.eval_interval)
    use_val = len(X_val_t) > 0

    loss_hist = []
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    print(
        f"[GQHAN] dataset={dataset_choice} train={len(X_train_t)} val={len(X_val_t)} "
        f"test={len(X_test_t)} lr={lr} batch={batch_size} epochs={epochs}"
    )

    init_mask, _ = qhas_mask(model.theta_fo)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        loss_hist.append(train_loss)

        val_loss = eval_loss(model, X_val_t, y_val_t, loss_fn, batch_size) if use_val else float("nan")
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

        if epoch == 1 or epoch % eval_interval == 0 or epoch == epochs:
            train_acc = eval_accuracy(model, X_train_t, y_train_t, batch_size)
            test_acc = eval_accuracy(model, X_test_t, y_test_t, batch_size)
            print(
                f"[epoch {epoch:03d}/{epochs}] loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} train_acc={train_acc*100:.2f}% test_acc={test_acc*100:.2f}%"
            )

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    else:
        model.load_state_dict(best_state)

    # Final Stats
    last10_loss = np.mean(loss_hist[-10:]) if loss_hist else float("nan")
    train_acc = eval_accuracy(model, X_train_t, y_train_t, batch_size)
    test_acc = eval_accuracy(model, X_test_t, y_test_t, batch_size)

    print("\n==== Summary ====")
    print(f"Train Acc: {train_acc*100:.2f}%")
    print(f"Test  Acc: {test_acc*100:.2f}%")
    print(f"Loss (last10): {last10_loss:.4f}")
    if use_val:
        print(f"Best Val Loss: {best_val_loss:.4f}")

    with torch.no_grad():
        test_out = model(X_test_t)
    m = binary_metrics(test_out, y_test_t)
    print("\n==== Test metrics ====")
    print(
        f"accuracy={m['acc']:.4f} auroc={m['auroc']:.4f} auprc={m['auprc']:.4f} "
        f"precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f}"
    )

    print("\n==== QHAS ====")
    final_mask, _ = qhas_mask(model.theta_fo)
    print("Init Mask:", init_mask)
    print("Final Mask:", final_mask)

    summary = {
        "dataset": dataset_choice,
        "lr": lr,
        "batch_size": batch_size,
        "best_val_loss": best_val_loss if use_val else float("nan"),
        "test_metrics": m,
    }
    return summary

def main():
    parser = argparse.ArgumentParser(description="GQHAN Final Complete")
    parser.add_argument("--dataset-choice", type=str, default="")
    parser.add_argument("--dataset-labels", type=int, nargs="*", default=None)
    parser.add_argument("--samples-per-label", type=int, default=10000)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--val-count", type=int, required=True)
    parser.add_argument("--test-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides --steps).")
    parser.add_argument("--steps", type=int, default=120, help="Deprecated; used to derive epochs if --epochs is unset.")
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.09)
    parser.add_argument("--eval-interval", type=int, default=10, help="Eval/print interval in epochs.")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-delta", type=float, default=0.0)
    parser.add_argument(
        "--binary-data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "Binary_dataset",
        help="Root directory for Binary_dataset splits",
    )
    parser.add_argument(
        "--multi-data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "Multi_dataset",
        help="Root directory for Multi_dataset splits",
    )
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--grid-datasets", type=str, nargs="+", default=["mnist", "fmnist", "cifar10", "pcam"])
    parser.add_argument("--grid-lrs", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument("--grid-batch-sizes", type=int, nargs="+", default=[16])
    args = parser.parse_args()
    if args.dataset_labels is None:
        if args.dataset_choice.lower() == "mnist":
            args.dataset_labels = [1, 7]
        else:
            args.dataset_labels = [0, 1]

    log_prefix = "gqhan_grid" if args.grid_search else f"gqhan_{args.dataset_choice}"
    log_path = setup_logging(log_prefix)
    print(f"[log] saving to {log_path}")

    if args.grid_search:
        run_grid_parallel(args)
    else:
        run_experiment(args, args.dataset_choice, args.lr, args.batch_size)

if __name__ == "__main__":
    main()
