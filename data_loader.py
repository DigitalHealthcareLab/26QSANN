from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


def _normalize_image_size(image_size: Sequence[int] | int) -> Tuple[int, int]:
    if isinstance(image_size, int):
        return (image_size, image_size)
    if len(image_size) != 2:
        raise ValueError("image_size must be an int or a length-2 sequence")
    return (int(image_size[0]), int(image_size[1]))


def _cifar10_rgb_to_grayscale(image_array: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale using standard luminance weights.
    """
    if image_array.ndim == 2:
        gray = image_array.astype(np.float32)
    else:
        gray = np.dot(image_array[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    return (gray / 255.0).astype(np.float32)



def _stratified_split_indices_counts(
    labels: Sequence[int],
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    total = len(labels)
    if total == 0:
        raise ValueError("Dataset is empty; cannot split.")
    if len(set(labels)) < 2:
        raise ValueError("Stratified split requires at least two classes.")
    if train_count < 0 or val_count < 0 or test_count < 0:
        raise ValueError("train_count, val_count, and test_count must be >= 0.")
    if train_count == 0:
        raise ValueError("train_count must be > 0 for count-based split.")
    total_target = train_count + val_count + test_count
    if total_target == 0:
        raise ValueError("At least one of train_count/val_count/test_count must be > 0.")
    if total_target > total:
        raise ValueError("train_count + val_count + test_count exceeds dataset size.")

    indices = np.arange(total)
    temp_count = val_count + test_count
    if temp_count == 0:
        if train_count == total:
            return indices.tolist(), [], []
        train_idx, _, _, _ = train_test_split(
            indices,
            labels,
            train_size=train_count,
            stratify=labels,
            random_state=seed,
            shuffle=True,
        )
        return train_idx.tolist(), [], []

    train_idx, temp_idx, train_y, temp_y = train_test_split(
        indices,
        labels,
        train_size=train_count,
        test_size=temp_count,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )

    if val_count == 0:
        return train_idx.tolist(), [], temp_idx.tolist()
    if test_count == 0:
        return train_idx.tolist(), temp_idx.tolist(), []

    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        temp_y,
        train_size=val_count,
        test_size=test_count,
        stratify=temp_y,
        random_state=seed,
        shuffle=True,
    )
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def split_dataset_by_counts(
    dataset: Dataset,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    labels = getattr(dataset, "labels", None)
    if labels is None:
        raise ValueError("Dataset must expose a 'labels' attribute for stratified split.")
    train_idx, val_idx, test_idx = _stratified_split_indices_counts(
        labels, train_count, val_count, test_count, seed
    )
    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


class TensorListDataset(Dataset):
    """
    Simple dataset wrapping a list of tensors and integer labels.
    """

    def __init__(self, images: List[torch.Tensor], labels: List[int]) -> None:
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length")
        if len(images) == 0:
            raise ValueError("images list is empty")
        self.images = images
        self.labels = [int(l) for l in labels]
        self.num_channels = images[0].shape[0]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx], f"sample_{idx}"


def _normalize_npz_data(data: np.ndarray) -> np.ndarray:
    if data.ndim == 3:
        data = data[:, np.newaxis, :, :]
    elif data.ndim == 4:
        if data.shape[1] not in (1, 2, 3) and data.shape[-1] in (1, 2, 3):
            data = np.transpose(data, (0, 3, 1, 2))
    else:
        raise ValueError(f"Expected npz data with 3 or 4 dims, got shape {data.shape}")
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / 255.0
    else:
        data = data.astype(np.float32, copy=False)
    return data


class NpzDataset(Dataset):
    """
    Dataset wrapper for pre-split npz files containing (data, labels).
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        name_prefix: str = "sample",
        resize_to: Sequence[int] | int | None = None,
    ) -> None:
        data = _normalize_npz_data(data)
        if len(labels) != len(data):
            raise ValueError("data and labels must have the same length")
        self.images = torch.from_numpy(data)
        self.labels = [int(l) for l in labels.tolist()]
        self.targets = self.labels
        self.num_channels = self.images.shape[1]
        self.name_prefix = name_prefix
        self.resize_to = _normalize_image_size(resize_to) if resize_to is not None else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        if self.resize_to is not None and img.shape[-2:] != self.resize_to:
            img = TF.resize(img, self.resize_to)
        return img, self.labels[idx], f"{self.name_prefix}_{idx}"


class PCamDataset(Dataset):
    """
    PatchCamelyon (Camelyon16) patches stored as .npy under class0/class1 folders.
    Expects data shaped [H, W, C] (C can be 1, 2, or 3); scales to [0, 1].
    """

    CLASS_MAP = [("class0", 0), ("class1", 1)]

    def __init__(
        self,
        root: str | Path,
        image_size: Sequence[int] | int,
        samples_per_class: int | None = None,
        seed: int = 42,
        force_grayscale: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_size = _normalize_image_size(image_size)
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.force_grayscale = force_grayscale
        self.items = self._gather_items()
        self.labels = [lbl for _, lbl in self.items]
        self.targets = self.labels
        self.num_channels = self._infer_channels()

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
            raise FileNotFoundError(f"No .npy files found under {self.root}. Expected class0/ and class1/ subfolders.")
        rng.shuffle(items)
        return items

    def _infer_channels(self) -> int:
        sample = np.load(self.items[0][0])
        if sample.ndim == 2:
            return 1
        if sample.ndim == 3:
            return sample.shape[-1]
        raise ValueError(f"Unexpected PCam sample shape {sample.shape}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        if arr.shape[2] not in (1, 2, 3):
            raise ValueError(f"Unsupported channel count {arr.shape[2]} in sample {path}")
        img = torch.from_numpy(arr).permute(2, 0, 1).float()
        if img.max() > 1.0:
            img = img / 255.0
        img = TF.resize(img, self.image_size)
        if self.force_grayscale and img.shape[0] == 3:
            img = TF.rgb_to_grayscale(img)
        return img, label, path.stem


class FilteredRemappedDataset(Dataset):
    """
    Dataset wrapper that filters by class, applies per-class limits, and remaps labels to 0..K-1.
    """

    def __init__(
        self,
        base: Dataset,
        indices: List[int],
        labels: List[int],
        label_map: dict[int, int],
    ) -> None:
        self.base = base
        self.indices = indices
        self.labels = labels
        self.targets = labels
        self.label_map = label_map
        if hasattr(base, "num_channels"):
            self.num_channels = base.num_channels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        item = self.base[base_idx]
        if isinstance(item, tuple):
            if len(item) >= 2:
                img = item[0]
                rest = item[2:] if len(item) > 2 else ()
                mapped = self.labels[idx]
                if rest:
                    return (img, mapped, *rest)
                return (img, mapped)
        return item


def filter_remap_dataset(
    ds: Dataset,
    classes: Sequence[int],
    per_class_limit: int | None,
    seed: int,
) -> FilteredRemappedDataset:
    if isinstance(classes, np.ndarray):
        classes = classes.tolist()
    allowed = sorted({int(c) for c in classes})
    allowed_set = set(allowed)
    label_map = {orig: idx for idx, orig in enumerate(allowed)}

    targets = getattr(ds, "labels", None)
    if targets is None:
        targets = getattr(ds, "targets", None)
    if targets is None:
        raise ValueError("Dataset must expose labels or targets for filtering.")

    if torch.is_tensor(targets):
        y = targets.detach().cpu().numpy().astype(int).tolist()
    elif isinstance(targets, np.ndarray):
        y = targets.astype(int).tolist()
    else:
        y = [int(v) for v in targets]

    rng = np.random.default_rng(seed)
    by_class: dict[int, List[int]] = {c: [] for c in allowed}
    for idx, lbl in enumerate(y):
        if lbl in allowed_set:
            by_class[lbl].append(idx)

    indices: List[int] = []
    labels: List[int] = []
    for orig in allowed:
        idxs = by_class.get(orig, [])
        if per_class_limit is not None and len(idxs) > per_class_limit:
            rng.shuffle(idxs)
            idxs = idxs[:per_class_limit]
        indices.extend(idxs)
        labels.extend([label_map[orig]] * len(idxs))

    if not indices:
        raise ValueError(f"No samples found for classes {allowed}.")

    order = np.arange(len(indices))
    rng.shuffle(order)
    indices = [indices[i] for i in order]
    labels = [labels[i] for i in order]
    return FilteredRemappedDataset(ds, indices, labels, label_map)


def _collect_standard_samples(
    datasets_list: Sequence[Dataset],
    allowed_labels: Sequence[int] | None,
    samples_per_label: int | None,
) -> Tuple[List[torch.Tensor], List[int], dict]:
    counts: Counter = Counter()
    images: List[torch.Tensor] = []
    labels: List[int] = []
    label_map = None
    if allowed_labels:
        allowed_sorted = sorted(set(int(l) for l in allowed_labels))
        label_map = {orig: idx for idx, orig in enumerate(allowed_sorted)}
        allowed_set = set(allowed_sorted)
    else:
        allowed_set = None

    for ds in datasets_list:
        for img, lbl in ds:
            if torch.is_tensor(lbl):
                lbl_val = int(lbl.view(-1)[0].item())
            elif isinstance(lbl, (list, tuple, np.ndarray)):
                lbl_val = int(np.array(lbl).reshape(-1)[0].item())
            else:
                lbl_val = int(lbl)
            if allowed_set is not None and lbl_val not in allowed_set:
                continue
            if samples_per_label is not None and counts[lbl_val] >= samples_per_label:
                continue
            mapped_label = label_map[lbl_val] if label_map is not None else lbl_val
            if torch.is_tensor(img):
                img_t = img
            else:
                img_t = transforms.ToTensor()(img)
            if img_t.dim() == 2:
                img_t = img_t.unsqueeze(0)
            images.append(img_t)
            labels.append(mapped_label)
            counts[lbl_val] += 1
            if allowed_set and samples_per_label is not None:
                if all(counts[l] >= samples_per_label for l in allowed_set):
                    return images, labels, label_map or {}
    return images, labels, label_map or {}


def build_standard_image_dataset(
    dataset_choice: str,
    image_size: Sequence[int] | int,
    dataset_labels: Sequence[int] | None = None,
    samples_per_label: int | None = None,
    root: str | Path = "data",
) -> TensorListDataset:
    """
    Build a tensor dataset for torchvision datasets.
    """
    img_size = _normalize_image_size(image_size)
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    ds_list: List[Dataset] = []
    choice = dataset_choice.lower()

    if choice == "mnist":
        if dataset_labels is None:
            raise ValueError("dataset_labels must be provided for MNIST")
        base = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        ds_list = [base, test]
    elif choice in ("fmnist", "fashionmnist"):
        if dataset_labels is None:
            raise ValueError("dataset_labels must be provided for FMNIST")
        base = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
        ds_list = [base, test]
    elif choice == "cifar10":
        cifar_transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.Lambda(lambda img: _cifar10_rgb_to_grayscale(np.asarray(img))),
                transforms.ToTensor(),
            ]
        )
        base = datasets.CIFAR10(root=root, train=True, download=True, transform=cifar_transform)
        test = datasets.CIFAR10(root=root, train=False, download=True, transform=cifar_transform)
        ds_list = [base, test]
        if dataset_labels is None:
            dataset_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  
    else:
        raise ValueError(f"Unsupported dataset_choice '{dataset_choice}'")

    images, labels, label_map = _collect_standard_samples(ds_list, dataset_labels, samples_per_label)
    ds = TensorListDataset(images, labels)
    ds.label_map = label_map
    return ds


def _default_pcam_root() -> Path:
    return Path(__file__).resolve().parent / "data" / "camelyon" / "RGB"


def _default_binary_root() -> Path:
    return Path(__file__).resolve().parent / "data" / "Binary_dataset"


def _default_multi_root() -> Path:
    return Path(__file__).resolve().parent / "data" / "Multi_dataset"


def _resolve_binary_key(dataset_choice: str) -> str:
    choice = dataset_choice.lower()
    if choice in ("fmnist", "fashionmnist"):
        return "fmnist"
    return choice


def _resolve_multi_key(dataset_choice: str, image_size: Sequence[int] | int) -> str:
    choice = dataset_choice.lower()
    if choice in ("mnist8", "mnist28", "fmnist28", "cifar32"):
        return choice
    size = _normalize_image_size(image_size)
    if size[0] != size[1]:
        raise ValueError("Multi_dataset expects square image_size.")
    img = size[0]
    if choice == "mnist":
        if img == 8:
            return "mnist8"
        if img == 28:
            return "mnist28"
    elif choice in ("fmnist", "fashionmnist"):
        if img == 28:
            return "fmnist28"
    elif choice == "cifar10":
        if img == 32:
            return "cifar32"
    raise ValueError(f"Multi_dataset has no entry for dataset_choice={dataset_choice}, image_size={img}.")


def _load_npz_dataset(
    split_path: Path,
    name_prefix: str,
    resize_to: Sequence[int] | int | None = None,
) -> NpzDataset:
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    with np.load(split_path) as data:
        if "data" not in data or "labels" not in data:
            raise ValueError(f"{split_path} must contain 'data' and 'labels'")
        return NpzDataset(data["data"], data["labels"], name_prefix=name_prefix, resize_to=resize_to)


def _load_multi_combined_split(
    split_path: Path,
    ds_key: str,
    name_prefix: str,
    resize_to: Sequence[int] | int | None = None,
) -> NpzDataset:
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    data_key = f"{ds_key}_data"
    label_key = f"{ds_key}_labels"
    with np.load(split_path) as data:
        if data_key not in data or label_key not in data:
            raise ValueError(f"{split_path} must contain '{data_key}' and '{label_key}'")
        return NpzDataset(data[data_key], data[label_key], name_prefix=name_prefix, resize_to=resize_to)


def _empty_npz_dataset(
    shape: Sequence[int],
    name_prefix: str,
    resize_to: Sequence[int] | int | None = None,
) -> NpzDataset:
    data = np.empty((0,) + tuple(shape), dtype=np.float32)
    labels = np.empty((0,), dtype=np.int64)
    return NpzDataset(data, labels, name_prefix=name_prefix, resize_to=resize_to)


def build_pcam_dataset(
    image_size: Sequence[int] | int,
    samples_per_class: int | None = None,
    root: str | Path | None = None,
    seed: int = 42,
    force_grayscale: bool = False,
) -> PCamDataset:
    data_root = Path(root) if root is not None else _default_pcam_root()
    return PCamDataset(
        root=data_root,
        image_size=image_size,
        samples_per_class=samples_per_class,
        seed=seed,
        force_grayscale=force_grayscale,
    )


def _pcam_split_dirs(root: Path) -> Tuple[Path, Path, Path | None] | None:
    train_dir = root / "train"
    test_dir = root / "test"
    val_dir = root / "val"
    has_train = train_dir.is_dir()
    has_test = test_dir.is_dir()
    has_val = val_dir.is_dir()
    if has_train and has_test:
        return train_dir, test_dir, val_dir if has_val else None
    if has_train or has_test or has_val:
        raise FileNotFoundError(
            f"PCam root {root} contains partial split directories; expected both train/ and test/ if split exists."
        )
    return None


def build_standard_loaders(
    dataset_choice: str,
    image_size: Sequence[int] | int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    train_count: int | None = None,
    val_count: int | None = None,
    test_count: int | None = None,
    seed: int = 42,
    dataset_labels: Sequence[int] | None = None,
    samples_per_label: int | None = None,
    balance_sampler: bool = False,
    binary_root: str | Path | None = None,
    multi_root: str | Path | None = None,
) -> Tuple[Tuple[DataLoader, Dataset], Tuple[DataLoader, Dataset], Tuple[DataLoader, Dataset]]:
    """
    Construct DataLoaders from prepared Binary_dataset or Multi_dataset splits.
    """
    choice = dataset_choice.lower()
    if choice == "pcam":
        task = "binary"
    elif dataset_labels is not None:
        if len(dataset_labels) == 2:
            task = "binary"
        elif len(dataset_labels) == 10:
            task = "multi"
        else:
            raise ValueError("dataset_labels must have length 2 (binary) or 10 (multi).")
    else:
        task = "multi"

    if train_count is None or val_count is None or test_count is None:
        raise ValueError("train_count, val_count, and test_count are required when using prepared datasets.")

    if task == "binary":
        root = Path(binary_root) if binary_root is not None else _default_binary_root()
        ds_key = _resolve_binary_key(dataset_choice)
        ds_dir = root / ds_key
        resize_to = _normalize_image_size(image_size) if ds_key == "pcam" else None
        train_set = _load_npz_dataset(ds_dir / "train.npz", f"{ds_key}_train", resize_to=resize_to)
        val_set = _load_npz_dataset(ds_dir / "val.npz", f"{ds_key}_val", resize_to=resize_to)
        test_set = _load_npz_dataset(ds_dir / "test.npz", f"{ds_key}_test", resize_to=resize_to)
    else:
        root = Path(multi_root) if multi_root is not None else _default_multi_root()
        ds_key = _resolve_multi_key(dataset_choice, image_size)
        ds_dir = root / ds_key
        resize_to = None
        if ds_dir.is_dir():
            train_set = _load_npz_dataset(ds_dir / "train.npz", f"{ds_key}_train", resize_to=resize_to)
            val_path = ds_dir / "val.npz"
            if val_path.is_file():
                val_set = _load_npz_dataset(val_path, f"{ds_key}_val", resize_to=resize_to)
            else:
                val_set = _empty_npz_dataset(train_set[0][0].shape, f"{ds_key}_val", resize_to=resize_to)
            test_set = _load_npz_dataset(ds_dir / "test.npz", f"{ds_key}_test", resize_to=resize_to)
        else:
            train_set = _load_multi_combined_split(root / "train.npz", ds_key, f"{ds_key}_train")
            test_set = _load_multi_combined_split(root / "test.npz", ds_key, f"{ds_key}_test")
            val_set = _empty_npz_dataset(train_set[0][0].shape, f"{ds_key}_val")

    if resize_to is None:
        expected = _normalize_image_size(image_size)
        sample = train_set[0][0]
        if sample.shape[-2:] != expected:
            raise ValueError(
                f"Prepared dataset {ds_key} has shape {sample.shape[-2:]}, expected {expected}. "
                "Check --image-size."
            )

    train_sampler = None
    if balance_sampler:
        base_dataset = train_set.dataset if hasattr(train_set, "dataset") else train_set
        if hasattr(train_set, "indices"):
            indices = train_set.indices
        else:
            indices = list(range(len(train_set)))
        labels = [base_dataset.labels[i] for i in indices]
        total = len(labels)
        counts = Counter(labels)
        class_weights = {cls: total / (2 * max(1, cnt)) for cls, cnt in counts.items()}
        weights = torch.as_tensor([class_weights[lbl] for lbl in labels], dtype=torch.double)
        train_sampler = WeightedRandomSampler(weights, num_samples=total, replacement=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=not balance_sampler,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return (train_loader, train_set), (val_loader, val_set), (test_loader, test_set)
