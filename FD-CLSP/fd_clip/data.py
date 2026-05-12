from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class SampleMeta:
    path: Path
    label_idx: int
    label_text: str


FAULT_DESCRIPTIONS = {
    "normal": "normal bearing condition",
    "ball": "ball fault",
    "inner_race": "inner race fault",
    "outer_race": "outer race fault",
}


def _normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float32)
    mean = signal.mean()
    std = signal.std()
    if std < 1e-6:
        std = 1.0
    return (signal - mean) / std


def _window_signal(signal: np.ndarray, window_size: int, stride: int) -> Iterable[np.ndarray]:
    total = len(signal)
    if total < window_size:
        padding = np.zeros(window_size - total, dtype=np.float32)
        yield np.concatenate([signal, padding], axis=0)
        return

    for start in range(0, total - window_size + 1, stride):
        yield signal[start : start + window_size]


def _extract_drive_end_signal(mat_path: Path) -> np.ndarray:
    mat = sio.loadmat(mat_path)
    key = next((key for key in mat if key.endswith("_DE_time")), None)
    if key is None:
        raise KeyError(f"No drive-end signal found in {mat_path}")
    signal = np.asarray(mat[key]).reshape(-1)
    return _normalize_signal(signal)


def _build_fault_label_name(file_path: Path, root: Path) -> str:
    relative = file_path.relative_to(root)
    parts = relative.parts
    if parts[0] == "Ball":
        return "ball"
    if parts[0] == "Inner Race":
        return "inner_race"
    if parts[0] == "Outer Race":
        return "outer_race"
    raise ValueError(f"Unsupported fault label structure for {file_path}")


def _build_normal_label_name(file_path: Path, root: Path) -> str:
    relative = file_path.relative_to(root)
    parts = relative.parts
    if parts:
        return "normal"
    raise ValueError(f"Unsupported normal label structure for {file_path}")


def discover_mat_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    return sorted(root_path.rglob("*.mat"))


class CWRUClipDataset(Dataset):
    def __init__(
        self,
        fault_root: str | Path,
        normal_root: str | Path | None = None,
        window_size: int = 2048,
        stride: int = 1024,
        limit_per_file: int | None = None,
    ) -> None:
        self.fault_root = Path(fault_root)
        self.normal_root = Path(normal_root) if normal_root is not None else None
        self.window_size = window_size
        self.stride = stride
        self.samples: list[tuple[np.ndarray, int]] = []
        self.sample_metas: list[SampleMeta] = []
        self.label_to_idx: dict[str, int] = {}
        self.idx_to_text: dict[int, str] = {}
        self._build(limit_per_file=limit_per_file)

    def _build(self, limit_per_file: int | None) -> None:
        sources: list[tuple[Path, Callable[[Path, Path], str]]] = [(self.fault_root, _build_fault_label_name)]
        if self.normal_root is not None:
            sources.append((self.normal_root, _build_normal_label_name))

        for root, label_builder in sources:
            for mat_path in discover_mat_files(root):
                label_name = label_builder(mat_path, root)
                if label_name not in self.label_to_idx:
                    label_idx = len(self.label_to_idx)
                    self.label_to_idx[label_name] = label_idx
                    self.idx_to_text[label_idx] = FAULT_DESCRIPTIONS.get(label_name, label_name.replace("_", " "))
                label_idx = self.label_to_idx[label_name]

                signal = _extract_drive_end_signal(mat_path)
                for index, window in enumerate(_window_signal(signal, self.window_size, self.stride)):
                    self.samples.append((window, label_idx))
                    self.sample_metas.append(
                        SampleMeta(
                            path=mat_path,
                            label_idx=label_idx,
                            label_text=self.idx_to_text[label_idx],
                        )
                    )
                    if limit_per_file is not None and index + 1 >= limit_per_file:
                        break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        signal, label_idx = self.samples[index]
        meta = self.sample_metas[index]
        return {
            "signal": torch.from_numpy(signal).float(),
            "label_idx": label_idx,
            "label_text": self.idx_to_text[label_idx],
            "source_path": str(meta.path),
        }


def build_grouped_splits(
    dataset: CWRUClipDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    rng = np.random.default_rng(seed)
    label_to_files: dict[int, list[Path]] = {}
    for meta in dataset.sample_metas:
        label_to_files.setdefault(meta.label_idx, [])
        if meta.path not in label_to_files[meta.label_idx]:
            label_to_files[meta.label_idx].append(meta.path)

    train_files: set[Path] = set()
    val_files: set[Path] = set()
    test_files: set[Path] = set()

    for _, files in label_to_files.items():
        files = list(files)
        rng.shuffle(files)
        total = len(files)
        if total < 3:
            raise ValueError("Each class needs at least 3 source files for grouped train/val/test split.")

        train_count = max(1, int(round(total * train_ratio)))
        val_count = max(1, int(round(total * val_ratio)))
        if train_count + val_count >= total:
            val_count = 1
            train_count = total - 2
        test_count = total - train_count - val_count
        if test_count < 1:
            test_count = 1
            train_count = max(1, train_count - 1)

        train_files.update(files[:train_count])
        val_files.update(files[train_count : train_count + val_count])
        test_files.update(files[train_count + val_count : train_count + val_count + test_count])

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for index, meta in enumerate(dataset.sample_metas):
        if meta.path in train_files:
            train_indices.append(index)
        elif meta.path in val_files:
            val_indices.append(index)
        elif meta.path in test_files:
            test_indices.append(index)
        else:
            raise RuntimeError(f"Sample from {meta.path} was not assigned to any split.")

    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


def _collect_label_to_files(dataset: CWRUClipDataset) -> dict[int, list[Path]]:
    label_to_files: dict[int, list[Path]] = {}
    for meta in dataset.sample_metas:
        label_to_files.setdefault(meta.label_idx, [])
        if meta.path not in label_to_files[meta.label_idx]:
            label_to_files[meta.label_idx].append(meta.path)
    return label_to_files


def build_grouped_kfold_splits(
    dataset: CWRUClipDataset,
    num_folds: int,
    seed: int = 42,
) -> list[tuple[Dataset, Dataset, Dataset]]:
    if num_folds < 3:
        raise ValueError("num_folds must be at least 3 because the workflow needs train/val/test folds.")

    rng = np.random.default_rng(seed)
    label_to_files = _collect_label_to_files(dataset)
    label_to_fold_files: dict[int, list[list[Path]]] = {}

    for label_idx, files in label_to_files.items():
        files = list(files)
        if len(files) < num_folds:
            raise ValueError(f"Class {label_idx} has only {len(files)} files, fewer than num_folds={num_folds}.")
        rng.shuffle(files)
        folds = [files[fold_idx::num_folds] for fold_idx in range(num_folds)]
        label_to_fold_files[label_idx] = folds

    split_triples: list[tuple[Dataset, Dataset, Dataset]] = []
    for fold_idx in range(num_folds):
        test_files: set[Path] = set()
        val_files: set[Path] = set()
        train_files: set[Path] = set()

        for _, folds in label_to_fold_files.items():
            val_fold_idx = (fold_idx + 1) % num_folds
            test_files.update(folds[fold_idx])
            val_files.update(folds[val_fold_idx])
            for inner_idx, fold_files in enumerate(folds):
                if inner_idx not in {fold_idx, val_fold_idx}:
                    train_files.update(fold_files)

        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []
        for index, meta in enumerate(dataset.sample_metas):
            if meta.path in train_files:
                train_indices.append(index)
            elif meta.path in val_files:
                val_indices.append(index)
            elif meta.path in test_files:
                test_indices.append(index)
            else:
                raise RuntimeError(f"Sample from {meta.path} was not assigned to any fold.")

        split_triples.append(
            (Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices))
        )

    return split_triples
