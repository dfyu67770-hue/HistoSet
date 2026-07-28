"""Dataset utilities for prepared HistoSet NPZ tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required for histoset.torch_data. Install with `pip install -e .[dl]`."
    ) from exc


IGNORE_INDEX = 255


def read_manifest(manifest: str | Path) -> pd.DataFrame:
    path = Path(manifest)
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    table = pd.read_csv(path)
    required = {"split", "npz_path"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    return table


class HistoSetFullImageDataset(Dataset):
    """Prepared full-image tensor dataset."""

    def __init__(self, manifest: str | Path, split: str, image_size: int | None = None) -> None:
        table = read_manifest(manifest)
        self.rows = table[table["split"].astype(str) == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No rows found for split={split}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows.iloc[index]
        data = np.load(row["npz_path"])
        image = torch.from_numpy(data["image"]).permute(2, 0, 1).float() / 255.0
        foreground = torch.from_numpy(data["foreground"].astype(bool))
        explanation_counts = torch.from_numpy(data["explanation_counts"].astype(np.float32))
        pattern_counts = torch.from_numpy(data["pattern_counts"].astype(np.float32))
        majority_explanation = torch.from_numpy(data["majority_explanation"].astype(np.int64))
        majority_pattern = torch.from_numpy(data["majority_pattern"].astype(np.int64))
        if self.image_size and self.image_size != image.shape[-1]:
            size = (self.image_size, self.image_size)
            image = F.interpolate(image[None], size=size, mode="bilinear", align_corners=False)[0]
            foreground = F.interpolate(foreground.float()[None, None], size=size, mode="nearest")[0, 0].bool()
            explanation_counts = F.interpolate(explanation_counts[None], size=size, mode="nearest")[0]
            pattern_counts = F.interpolate(pattern_counts[None], size=size, mode="nearest")[0]
            majority_explanation = F.interpolate(
                majority_explanation.float()[None, None], size=size, mode="nearest"
            )[0, 0].long()
            majority_pattern = F.interpolate(majority_pattern.float()[None, None], size=size, mode="nearest")[0, 0].long()
        sample_id = row.get("sample_id", row.get("source_id", str(index)))
        return {
            "image": image,
            "foreground": foreground,
            "explanation_counts": explanation_counts,
            "pattern_counts": pattern_counts,
            "majority_explanation": majority_explanation,
            "majority_pattern": majority_pattern,
            "has_pattern_supervision": bool(int(data["has_pattern_supervision"])),
            "has_explanation_supervision": bool(int(data["has_explanation_supervision"])),
            "dataset": str(row.get("dataset", "unknown")),
            "sample_id": str(sample_id),
        }


class HistoSetPatchDataset(Dataset):
    """Balanced patch sampler for HistoSet training."""

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        patch_size: int = 128,
        steps_per_epoch: int = 256,
        seed: int = 1,
        sampling_target: str = "mixed",
    ) -> None:
        table = read_manifest(manifest)
        self.rows = table[table["split"].astype(str) == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No rows found for split={split}")
        self.patch_size = patch_size
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.sampling_target = sampling_target
        self.explanation_rows = self._index_rows("explanation_counts", 10)
        self.pattern_rows = self._index_rows("pattern_counts", 4)

    def _index_rows(self, key: str, n_classes: int) -> dict[int, list[int]]:
        indexed: dict[int, list[int]] = {c: [] for c in range(n_classes)}
        for row_index, row in self.rows.iterrows():
            data = np.load(row["npz_path"])
            counts = data[key]
            foreground = data["foreground"].astype(bool)
            for class_index in range(n_classes):
                if np.any(foreground & (counts[class_index] > 0)):
                    indexed[class_index].append(int(row_index))
        return indexed

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _choose_row_and_mask(self, rng: np.random.Generator, index: int) -> tuple[pd.Series, np.ndarray]:
        use_explanation = self.sampling_target in {"explanation", "mixed"} and index % 3 == 0
        if use_explanation:
            active = [k for k, rows in self.explanation_rows.items() if rows]
            class_index = active[index % len(active)]
            row = self.rows.iloc[int(rng.choice(self.explanation_rows[class_index]))]
            data = np.load(row["npz_path"])
            return row, data["foreground"].astype(bool) & (data["explanation_counts"][class_index] > 0)
        active = [k for k, rows in self.pattern_rows.items() if rows]
        class_index = active[index % len(active)]
        row = self.rows.iloc[int(rng.choice(self.pattern_rows[class_index]))]
        data = np.load(row["npz_path"])
        return row, data["foreground"].astype(bool) & (data["pattern_counts"][class_index] > 0)

    def __getitem__(self, index: int) -> dict[str, object]:
        rng = np.random.default_rng(self.seed * 1_000_003 + index)
        row, target_mask = self._choose_row_and_mask(rng, index)
        data = np.load(row["npz_path"])
        image = data["image"]
        foreground = data["foreground"].astype(bool)
        ys, xs = np.where(target_mask)
        if len(xs) == 0:
            ys, xs = np.where(foreground)
        if len(xs) == 0:
            center_y = center_x = image.shape[0] // 2
        else:
            chosen = int(rng.integers(0, len(xs)))
            center_y, center_x = int(ys[chosen]), int(xs[chosen])
        half = self.patch_size // 2
        y0 = int(np.clip(center_y - half, 0, image.shape[0] - self.patch_size))
        x0 = int(np.clip(center_x - half, 0, image.shape[1] - self.patch_size))
        y_slice = slice(y0, y0 + self.patch_size)
        x_slice = slice(x0, x0 + self.patch_size)
        patch_image = image[y_slice, x_slice].astype(np.float32) / 255.0
        patch_foreground = data["foreground"][y_slice, x_slice].astype(bool)
        explanation_counts = data["explanation_counts"][:, y_slice, x_slice].astype(np.float32)
        pattern_counts = data["pattern_counts"][:, y_slice, x_slice].astype(np.float32)
        majority_explanation = data["majority_explanation"][y_slice, x_slice].astype(np.int64)
        majority_pattern = data["majority_pattern"][y_slice, x_slice].astype(np.int64)
        if rng.random() < 0.5:
            patch_image = np.flip(patch_image, axis=1).copy()
            patch_foreground = np.flip(patch_foreground, axis=1).copy()
            explanation_counts = np.flip(explanation_counts, axis=2).copy()
            pattern_counts = np.flip(pattern_counts, axis=2).copy()
            majority_explanation = np.flip(majority_explanation, axis=1).copy()
            majority_pattern = np.flip(majority_pattern, axis=1).copy()
        return {
            "image": torch.from_numpy(patch_image).permute(2, 0, 1).float(),
            "foreground": torch.from_numpy(patch_foreground).bool(),
            "explanation_counts": torch.from_numpy(explanation_counts),
            "pattern_counts": torch.from_numpy(pattern_counts),
            "majority_explanation": torch.from_numpy(majority_explanation),
            "majority_pattern": torch.from_numpy(majority_pattern),
            "has_pattern_supervision": bool(int(data["has_pattern_supervision"])),
            "has_explanation_supervision": bool(int(data["has_explanation_supervision"])),
            "dataset": str(row.get("dataset", "unknown")),
            "sample_id": str(row.get("sample_id", row.get("source_id", index))),
        }
