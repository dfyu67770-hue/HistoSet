"""Training and evaluation engine for HistoSet."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required for histoset.torch_engine. Install with `pip install -e .[dl]`."
    ) from exc

from .torch_data import HistoSetFullImageDataset, HistoSetPatchDataset
from .torch_losses import (
    class_balanced_soft_cross_entropy,
    hierarchy_consistency_loss,
    remap_explanation_to_pattern,
    soft_cross_entropy,
    soft_dice_loss,
    soft_targets,
    tumor_margin_loss,
)
from .torch_models import TinyHistoSetUNet


@dataclass
class TrainingConfig:
    manifest: str
    output_dir: str
    mode: str = "histoset_hierarchy"
    seed: int = 409
    epochs: int = 8
    patch_size: int = 128
    steps_per_epoch: int = 256
    batch_size: int = 8
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    image_size: int = 256
    base_channels: int = 16
    device: str = "auto"
    hierarchy_weight: float = 0.25
    pattern_weight: float = 0.50
    dice_weight: float = 0.85
    tumor_margin_weight: float = 0.20
    tumor_margin: float = 0.20


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> "torch.device":
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def dice_from_labels(prediction: "torch.Tensor", target: "torch.Tensor", n_classes: int) -> tuple[float, float]:
    mask = target != 255
    dice_values = []
    true_positive_total = 0.0
    prediction_total = 0.0
    target_total = 0.0
    for class_index in range(n_classes):
        pred = (prediction == class_index) & mask
        tgt = (target == class_index) & mask
        denom = pred.sum() + tgt.sum()
        true_positive_total += float((pred & tgt).sum().item())
        prediction_total += float(pred.sum().item())
        target_total += float(tgt.sum().item())
        if denom > 0:
            dice_values.append((2 * (pred & tgt).sum().float() / denom.float()).item())
    if not dice_values:
        return math.nan, math.nan
    micro = 2 * true_positive_total / (prediction_total + target_total) if prediction_total + target_total else math.nan
    return float(micro), float(np.mean(dice_values))


def build_model(mode: str, base_channels: int = 16) -> TinyHistoSetUNet:
    direct_pattern_only = mode in {"hard_pattern_ce", "soft_pattern_ce"}
    return TinyHistoSetUNet(
        n_explanation_classes=4 if direct_pattern_only else 10,
        n_pattern_classes=4,
        base_channels=base_channels,
        dual_pattern_head=mode == "histoset_hierarchy",
    )


def training_loss(output: dict[str, "torch.Tensor"], batch: dict[str, object], mode: str, config: TrainingConfig) -> "torch.Tensor":
    foreground = batch["foreground"]
    assert isinstance(foreground, torch.Tensor)
    if mode == "hard_pattern_ce":
        target = batch["majority_pattern"]
        assert isinstance(target, torch.Tensor)
        return F.cross_entropy(output["explanation_logits"], target, ignore_index=255)
    if mode == "soft_pattern_ce":
        counts = batch["pattern_counts"]
        assert isinstance(counts, torch.Tensor)
        target = soft_targets(counts, foreground)
        return soft_cross_entropy(output["explanation_logits"], target, foreground)
    if mode == "hard_explanation_ce":
        target = batch["majority_explanation"]
        assert isinstance(target, torch.Tensor)
        return F.cross_entropy(output["explanation_logits"], target, ignore_index=255)

    explanation_counts = batch["explanation_counts"]
    pattern_counts = batch["pattern_counts"]
    assert isinstance(explanation_counts, torch.Tensor)
    assert isinstance(pattern_counts, torch.Tensor)
    explanation_target = soft_targets(explanation_counts, foreground)
    loss = class_balanced_soft_cross_entropy(output["explanation_logits"], explanation_target, foreground)
    if config.dice_weight:
        loss = loss + config.dice_weight * soft_dice_loss(output["explanation_logits"], explanation_target, foreground)
    if mode == "histoset_hierarchy":
        pattern_target = soft_targets(pattern_counts, foreground)
        loss = loss + config.pattern_weight * soft_dice_loss(output["pattern_logits"], pattern_target, foreground)
        loss = loss + config.hierarchy_weight * hierarchy_consistency_loss(
            output["explanation_logits"], output["pattern_logits"], foreground
        )
    if config.tumor_margin_weight:
        loss = loss + config.tumor_margin_weight * tumor_margin_loss(
            output["explanation_logits"], explanation_target, foreground, config.tumor_margin
        )
    return loss


@torch.no_grad()
def evaluate_model(
    model: "torch.nn.Module", manifest: str | Path, split: str, mode: str, image_size: int, device: "torch.device"
) -> pd.DataFrame:
    model.eval()
    dataset = HistoSetFullImageDataset(manifest, split=split, image_size=image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rows = []
    for batch in loader:
        image = batch["image"].to(device)
        foreground = batch["foreground"].to(device)
        output = model(image)
        if mode in {"hard_pattern_ce", "soft_pattern_ce"}:
            pattern_probability = F.softmax(output["explanation_logits"], dim=1)
            explanation_probability = None
        else:
            explanation_probability = F.softmax(output["explanation_logits"], dim=1)
            pattern_probability = remap_explanation_to_pattern(explanation_probability)
            if "pattern_logits" in output:
                direct_pattern_probability = F.softmax(output["pattern_logits"], dim=1)
            else:
                direct_pattern_probability = pattern_probability

        majority_pattern = batch["majority_pattern"]
        pattern_prediction = pattern_probability.argmax(dim=1).cpu()
        pattern_dice, pattern_macro_dice = dice_from_labels(pattern_prediction, majority_pattern, 4)
        row = {
            "split": split,
            "sample_id": batch["sample_id"][0],
            "dataset": batch["dataset"][0],
            "mode": mode,
            "dice_pattern": pattern_dice,
            "macro_dice_pattern": pattern_macro_dice,
        }
        if explanation_probability is not None and bool(batch["has_explanation_supervision"][0]):
            majority_explanation = batch["majority_explanation"]
            explanation_prediction = explanation_probability.argmax(dim=1).cpu()
            explanation_dice, explanation_macro_dice = dice_from_labels(explanation_prediction, majority_explanation, 10)
            explanation_target = soft_targets(batch["explanation_counts"].to(device), foreground)
            l1 = (explanation_probability - explanation_target).abs().sum(dim=1)
            row.update(
                {
                    "dice_explanation": explanation_dice,
                    "macro_dice_explanation": explanation_macro_dice,
                    "l1_explanation": float((l1[foreground] / 2).mean().item()),
                }
            )
            if "pattern_logits" in output:
                hierarchy_l1 = (direct_pattern_probability - pattern_probability).abs().sum(dim=1)
                row["hierarchy_l1"] = float((hierarchy_l1[foreground] / 2).mean().item())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    value_columns = [c for c in metrics.columns if c.startswith(("dice_", "macro_dice_", "l1_", "hierarchy_"))]
    return (
        metrics.groupby(["split", "mode"], dropna=False)[value_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def train_model(config: TrainingConfig) -> dict[str, str]:
    seed_all(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    device = resolve_device(config.device)

    train_dataset = HistoSetPatchDataset(
        config.manifest,
        split="train",
        patch_size=config.patch_size,
        steps_per_epoch=config.steps_per_epoch,
        seed=config.seed,
        sampling_target="mixed",
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    model = build_model(config.mode, base_channels=config.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            tensor_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            output = model(tensor_batch["image"])
            loss = training_loss(output, tensor_batch, config.mode, config)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses))})

    checkpoint_path = checkpoint_dir / f"histoset_{config.mode}_seed{config.seed}.pt"
    torch.save({"state_dict": model.cpu().state_dict(), "config": asdict(config)}, checkpoint_path)

    device = torch.device("cpu")
    model = build_model(config.mode, base_channels=config.base_channels).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    metric_frames = []
    for split in ["val", "test"]:
        metric_frames.append(evaluate_model(model, config.manifest, split, config.mode, config.image_size, device))
    metrics = pd.concat(metric_frames, ignore_index=True)
    summary = summarize_metrics(metrics)

    history_path = output_dir / f"history_{config.mode}_seed{config.seed}.csv"
    metrics_path = output_dir / f"metrics_{config.mode}_seed{config.seed}.csv"
    summary_path = output_dir / f"metrics_summary_{config.mode}_seed{config.seed}.csv"
    config_path = output_dir / f"config_{config.mode}_seed{config.seed}.json"
    pd.DataFrame(history).to_csv(history_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return {
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "metrics": str(metrics_path),
        "summary": str(summary_path),
        "config": str(config_path),
    }
