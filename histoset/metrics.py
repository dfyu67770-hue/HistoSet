from __future__ import annotations

import numpy as np


def dice_score(prediction: np.ndarray, target: np.ndarray, label: int, valid_mask: np.ndarray | None = None, eps: float = 1e-7) -> float:
    """Compute Dice score for a single hard-label class."""

    pred = np.asarray(prediction) == label
    tgt = np.asarray(target) == label
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        pred = pred & valid
        tgt = tgt & valid
    inter = float(np.logical_and(pred, tgt).sum())
    denom = float(pred.sum() + tgt.sum())
    return float((2.0 * inter + eps) / (denom + eps))


def macro_dice(prediction: np.ndarray, target: np.ndarray, labels: list[int], valid_mask: np.ndarray | None = None) -> float:
    """Average hard-label Dice over a label list."""

    if not labels:
        raise ValueError("labels must not be empty")
    return float(np.mean([dice_score(prediction, target, label, valid_mask) for label in labels]))


def soft_dice(probabilities: np.ndarray, soft_targets: np.ndarray, valid_mask: np.ndarray | None = None, eps: float = 1e-7) -> np.ndarray:
    """Compute per-class soft Dice for probabilistic targets."""

    prob = np.asarray(probabilities, dtype=float)
    tgt = np.asarray(soft_targets, dtype=float)
    if prob.shape != tgt.shape:
        raise ValueError("probabilities and soft_targets must have the same shape")
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        prob = prob * valid
        tgt = tgt * valid
    inter = (prob * tgt).sum(axis=tuple(range(1, prob.ndim)))
    denom = prob.sum(axis=tuple(range(1, prob.ndim))) + tgt.sum(axis=tuple(range(1, tgt.ndim)))
    return (2.0 * inter + eps) / (denom + eps)
