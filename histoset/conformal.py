from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def conformal_threshold(true_class_probabilities: Iterable[float], alpha: float = 0.10) -> float:
    """Return the split-conformal probability threshold for class-set inclusion.

    Parameters
    ----------
    true_class_probabilities:
        Probability assigned to the true class for each calibration example.
    alpha:
        Target miscoverage rate.
    """

    probs = np.asarray(list(true_class_probabilities), dtype=float)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("true_class_probabilities must be a non-empty one-dimensional array")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    scores = 1.0 - np.clip(probs, 0.0, 1.0)
    rank = min(math.ceil((probs.size + 1) * (1.0 - alpha)), probs.size)
    q = np.sort(scores)[rank - 1]
    return float(1.0 - q)


def prediction_sets(probabilities: np.ndarray, thresholds: np.ndarray | float) -> np.ndarray:
    """Convert class probabilities into calibrated prediction sets."""

    prob = np.asarray(probabilities, dtype=float)
    thr = np.asarray(thresholds, dtype=float)
    if prob.ndim < 2:
        raise ValueError("probabilities must have a class axis")
    if thr.ndim == 0:
        return prob >= float(thr)
    shape = (thr.shape[0],) + (1,) * (prob.ndim - 1)
    return prob >= thr.reshape(shape)


def set_size(sets: np.ndarray, axis: int = 0) -> np.ndarray:
    """Count the number of included labels in each prediction set."""

    return np.asarray(sets, dtype=bool).sum(axis=axis)
