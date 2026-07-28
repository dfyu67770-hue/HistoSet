from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConceptHierarchy:
    """Mapping between sub-explanations, explanation concepts, and patterns."""

    sub_to_explanation: np.ndarray
    explanation_to_pattern: np.ndarray

    def validate(self) -> None:
        if self.sub_to_explanation.ndim != 1:
            raise ValueError("sub_to_explanation must be one-dimensional")
        if self.explanation_to_pattern.ndim != 1:
            raise ValueError("explanation_to_pattern must be one-dimensional")
        if self.sub_to_explanation.size and self.sub_to_explanation.max() >= self.explanation_to_pattern.size:
            raise ValueError("sub_to_explanation contains an explanation index outside explanation_to_pattern")

    def sub_to_pattern(self) -> np.ndarray:
        self.validate()
        return self.explanation_to_pattern[self.sub_to_explanation]

    def aggregate(self, child_probabilities: np.ndarray, parent_count: int, mapping: np.ndarray) -> np.ndarray:
        """Aggregate child probabilities to parent probabilities by summation."""

        probs = np.asarray(child_probabilities, dtype=float)
        out = np.zeros((parent_count,) + probs.shape[1:], dtype=float)
        for child, parent in enumerate(mapping.astype(int)):
            out[parent] += probs[child]
        return out

    def sub_to_explanation_probabilities(self, sub_probabilities: np.ndarray) -> np.ndarray:
        self.validate()
        return self.aggregate(sub_probabilities, self.explanation_to_pattern.size, self.sub_to_explanation)

    def explanation_to_pattern_probabilities(self, explanation_probabilities: np.ndarray, pattern_count: int) -> np.ndarray:
        self.validate()
        return self.aggregate(explanation_probabilities, pattern_count, self.explanation_to_pattern)
