"""Plackett--Luce sampling and online preference estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


def sample_top_m(
    scores: np.ndarray, m: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample an ordered top-m subset from a Plackett--Luce model."""

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not 1 <= m <= scores.size:
        raise ValueError("m must lie in [1, number of objectives]")

    remaining = np.arange(scores.size)
    ranking = np.empty(m, dtype=int)
    for position in range(m):
        logits = scores[remaining]
        weights = np.exp(logits - np.max(logits))
        probabilities = weights / weights.sum()
        selected_position = int(rng.choice(remaining.size, p=probabilities))
        ranking[position] = remaining[selected_position]
        remaining = np.delete(remaining, selected_position)
    return ranking


def _negative_log_likelihood_gradient(
    ranking: Sequence[int], scores: np.ndarray
) -> np.ndarray:
    """Gradient of top-m PL negative log-likelihood."""

    dimension = scores.size
    gradient = np.zeros(dimension, dtype=float)
    selected = np.zeros(dimension, dtype=bool)
    for item in ranking:
        remaining = np.flatnonzero(~selected)
        logits = scores[remaining]
        weights = np.exp(logits - np.max(logits))
        probabilities = weights / weights.sum()
        gradient[remaining] += probabilities
        gradient[int(item)] -= 1.0
        selected[int(item)] = True
    return gradient


@dataclass
class OnlinePlackettLuce:
    """Warm-started projected SGD estimator on the zero-mean PL subspace."""

    dimension: int
    learning_rate: float = 0.1
    decay: float = 0.02
    momentum: float = 0.9
    gradient_clip: float = 1.0
    score_bound: Optional[float] = None

    def __post_init__(self) -> None:
        self.scores = np.zeros(self.dimension, dtype=float)
        self.velocity = np.zeros(self.dimension, dtype=float)
        self.steps = 0

    def update(self, ranking: Sequence[int]) -> np.ndarray:
        ranking_array = np.asarray(ranking, dtype=int)
        self._validate_ranking(ranking_array)
        self.steps += 1
        rate = self.learning_rate / (1.0 + self.decay * self.steps)
        gradient = -_negative_log_likelihood_gradient(ranking_array, self.scores)
        norm = np.linalg.norm(gradient)
        if norm > self.gradient_clip:
            gradient *= self.gradient_clip / norm
        self.velocity = self.momentum * self.velocity + (1.0 - self.momentum) * gradient
        self.scores += rate * self.velocity
        self._project()
        return self.estimate()

    def estimate(self) -> np.ndarray:
        return self.scores.copy()

    def _project(self) -> None:
        self.scores -= self.scores.mean()
        if self.score_bound is not None:
            self.scores = np.clip(self.scores, -self.score_bound, self.score_bound)
            self.scores -= self.scores.mean()

    def _validate_ranking(self, ranking: np.ndarray) -> None:
        if ranking.ndim != 1 or ranking.size == 0:
            raise ValueError("ranking must be a non-empty one-dimensional array")
        if np.unique(ranking).size != ranking.size:
            raise ValueError("ranking cannot contain duplicate objectives")
        if ranking.min() < 0 or ranking.max() >= self.dimension:
            raise ValueError("ranking contains an invalid objective index")


@dataclass
class GroupLassoPlackettLuce:
    """Robust PL estimator with a group-sparse per-query perturbation.

    The estimator approximately minimizes the objective in the paper using
    warm-started proximal-gradient steps over a bounded history window.
    """

    dimension: int
    learning_rate: float = 0.05
    penalty: float = 0.1
    inner_steps: int = 5
    gradient_clip: float = 5.0
    max_history: Optional[int] = 50
    score_bound: Optional[float] = None

    def __post_init__(self) -> None:
        self.scores = np.zeros(self.dimension, dtype=float)
        self.rankings: List[np.ndarray] = []
        self.perturbations: List[np.ndarray] = []

    def update(self, ranking: Sequence[int]) -> np.ndarray:
        ranking_array = np.asarray(ranking, dtype=int)
        if (
            ranking_array.ndim != 1
            or ranking_array.size == 0
            or np.unique(ranking_array).size != ranking_array.size
            or ranking_array.min() < 0
            or ranking_array.max() >= self.dimension
        ):
            raise ValueError("invalid ranking")
        self.rankings.append(ranking_array.copy())
        self.perturbations.append(np.zeros(self.dimension, dtype=float))
        if self.max_history is not None and len(self.rankings) > self.max_history:
            self.rankings = self.rankings[-self.max_history :]
            self.perturbations = self.perturbations[-self.max_history :]

        for _ in range(self.inner_steps):
            gradients = [
                _negative_log_likelihood_gradient(ranking_i, self.scores + delta_i)
                for ranking_i, delta_i in zip(self.rankings, self.perturbations)
            ]
            count = len(gradients)
            score_gradient = np.mean(gradients, axis=0)
            score_gradient = self._clip(score_gradient)
            self.scores -= self.learning_rate * score_gradient
            self._project_scores()

            new_perturbations = []
            for delta_i, gradient_i in zip(self.perturbations, gradients):
                candidate = delta_i - self.learning_rate * self._clip(gradient_i / count)
                norm = np.linalg.norm(candidate)
                threshold = self.learning_rate * self.penalty
                if norm <= threshold:
                    candidate = np.zeros_like(candidate)
                else:
                    candidate *= 1.0 - threshold / norm
                candidate -= candidate.mean()
                new_perturbations.append(candidate)
            self.perturbations = new_perturbations
        return self.estimate()

    def estimate(self) -> np.ndarray:
        return self.scores.copy()

    def _clip(self, gradient: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(gradient)
        if norm > self.gradient_clip:
            return gradient * (self.gradient_clip / norm)
        return gradient

    def _project_scores(self) -> None:
        self.scores -= self.scores.mean()
        if self.score_bound is not None:
            self.scores = np.clip(self.scores, -self.score_bound, self.score_bound)
            self.scores -= self.scores.mean()
