"""Synthetic and real-world environments used by the paper experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .plackett_luce import sample_top_m


def _round_rng(seed: int, round_index: int, stream: int, index: int = 0) -> np.random.Generator:
    sequence = np.random.SeedSequence([seed, round_index, stream, index])
    return np.random.default_rng(sequence)


@dataclass(frozen=True)
class BanditEnvironment:
    """Stationary preference-aware multi-objective bandit environment."""

    arm_means: np.ndarray
    user_preferences: np.ndarray
    horizon: int
    seed: int
    reward_std: float = np.sqrt(0.5)
    preference_std: float = np.sqrt(0.5)
    available_size: Optional[int] = None
    reward_lower: Optional[float] = None
    reward_upper: Optional[float] = None
    corruption_rate: float = 0.0

    def __post_init__(self) -> None:
        means = np.asarray(self.arm_means, dtype=float)
        preferences = np.asarray(self.user_preferences, dtype=float)
        if means.ndim != 2 or preferences.ndim != 2:
            raise ValueError("arm_means and user_preferences must be matrices")
        if means.shape[1] != preferences.shape[1]:
            raise ValueError("reward and preference dimensions do not match")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if not 0.0 <= self.corruption_rate <= 1.0:
            raise ValueError("corruption_rate must lie in [0, 1]")
        size = self.num_arms if self.available_size is None else self.available_size
        if not 1 <= size <= self.num_arms:
            raise ValueError("available_size must lie in [1, num_arms]")

    @property
    def num_arms(self) -> int:
        return int(self.arm_means.shape[0])

    @property
    def num_users(self) -> int:
        return int(self.user_preferences.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.arm_means.shape[1])

    def user_at(self, round_index: int) -> int:
        return int(_round_rng(self.seed, round_index, 0).integers(self.num_users))

    def available_arms_at(self, round_index: int) -> np.ndarray:
        size = self.num_arms if self.available_size is None else self.available_size
        if size == self.num_arms:
            return np.arange(self.num_arms, dtype=int)
        return np.sort(
            _round_rng(self.seed, round_index, 1).choice(
                self.num_arms, size=size, replace=False
            )
        )

    def reward_at(self, round_index: int, arm: int) -> np.ndarray:
        reward = self.arm_means[arm] + _round_rng(
            self.seed, round_index, 2, arm
        ).normal(0.0, self.reward_std, self.dimension)
        if self.reward_lower is not None or self.reward_upper is not None:
            lower = -np.inf if self.reward_lower is None else self.reward_lower
            upper = np.inf if self.reward_upper is None else self.reward_upper
            reward = np.clip(reward, lower, upper)
        return reward

    def instantaneous_preference(self, round_index: int, user: int) -> np.ndarray:
        return self.user_preferences[user] + _round_rng(
            self.seed, round_index, 3, user
        ).normal(0.0, self.preference_std, self.dimension)

    def utility_at(self, round_index: int, user: int, reward: np.ndarray) -> float:
        return float(self.instantaneous_preference(round_index, user) @ reward)

    def ranking_at(self, round_index: int, user: int, top_m: int) -> np.ndarray:
        rng = _round_rng(self.seed, round_index, 4, user)
        scores = self.user_preferences[user].copy()
        if rng.random() < self.corruption_rate:
            scores = scores[rng.permutation(self.dimension)]
        return sample_top_m(scores, top_m, rng)

    def expected_regret(self, user: int, arm: int, available: np.ndarray) -> float:
        utilities = self.arm_means[available] @ self.user_preferences[user]
        best = float(np.max(utilities))
        selected = float(self.arm_means[arm] @ self.user_preferences[user])
        return best - selected


def build_environment(
    raw: Mapping[str, Any], horizon: int, seed: int, project_root: Optional[Path] = None
) -> BanditEnvironment:
    """Create an environment from a JSON-compatible mapping."""

    environment_type = str(raw["type"])
    root = Path.cwd() if project_root is None else project_root
    rng = np.random.default_rng(seed)

    if environment_type == "synthetic":
        num_arms = int(raw.get("num_arms", 40))
        dimension = int(raw.get("dimension", 20))
        num_users = int(raw.get("num_users", 20))
        distribution = str(raw.get("mean_distribution", "uniform"))
        if distribution == "uniform":
            low, high = raw.get("mean_range", [0.0, 5.0])
            arm_means = rng.uniform(float(low), float(high), (num_arms, dimension))
        elif distribution == "dirichlet":
            concentration = rng.integers(1, num_arms + 1, (dimension, num_arms))
            columns = [
                rng.dirichlet(concentration[d]) * (num_arms // 2)
                for d in range(dimension)
            ]
            arm_means = np.asarray(columns).T
        else:
            raise ValueError("mean_distribution must be 'uniform' or 'dirichlet'")
        pref_low, pref_high = raw.get("preference_range", [0.0, 5.0])
        preferences = rng.uniform(float(pref_low), float(pref_high), (num_users, dimension))
    elif environment_type in {"tripadvisor", "beeradvocate"}:
        rewards_path = root / str(raw["rewards_path"])
        preferences_path = root / str(raw["preferences_path"])
        rewards = np.load(str(rewards_path))
        preferences = np.load(str(preferences_path))
        if rewards.ndim == 3:
            arm_means = rewards.mean(axis=0)
        elif rewards.ndim == 2:
            arm_means = rewards
        else:
            raise ValueError("real-world rewards must be a matrix or user-arm tensor")
        requested_users = int(raw.get("num_users", preferences.shape[0]))
        preferences = preferences[:requested_users]
    else:
        raise ValueError("unknown environment type: {}".format(environment_type))

    return BanditEnvironment(
        arm_means=np.asarray(arm_means, dtype=float),
        user_preferences=np.asarray(preferences, dtype=float),
        horizon=horizon,
        seed=seed,
        reward_std=float(raw.get("reward_std", np.sqrt(0.5))),
        preference_std=float(raw.get("preference_std", np.sqrt(0.5))),
        available_size=(
            None if raw.get("available_size") is None else int(raw["available_size"])
        ),
        reward_lower=raw.get("reward_lower"),
        reward_upper=raw.get("reward_upper"),
        corruption_rate=float(raw.get("corruption_rate", 0.0)),
    )
