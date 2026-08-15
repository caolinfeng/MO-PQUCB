"""Bandit policies used in the MO-PQUCB experiments."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from .plackett_luce import GroupLassoPlackettLuce, OnlinePlackettLuce


def _stable_inverse(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(matrix, hermitian=True)


def _random_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    maximum = np.nanmax(values)
    candidates = np.flatnonzero(np.isclose(values, maximum, rtol=1e-12, atol=1e-12))
    return int(rng.choice(candidates))


def _pareto_front(values: np.ndarray) -> np.ndarray:
    """Return row indices not strictly Pareto dominated by another row."""

    keep = np.ones(values.shape[0], dtype=bool)
    for index in range(values.shape[0]):
        dominates = np.all(values >= values[index], axis=1) & np.any(
            values > values[index], axis=1
        )
        dominates[index] = False
        if np.any(dominates):
            keep[index] = False
    return np.flatnonzero(keep)


class BasePolicy:
    """Common policy interface and shared objective-reward statistics."""

    def __init__(
        self,
        num_arms: int,
        num_users: int,
        dimension: int,
        seed: int,
        arm_features: Optional[np.ndarray] = None,
    ) -> None:
        self.num_arms = num_arms
        self.num_users = num_users
        self.dimension = dimension
        self.rng = np.random.default_rng(seed)
        self.arm_features = (
            None if arm_features is None else np.asarray(arm_features, dtype=float)
        )
        self.arm_counts = np.zeros(num_arms, dtype=int)
        self.reward_sums = np.zeros((num_arms, dimension), dtype=float)

    @property
    def reward_estimates(self) -> np.ndarray:
        denominator = np.maximum(1, self.arm_counts)[:, None]
        return self.reward_sums / denominator

    def observe_query(
        self, user: int, ranking: Sequence[int], round_index: int
    ) -> None:
        del user, ranking, round_index

    def observe_conversation(
        self,
        user: int,
        round_index: int,
        utility_oracle: Callable[[np.ndarray], float],
    ) -> None:
        del user, round_index, utility_oracle

    def select(self, user: int, available: np.ndarray, round_index: int) -> int:
        raise NotImplementedError

    def update(
        self, user: int, arm: int, reward: np.ndarray, utility: float
    ) -> None:
        del user, utility
        self.arm_counts[arm] += 1
        self.reward_sums[arm] += reward

    def preference_estimate(self, user: int) -> Optional[np.ndarray]:
        del user
        return None


class ScalarizedUCBPolicy(BasePolicy):
    def __init__(self, *args: Any, confidence_scale: float = 1.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.confidence_scale = confidence_scale

    def select(self, user: int, available: np.ndarray, round_index: int) -> int:
        del user
        counts = np.maximum(1, self.arm_counts[available])
        bonus = self.confidence_scale * np.sqrt(
            2.0 * np.log(round_index + 2.0) / counts
        )
        values = np.sum(self.reward_estimates[available] + bonus[:, None], axis=1)
        return int(available[_random_argmax(values, self.rng)])


class ScalarizedMOSSPolicy(BasePolicy):
    def __init__(
        self, *args: Any, horizon: int, confidence_scale: float = 1.0, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.horizon = horizon
        self.confidence_scale = confidence_scale

    def select(self, user: int, available: np.ndarray, round_index: int) -> int:
        del user, round_index
        counts = np.maximum(1, self.arm_counts[available])
        log_term = np.log(np.maximum(1.0, self.horizon / (self.num_arms * counts)))
        bonus = self.confidence_scale * np.sqrt(4.0 * log_term / counts)
        values = np.sum(self.reward_estimates[available] + bonus[:, None], axis=1)
        return int(available[_random_argmax(values, self.rng)])


class ParetoUCBPolicy(BasePolicy):
    def __init__(self, *args: Any, confidence_scale: float = 1.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.confidence_scale = confidence_scale

    def select(self, user: int, available: np.ndarray, round_index: int) -> int:
        del user
        counts = np.maximum(1, self.arm_counts[available])
        bonus = self.confidence_scale * np.sqrt(
            2.0 * np.log(round_index + 2.0) / counts
        )
        optimistic = self.reward_estimates[available] + bonus[:, None]
        front = _pareto_front(optimistic)
        return int(available[int(self.rng.choice(front))])


class ParetoTSPolicy(BasePolicy):
    def __init__(self, *args: Any, posterior_scale: float = 1.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.posterior_scale = posterior_scale

    def select(self, user: int, available: np.ndarray, round_index: int) -> int:
        del user, round_index
        counts = np.maximum(1, self.arm_counts[available])
        samples = self.rng.normal(
            self.reward_estimates[available],
            self.posterior_scale / np.sqrt(counts)[:, None],
        )
        front = _pareto_front(samples)
        return int(available[int(self.rng.choice(front))])


class PRUCBPolicy(BasePolicy):
    """PRUCB baseline transcribed from ``contextual_top_m.ipynb``."""

    def __init__(
        self,
        *args: Any,
        regularization: float = 0.1,
        epsilon: float = 1.5,
        omega: float = 10.0,
        reward_scale: float = 0.05,
        initial_reward_ucb: float = 100.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.regularization = regularization
        self.epsilon = epsilon
        self.omega = omega
        self.reward_scale = reward_scale
        self.optimistic_rewards = np.full(
            (self.num_arms, self.dimension), initial_reward_ucb, dtype=float
        )
        self.grams = np.tile(
            regularization * np.eye(self.dimension), (self.num_users, 1, 1)
        )
        self.responses = np.zeros((self.num_users, self.dimension), dtype=float)
        self.preferences = np.zeros((self.num_users, self.dimension), dtype=float)
        self.user_rounds = np.zeros(self.num_users, dtype=int)
        self._last_round = 0

    def select(self, user: int, available: np.ndarray, round_index: int) -> int:
        self._last_round = round_index
        beta = self.epsilon * np.sqrt(
            self.dimension
            * np.log1p(self.dimension * self.user_rounds[user] / self.regularization)
        )
        inverse = _stable_inverse(self.grams[user])
        rewards = self.optimistic_rewards[available]
        preference_bonus = beta * np.sqrt(
            np.maximum(0.0, np.einsum("ij,jk,ik->i", rewards, inverse, rewards))
        )
        values = rewards @ self.preferences[user] + preference_bonus
        return int(available[int(np.argmax(values))])

    def update(
        self, user: int, arm: int, reward: np.ndarray, utility: float
    ) -> None:
        super().update(user, arm, reward, utility)
        weight = self.omega / max(float(reward @ reward), 1e-12)
        self.grams[user] += weight * np.outer(reward, reward)
        self.responses[user] += weight * utility * reward
        self.preferences[user] = _stable_inverse(self.grams[user]) @ self.responses[user]
        self.user_rounds[user] += 1
        notebook_counts = self.arm_counts.astype(float) + 1.0
        radius = self.reward_scale * np.sqrt(
            2.0 * np.log(self._last_round + 1.0) / (notebook_counts - 0.999)
        )
        self.optimistic_rewards = self.reward_estimates + radius[:, None]

    def preference_estimate(self, user: int) -> np.ndarray:
        return self.preferences[user].copy()


class MOOFULPolicy(BasePolicy):
    """MO-OFUL/LinUCB baseline transcribed from the experiment notebook."""

    def __init__(
        self,
        *args: Any,
        regularization: float = 0.1,
        epsilon: float = 3.0,
        random_exploration: float = 0.05,
        response_feature: str = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.regularization = regularization
        self.epsilon = epsilon
        self.random_exploration = random_exploration
        if response_feature not in {"mean", "observed"}:
            raise ValueError("response_feature must be 'mean' or 'observed'")
        self.response_feature = response_feature
        self.grams = np.tile(
            regularization * np.eye(self.dimension), (self.num_users, 1, 1)
        )
        self.responses = np.zeros((self.num_users, self.dimension), dtype=float)
        self.preferences = np.zeros((self.num_users, self.dimension), dtype=float)
        self.user_rounds = np.zeros(self.num_users, dtype=int)

    def select(self, user: int, available: np.ndarray, round_index: int) -> int:
        if self.rng.random() < self.random_exploration:
            return int(self.rng.choice(available))
        beta = self.epsilon * np.sqrt(
            self.dimension
            * np.log1p(self.dimension * self.user_rounds[user] / self.regularization)
        )
        rewards = self.reward_estimates[available]
        inverse = _stable_inverse(self.grams[user])
        bonus = beta * np.sqrt(
            np.maximum(0.0, np.einsum("ij,jk,ik->i", rewards, inverse, rewards))
        )
        values = rewards @ self.preferences[user] + bonus
        return int(available[int(np.argmax(values))])

    def update(
        self, user: int, arm: int, reward: np.ndarray, utility: float
    ) -> None:
        super().update(user, arm, reward, utility)
        self.grams[user] += np.outer(reward, reward)
        feature = reward
        if self.response_feature == "mean" and self.arm_features is not None:
            feature = self.arm_features[arm]
        self.responses[user] += utility * feature
        self.preferences[user] = _stable_inverse(self.grams[user]) @ self.responses[user]
        self.user_rounds[user] += 1

    def preference_estimate(self, user: int) -> np.ndarray:
        return self.preferences[user].copy()


class ConversationalUCBPolicy(BasePolicy):
    """ConUCB with the notebook's Eq. (4), Eq. (8), and arm UCB terms."""

    def __init__(
        self,
        *args: Any,
        regularization: float = 0.1,
        epsilon: float = 3.0,
        keyword_subset_size: int = 3,
        conversation_rate: float = 5.0,
        random_exploration: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 < regularization < 1.0:
            raise ValueError("ConUCB regularization must lie strictly between 0 and 1")
        if self.arm_features is None:
            raise ValueError("ConUCB requires arm_features")
        self.regularization = regularization
        self.epsilon = epsilon
        self.conversation_rate = conversation_rate
        self.random_exploration = random_exploration
        subset = min(keyword_subset_size, self.dimension)
        top = np.argpartition(self.arm_features, -subset, axis=1)[:, -subset:]
        self.keyword_weights = np.zeros((self.num_arms, self.dimension), dtype=float)
        rows = np.arange(self.num_arms)[:, None]
        self.keyword_weights[rows, top] = 1.0 / subset
        self.query_grams = np.tile(
            regularization * np.eye(self.dimension), (self.num_users, 1, 1)
        )
        self.query_responses = np.zeros((self.num_users, self.dimension), dtype=float)
        self.grams = np.tile(
            (1.0 - regularization) * np.eye(self.dimension),
            (self.num_users, 1, 1),
        )
        self.responses = np.zeros((self.num_users, self.dimension), dtype=float)
        self.preferences = np.zeros((self.num_users, self.dimension), dtype=float)
        self.user_rounds = np.zeros(self.num_users, dtype=int)
        self.query_counts = np.zeros(self.num_users, dtype=int)

    def observe_conversation(
        self,
        user: int,
        round_index: int,
        utility_oracle: Callable[[np.ndarray], float],
    ) -> None:
        del round_index
        self.user_rounds[user] += 1
        local_time = int(self.user_rounds[user])
        budget = (
            int(self.conversation_rate * np.floor(np.log(local_time)))
            if local_time > 1
            else 0
        )
        new_queries = budget - int(self.query_counts[user])

        for _ in range(max(0, new_queries)):
            gram_inverse = _stable_inverse(self.grams[user])
            query_inverse = _stable_inverse(self.query_grams[user])
            best_value = -np.inf
            best_vector: Optional[np.ndarray] = None
            for keyword in range(self.keyword_weights.shape[1]):
                weights = self.keyword_weights[:, keyword]
                denominator = float(weights.sum())
                if denominator <= 0.0:
                    continue
                vector = (
                    (weights[:, None] * self.reward_estimates).sum(axis=0)
                    / denominator
                )
                transformed = gram_inverse @ (query_inverse @ vector)
                value = float(transformed @ transformed) / (
                    1.0 + float(vector @ query_inverse @ vector)
                )
                if value > best_value:
                    best_value = value
                    best_vector = vector
            if best_vector is None:
                break
            feedback = utility_oracle(best_vector)
            self.query_grams[user] += np.outer(best_vector, best_vector)
            self.query_responses[user] += best_vector * feedback
            self.query_counts[user] += 1

    def _refresh_preference(self, user: int) -> None:
        query_preference = (
            _stable_inverse(self.query_grams[user]) @ self.query_responses[user]
        )
        self.preferences[user] = _stable_inverse(self.grams[user]) @ (
            self.responses[user] + (1.0 - self.regularization) * query_preference
        )

    def select(self, user: int, available: np.ndarray, round_index: int) -> int:
        self._refresh_preference(user)
        if self.rng.random() < self.random_exploration:
            return int(self.rng.choice(available))
        del round_index
        t = max(1.0, float(self.user_rounds[user]))
        log_term = np.log1p(
            self.regularization
            * t
            / ((1.0 - self.regularization) * self.dimension)
        )
        alpha = self.epsilon * np.sqrt(self.dimension * log_term)
        alpha_tilde = self.epsilon * np.sqrt(self.dimension + log_term)
        gram_inverse = _stable_inverse(self.grams[user])
        query_inverse = _stable_inverse(self.query_grams[user])
        rewards = self.reward_estimates[available]
        first_radius = np.sqrt(
            np.maximum(0.0, np.einsum("ij,jk,ik->i", rewards, gram_inverse, rewards))
        )
        transformed = rewards @ gram_inverse
        second_radius = np.sqrt(
            np.maximum(
                0.0,
                np.einsum("ij,jk,ik->i", transformed, query_inverse, transformed),
            )
        )
        values = (
            rewards @ self.preferences[user]
            + alpha * self.regularization * first_radius
            + alpha_tilde * (1.0 - self.regularization) * second_radius
        )
        return int(available[int(np.argmax(values))])

    def update(
        self, user: int, arm: int, reward: np.ndarray, utility: float
    ) -> None:
        super().update(user, arm, reward, utility)
        self.grams[user] += np.outer(reward, reward)
        self.responses[user] += utility * reward

    def preference_estimate(self, user: int) -> np.ndarray:
        return self.preferences[user].copy()


class MOPQUCBPolicy(BasePolicy):
    """MO-PQUCB with the paper's QE anchor and dual-exploration bonuses."""

    def __init__(
        self,
        *args: Any,
        horizon: int,
        top_m: int,
        alpha: Optional[float] = None,
        regularization: float = 0.1,
        rho: float = 0.05,
        beta_scale: float = 1.0,
        reward_scale: float = 1.0,
        ucb_mode: str = "empirical",
        epsilon: float = 1.5,
        omega: float = 5.0,
        initial_reward_ucb: float = 100.0,
        robust: bool = False,
        pl_learning_rate: float = 0.1,
        pl_decay: float = 0.02,
        pl_momentum: float = 0.9,
        group_lasso_penalty: float = 0.1,
        group_lasso_steps: int = 5,
        group_lasso_history: Optional[int] = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.horizon = horizon
        self.top_m = top_m
        self.alpha = float(1.0 / np.log(max(3, horizon)) if alpha is None else alpha)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")
        self.regularization = regularization
        self.rho = rho
        self.beta_scale = beta_scale
        self.reward_scale = reward_scale
        self.ucb_mode = ucb_mode
        if self.ucb_mode not in {"empirical", "theoretical"}:
            raise ValueError("ucb_mode must be 'empirical' or 'theoretical'")
        self.epsilon = epsilon
        self.omega = omega
        self.initial_reward_ucb = initial_reward_ucb
        self.projector = np.eye(self.dimension) - np.ones(
            (self.dimension, self.dimension)
        ) / self.dimension
        self.anchor_matrix = self.projector + regularization * np.eye(self.dimension)
        if self.ucb_mode == "empirical":
            initial = (1.0 - self.alpha) * self.projector + regularization * np.eye(
                self.dimension
            )
        else:
            initial = (1.0 - self.alpha) * self.anchor_matrix
        self.grams = np.tile(initial, (self.num_users, 1, 1))
        self.responses = np.zeros((self.num_users, self.dimension), dtype=float)
        self.query_anchors = np.zeros((self.num_users, self.dimension), dtype=float)
        self.preferences = np.zeros((self.num_users, self.dimension), dtype=float)
        self.query_counts = np.zeros(self.num_users, dtype=int)

        if robust:
            self.query_estimators = [
                GroupLassoPlackettLuce(
                    self.dimension,
                    learning_rate=pl_learning_rate,
                    penalty=group_lasso_penalty,
                    inner_steps=group_lasso_steps,
                    max_history=group_lasso_history,
                )
                for _ in range(self.num_users)
            ]
        else:
            self.query_estimators = [
                OnlinePlackettLuce(
                    self.dimension,
                    learning_rate=pl_learning_rate,
                    decay=pl_decay,
                    momentum=pl_momentum,
                )
                for _ in range(self.num_users)
            ]

    def observe_query(
        self, user: int, ranking: Sequence[int], round_index: int
    ) -> None:
        del round_index
        self.query_anchors[user] = self.query_estimators[user].update(ranking)
        self.query_counts[user] += 1
        self._refresh_preference(user)

    def _refresh_preference(self, user: int) -> None:
        anchor = self.anchor_matrix if self.ucb_mode == "theoretical" else self.projector
        right_hand_side = self.alpha * self.responses[user] + (
            1.0 - self.alpha
        ) * anchor @ self.query_anchors[user]
        self.preferences[user] = _stable_inverse(self.grams[user]) @ right_hand_side

    def _beta(self, user: int, round_index: int) -> float:
        if self.ucb_mode == "empirical":
            local_time = max(0.0, float(self.query_counts[user] - 1))
            elapsed = max(1.0, float(self.query_counts[user]))
            bandit_term = self.alpha * np.sqrt(
                self.dimension
                * np.log1p(self.dimension * local_time / self.regularization)
            )
            query_term = np.sqrt(
                (1.0 - self.alpha)
                * self.dimension
                * np.log(elapsed)
                / (self.top_m * elapsed)
            )
            regularization_term = np.sqrt(
                (1.0 - self.alpha) * self.regularization / self.dimension
            )
            return self.epsilon * (
                bandit_term + query_term + regularization_term
            )

        time = float(round_index + 2)
        query_information = max(1.0, self.top_m * self.query_counts[user])
        bandit_term = np.sqrt(
            self.alpha * self.dimension ** 3 * np.log1p(self.alpha * self.dimension * time)
        )
        query_term = self.dimension * np.sqrt(
            (1.0 - self.alpha) * np.log(time) / query_information
        )
        return self.beta_scale * (bandit_term + query_term)

    def select(self, user: int, available: np.ndarray, round_index: int) -> int:
        self._refresh_preference(user)
        estimates = self.reward_estimates[available]
        counts = np.maximum(1, self.arm_counts[available])
        inverse = _stable_inverse(self.grams[user])

        if self.ucb_mode == "empirical":
            radius = self.reward_scale * np.sqrt(
                2.0 * np.log(round_index + 1.0) / counts
            )
            optimistic_rewards = estimates + radius[:, None]
            unseen = self.arm_counts[available] == 0
            optimistic_rewards[unseen] = self.initial_reward_ucb
            estimated_utility = optimistic_rewards @ self.preferences[user]
            preference_norm = np.sqrt(
                np.maximum(
                    0.0,
                    np.einsum(
                        "ij,jk,ik->i",
                        optimistic_rewards,
                        inverse,
                        optimistic_rewards,
                    ),
                )
            )
            values = estimated_utility + self._beta(user, round_index) * preference_norm
            return int(available[_random_argmax(values, self.rng)])

        gamma = self.reward_scale * np.sqrt(
            np.log((round_index + 2.0) / self.rho) / counts
        )
        reward_bonus = gamma * np.linalg.norm(self.preferences[user], ord=1)
        optimistic_rewards = estimates + gamma[:, None]
        preference_norm = np.sqrt(
            np.maximum(
                0.0,
                np.einsum(
                    "ij,jk,ik->i", optimistic_rewards, inverse, optimistic_rewards
                ),
            )
        )
        preference_bonus = self._beta(user, round_index) * preference_norm
        values = estimates @ self.preferences[user] + reward_bonus + preference_bonus
        return int(available[_random_argmax(values, self.rng)])

    def update(
        self, user: int, arm: int, reward: np.ndarray, utility: float
    ) -> None:
        super().update(user, arm, reward, utility)
        weight = (
            self.omega / max(np.linalg.norm(reward) ** 2, 1e-12)
            if self.ucb_mode == "empirical"
            else 1.0
        )
        self.grams[user] += self.alpha * weight * np.outer(reward, reward)
        self.responses[user] += weight * utility * reward
        self._refresh_preference(user)

    def preference_estimate(self, user: int) -> np.ndarray:
        return self.preferences[user].copy()


def build_policy(
    name: str,
    *,
    num_arms: int,
    num_users: int,
    dimension: int,
    horizon: int,
    top_m: int,
    seed: int,
    arm_features: Optional[np.ndarray] = None,
    parameters: Optional[Mapping[str, Any]] = None,
) -> BasePolicy:
    """Build a named policy while keeping JSON configuration concise."""

    params: Dict[str, Any] = dict(parameters or {})
    common = dict(
        num_arms=num_arms,
        num_users=num_users,
        dimension=dimension,
        seed=seed,
        arm_features=arm_features,
    )
    aliases = {
        "pqucb": "mo_pqucb",
        "pqucb_gl": "mo_pqucb_gl",
        "oful": "mo_oful",
    }
    canonical = aliases.get(name.lower(), name.lower())

    if canonical == "mo_pqucb":
        return MOPQUCBPolicy(
            **common, horizon=horizon, top_m=top_m, robust=False, **params
        )
    if canonical == "mo_pqucb_gl":
        return MOPQUCBPolicy(
            **common, horizon=horizon, top_m=top_m, robust=True, **params
        )
    if canonical == "prucb":
        return PRUCBPolicy(**common, **params)
    if canonical == "mo_oful":
        return MOOFULPolicy(**common, **params)
    if canonical == "conucb":
        return ConversationalUCBPolicy(**common, **params)
    if canonical == "s_ucb":
        return ScalarizedUCBPolicy(**common, **params)
    if canonical == "s_moss":
        return ScalarizedMOSSPolicy(**common, horizon=horizon, **params)
    if canonical == "pareto_ucb":
        return ParetoUCBPolicy(**common, **params)
    if canonical == "pareto_ts":
        return ParetoTSPolicy(**common, **params)
    raise ValueError("unknown algorithm: {}".format(name))
