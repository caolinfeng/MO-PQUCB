"""Shared experiment loop and artifact serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

from .config import ExperimentConfig
from .environment import build_environment
from .policies import build_policy


@dataclass(frozen=True)
class ExperimentResult:
    algorithm: str
    instantaneous_regret: np.ndarray
    cumulative_regret: np.ndarray
    preference_error: np.ndarray

    def summary(self) -> Dict[str, Optional[float]]:
        final = self.cumulative_regret[:, -1]
        valid_errors = self.preference_error[np.isfinite(self.preference_error)]
        return {
            "final_regret_mean": float(np.mean(final)),
            "final_regret_standard_error": float(
                np.std(final, ddof=1) / np.sqrt(final.size) if final.size > 1 else 0.0
            ),
            "mean_preference_error": (
                float(np.mean(valid_errors)) if valid_errors.size else None
            ),
        }


def _stable_name_seed(name: str) -> int:
    return sum((index + 1) * value for index, value in enumerate(name.encode("utf-8")))


def run_experiment(
    config: ExperimentConfig,
    *,
    project_root: Optional[Path] = None,
    query_rankings: Optional[Mapping[int, np.ndarray]] = None,
) -> Dict[str, ExperimentResult]:
    """Run all configured algorithms with common deterministic environments.

    ``query_rankings`` optionally maps run index to an ``(T, D or m)`` array.
    It is used by the LLM pipeline to replay cached inferred rankings without
    making network calls during bandit evaluation.
    """

    raw_regrets: Dict[str, list] = {name: [] for name in config.algorithms}
    cumulative_regrets: Dict[str, list] = {name: [] for name in config.algorithms}
    preference_errors: Dict[str, list] = {name: [] for name in config.algorithms}

    for run_index in range(config.runs):
        run_seed = config.seed + run_index
        environment = build_environment(
            config.environment, config.horizon, run_seed, project_root=project_root
        )
        if config.top_m > environment.dimension:
            raise ValueError("top_m cannot exceed the objective dimension")

        for algorithm in config.algorithms:
            policy = build_policy(
                algorithm,
                num_arms=environment.num_arms,
                num_users=environment.num_users,
                dimension=environment.dimension,
                horizon=config.horizon,
                top_m=config.top_m,
                seed=run_seed + _stable_name_seed(algorithm),
                arm_features=environment.arm_means,
                parameters=config.parameters_for(algorithm),
            )
            regret = np.zeros(config.horizon, dtype=float)
            error = np.full(config.horizon, np.nan, dtype=float)
            cached = None if query_rankings is None else query_rankings.get(run_index)
            if cached is not None and cached.shape[0] < config.horizon:
                raise ValueError("cached query rankings are shorter than the horizon")

            for round_index in range(config.horizon):
                user = environment.user_at(round_index)
                available = environment.available_arms_at(round_index)
                ranking = (
                    np.asarray(cached[round_index, : config.top_m], dtype=int)
                    if cached is not None
                    else environment.ranking_at(round_index, user, config.top_m)
                )
                policy.observe_query(user, ranking, round_index)
                policy.observe_conversation(
                    user,
                    round_index,
                    lambda vector, u=user, t=round_index: environment.utility_at(
                        t, u, vector
                    ),
                )
                arm = policy.select(user, available, round_index)
                if arm not in available:
                    raise RuntimeError("policy selected an unavailable arm")
                reward = environment.reward_at(round_index, arm)
                utility = environment.utility_at(round_index, user, reward)
                regret[round_index] = environment.expected_regret(user, arm, available)
                policy.update(user, arm, reward, utility)
                estimate = policy.preference_estimate(user)
                if estimate is not None:
                    error[round_index] = np.linalg.norm(
                        environment.user_preferences[user] - estimate
                    )

            raw_regrets[algorithm].append(regret)
            cumulative_regrets[algorithm].append(np.cumsum(regret))
            preference_errors[algorithm].append(error)

    return {
        algorithm: ExperimentResult(
            algorithm=algorithm,
            instantaneous_regret=np.asarray(raw_regrets[algorithm]),
            cumulative_regret=np.asarray(cumulative_regrets[algorithm]),
            preference_error=np.asarray(preference_errors[algorithm]),
        )
        for algorithm in config.algorithms
    }


def save_results(
    config: ExperimentConfig,
    results: Mapping[str, ExperimentResult],
    output_directory: Path,
) -> Path:
    """Render plots and a compact summary without storing per-round arrays."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    experiment_directory = output_directory / config.name
    experiment_directory.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, Dict[str, Optional[float]]] = {}
    for name, result in results.items():
        summaries[name] = result.summary()

    rounds = np.arange(1, config.horizon + 1)
    figure, axis = plt.subplots(figsize=(5.0, 3.5))
    for name, result in results.items():
        values = result.cumulative_regret
        mean = values.mean(axis=0)
        standard_error = (
            values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
            if values.shape[0] > 1
            else np.zeros(values.shape[1])
        )
        axis.plot(rounds, mean, label=name)
        axis.fill_between(rounds, mean - standard_error, mean + standard_error, alpha=0.2)
    axis.set_xlabel("Interaction round")
    axis.set_ylabel("Cumulative regret")
    axis.legend(frameon=False)
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(
            str(experiment_directory / ("cumulative_regret." + extension)),
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(figure)

    estimable = {
        name: result
        for name, result in results.items()
        if np.any(np.isfinite(result.preference_error))
    }
    if estimable:
        figure, axis = plt.subplots(figsize=(5.0, 3.5))
        for name, result in estimable.items():
            values = result.preference_error
            mean = np.nanmean(values, axis=0)
            valid_counts = np.maximum(1, np.sum(np.isfinite(values), axis=0))
            standard_error = np.nanstd(values, axis=0) / np.sqrt(valid_counts)
            axis.plot(rounds, mean, label=name)
            axis.fill_between(
                rounds, mean - standard_error, mean + standard_error, alpha=0.2
            )
        axis.set_xlabel("Interaction round")
        axis.set_ylabel("Preference estimation error")
        axis.set_yscale("log")
        axis.legend(frameon=False)
        figure.tight_layout()
        for extension in ("pdf", "png"):
            figure.savefig(
                str(experiment_directory / ("preference_error." + extension)),
                bbox_inches="tight",
                dpi=300,
            )
        plt.close(figure)

    with (experiment_directory / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "experiment": config.name,
                "horizon": config.horizon,
                "runs": config.runs,
                "top_m": config.top_m,
                "algorithms": summaries,
            },
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    return experiment_directory
