from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mo_pqucb.config import ExperimentConfig, load_config
from mo_pqucb.environment import BanditEnvironment
from mo_pqucb.llm_pipeline import (
    generation_prompt,
    parse_queries,
    parse_rankings,
    query_response_schema,
    ranking_response_schema,
    ranking_prompt,
)
from mo_pqucb.plackett_luce import (
    GroupLassoPlackettLuce,
    OnlinePlackettLuce,
    sample_top_m,
)
from mo_pqucb.policies import (
    ConversationalUCBPolicy,
    MOOFULPolicy,
    MOPQUCBPolicy,
    PRUCBPolicy,
)
from mo_pqucb.runner import run_experiment, save_results


class PlackettLuceTests(unittest.TestCase):
    def test_sample_is_valid_top_m_ranking(self) -> None:
        ranking = sample_top_m(np.array([3.0, 2.0, 1.0, 0.0]), 3, np.random.default_rng(4))
        self.assertEqual(ranking.shape, (3,))
        self.assertEqual(len(set(ranking.tolist())), 3)

    def test_online_estimator_stays_identifiable(self) -> None:
        estimator = OnlinePlackettLuce(4)
        for _ in range(20):
            estimator.update([0, 1])
        estimate = estimator.estimate()
        self.assertAlmostEqual(float(estimate.sum()), 0.0, places=10)
        self.assertGreater(estimate[0], estimate[3])

    def test_group_lasso_estimator_is_finite(self) -> None:
        estimator = GroupLassoPlackettLuce(4, inner_steps=2, max_history=5)
        for ranking in ([0, 1], [0, 2], [3, 2], [0, 1]):
            estimator.update(ranking)
        self.assertTrue(np.all(np.isfinite(estimator.estimate())))
        self.assertAlmostEqual(float(estimator.estimate().sum()), 0.0, places=10)


class EnvironmentTests(unittest.TestCase):
    def test_feedback_is_deterministic_by_round(self) -> None:
        environment = BanditEnvironment(
            arm_means=np.ones((4, 3)),
            user_preferences=np.ones((2, 3)),
            horizon=5,
            seed=9,
            available_size=2,
        )
        np.testing.assert_array_equal(
            environment.available_arms_at(2), environment.available_arms_at(2)
        )
        np.testing.assert_allclose(environment.reward_at(2, 1), environment.reward_at(2, 1))
        np.testing.assert_array_equal(
            environment.ranking_at(2, 0, 2), environment.ranking_at(2, 0, 2)
        )


class ConfigAndRunnerTests(unittest.TestCase):
    def test_prucb_uses_notebook_weighted_update(self) -> None:
        policy = PRUCBPolicy(
            num_arms=2, num_users=1, dimension=2, seed=1,
            regularization=0.1, epsilon=1.5, omega=10.0, reward_scale=0.05,
        )
        arm = policy.select(0, np.array([0, 1]), 0)
        reward = np.array([3.0, 4.0])
        policy.update(0, arm, reward, 2.0)
        weight = 10.0 / 25.0
        np.testing.assert_allclose(
            policy.grams[0], 0.1 * np.eye(2) + weight * np.outer(reward, reward)
        )
        np.testing.assert_allclose(policy.responses[0], weight * 2.0 * reward)

    def test_mo_oful_uses_notebook_mean_response_feature(self) -> None:
        means = np.array([[1.0, 2.0], [3.0, 4.0]])
        policy = MOOFULPolicy(
            num_arms=2, num_users=1, dimension=2, seed=1,
            arm_features=means, regularization=0.1, epsilon=3.0,
            random_exploration=0.0, response_feature="mean",
        )
        policy.update(0, 1, np.array([2.5, 4.5]), 2.0)
        np.testing.assert_allclose(policy.responses[0], 2.0 * means[1])
        self.assertEqual(policy.user_rounds[0], 1)

    def test_conucb_uses_independent_per_user_query_budgets(self) -> None:
        means = np.array([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]])
        policy = ConversationalUCBPolicy(
            num_arms=2, num_users=2, dimension=3, seed=1,
            arm_features=means, regularization=0.1, epsilon=3.0,
            keyword_subset_size=2, random_exploration=0.0,
        )
        policy.arm_counts[:] = 1
        policy.reward_sums[:] = means
        initial = policy.query_grams[0].copy()
        seen = []
        for round_index in range(3):
            policy.observe_conversation(
                0, round_index, lambda vector: seen.append(vector) or 1.0
            )
        self.assertEqual(len(seen), 5)
        self.assertEqual(policy.query_counts[0], 5)
        self.assertEqual(policy.user_rounds[0], 3)
        policy.observe_conversation(1, 3, lambda vector: 1.0)
        self.assertEqual(policy.query_counts[1], 0)
        self.assertEqual(policy.user_rounds[1], 1)
        self.assertFalse(np.allclose(policy.query_grams[0], initial))
        np.testing.assert_allclose(policy.keyword_weights.sum(axis=1), 1.0)

    def test_empirical_pqucb_bonus_has_experimental_scale(self) -> None:
        policy = MOPQUCBPolicy(
            num_arms=40,
            num_users=20,
            dimension=20,
            seed=1,
            horizon=1000,
            top_m=20,
            alpha=None,
            regularization=0.1,
            ucb_mode="empirical",
            epsilon=1.5,
            reward_scale=0.05,
        )
        policy.query_counts[0] = 50
        self.assertLess(policy._beta(0, 999), 5.0)

    def test_load_config(self) -> None:
        raw = {
            "name": "tiny",
            "horizon": 4,
            "runs": 1,
            "top_m": 2,
            "algorithms": ["mo_pqucb"],
            "environment": {"type": "synthetic", "dimension": 3},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config.horizon, 4)

    def test_all_policy_families_run(self) -> None:
        algorithms = [
            "s_ucb",
            "s_moss",
            "pareto_ucb",
            "pareto_ts",
            "mo_oful",
            "conucb",
            "prucb",
            "mo_pqucb",
            "mo_pqucb_gl",
        ]
        config = ExperimentConfig.from_mapping(
            {
                "name": "tiny",
                "seed": 3,
                "horizon": 8,
                "runs": 1,
                "top_m": 2,
                "algorithms": algorithms,
                "environment": {
                    "type": "synthetic",
                    "num_arms": 5,
                    "num_users": 2,
                    "dimension": 3,
                    "available_size": 3,
                    "reward_std": 0.1,
                    "preference_std": 0.1,
                },
                "algorithm_overrides": {
                    "s_moss": {"confidence_scale": 1.0},
                    "mo_pqucb_gl": {
                        "group_lasso_steps": 1,
                        "group_lasso_history": 3,
                    },
                },
            }
        )
        results = run_experiment(config)
        self.assertEqual(set(results), set(algorithms))
        for result in results.values():
            self.assertEqual(result.cumulative_regret.shape, (1, 8))
            self.assertTrue(np.all(np.isfinite(result.cumulative_regret)))
        with tempfile.TemporaryDirectory() as directory:
            output = save_results(config, results, Path(directory))
            self.assertTrue((output / "cumulative_regret.pdf").is_file())
            self.assertTrue((output / "cumulative_regret.png").is_file())
            self.assertTrue((output / "preference_error.pdf").is_file())
            self.assertTrue((output / "summary.json").is_file())
            self.assertEqual(list(output.glob("*.npz")), [])


class LLMParsingTests(unittest.TestCase):
    def test_markdown_wrapped_json(self) -> None:
        queries = parse_queries('```json\n[{"id":0,"query":"cheap hotel"}]\n```', 1)
        self.assertEqual(queries, ["cheap hotel"])
        rankings = parse_rankings('[{"id":0,"rank":[2,0,1]}]', 1, 3)
        self.assertEqual(rankings, [[2, 0, 1]])

    def test_wrapped_structured_responses(self) -> None:
        queries = parse_queries(
            '{"results":[{"id":0,"query":"quiet hotel"}]}', 1
        )
        rankings = parse_rankings(
            '{"results":[{"id":0,"rank":{"p0":1,"p1":2,"p2":0}}]}', 1, 3
        )
        self.assertEqual(queries, ["quiet hotel"])
        self.assertEqual(rankings, [[1, 2, 0]])
        self.assertTrue(query_response_schema(2)["additionalProperties"] is False)
        schema = ranking_response_schema(2, 3)
        self.assertEqual(schema["properties"]["results"]["maxItems"], 2)
        rank_schema = schema["properties"]["results"]["items"]["properties"]["rank"]
        self.assertEqual(rank_schema["required"], ["p0", "p1", "p2"])

    def test_duplicate_ranking_is_minimally_repaired(self) -> None:
        with self.assertWarns(RuntimeWarning):
            rankings = parse_rankings(
                '{"results":[{"id":0,"rank":{"p0":2,"p1":2,"p2":0}}]}',
                1,
                3,
            )
        self.assertEqual(rankings, [[2, 1, 0]])

    def test_prompts_are_serializable(self) -> None:
        generated = generation_prompt([[2, 0, 1]], ["a", "b", "c"], [False])
        interpreted = ranking_prompt(["a sample query"], ["a", "b", "c"])
        self.assertIn('"rank": [2, 0, 1]', generated)
        self.assertIn("a sample query", interpreted)


if __name__ == "__main__":
    unittest.main()
