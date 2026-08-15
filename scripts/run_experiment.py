#!/usr/bin/env python3
"""Run a configured synthetic or real-world MO-PQUCB experiment."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mo_pqucb.config import load_config  # noqa: E402
from mo_pqucb.runner import run_experiment, save_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--horizon", type=int, help="override the configured horizon")
    parser.add_argument("--runs", type=int, help="override the configured run count")
    parser.add_argument(
        "--algorithms", nargs="+", help="run only the listed configured algorithms"
    )
    parser.add_argument("--output-dir", type=Path, help="override the output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    replacements = {}
    if args.horizon is not None:
        replacements["horizon"] = args.horizon
    if args.runs is not None:
        replacements["runs"] = args.runs
    if args.algorithms is not None:
        replacements["algorithms"] = args.algorithms
    if replacements:
        config = dataclasses.replace(config, **replacements)

    results = run_experiment(config, project_root=PROJECT_ROOT)
    destination = args.output_dir or (PROJECT_ROOT / config.output_dir)
    experiment_directory = save_results(config, results, destination)
    for name, result in results.items():
        summary = result.summary()
        print(
            "{}: final regret {:.3f} +/- {:.3f} SE".format(
                name,
                summary["final_regret_mean"],
                summary["final_regret_standard_error"],
            )
        )
    print("saved figures and summary to {}".format(experiment_directory))


if __name__ == "__main__":
    main()
