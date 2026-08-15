#!/usr/bin/env python3
"""Run top-m or corruption-rate ablations from one base configuration."""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mo_pqucb.config import load_config  # noqa: E402
from mo_pqucb.runner import run_experiment, save_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--top-m", type=int, nargs="+")
    group.add_argument("--corruption-rates", type=float, nargs="+")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--algorithms", nargs="+")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
    output_directory = args.output_dir or (PROJECT_ROOT / base.output_dir)
    common = {}
    if args.horizon is not None:
        common["horizon"] = args.horizon
    if args.runs is not None:
        common["runs"] = args.runs
    if args.algorithms is not None:
        common["algorithms"] = args.algorithms

    if args.top_m is not None:
        variants = [("top_m_{}".format(value), value, None) for value in args.top_m]
    else:
        variants = [
            ("corruption_{:g}".format(value), None, value)
            for value in args.corruption_rates
        ]

    for suffix, top_m, corruption_rate in variants:
        replacements = dict(common)
        replacements["name"] = "{}_{}".format(base.name, suffix)
        if top_m is not None:
            replacements["top_m"] = top_m
        if corruption_rate is not None:
            environment = dict(base.environment)
            environment["corruption_rate"] = corruption_rate
            replacements["environment"] = environment
            overrides = {
                name: dict(values) for name, values in base.algorithm_overrides.items()
            }
            if "mo_pqucb_gl" in overrides:
                overrides["mo_pqucb_gl"]["alpha"] = (
                    corruption_rate
                    + 1.0 / math.log(int(common.get("horizon", base.horizon)))
                ) / 2.0
            replacements["algorithm_overrides"] = overrides
        config = dataclasses.replace(base, **replacements)
        results = run_experiment(config, project_root=PROJECT_ROOT)
        destination = save_results(config, results, output_directory)
        print("saved figures and summary to {}".format(destination))


if __name__ == "__main__":
    main()
