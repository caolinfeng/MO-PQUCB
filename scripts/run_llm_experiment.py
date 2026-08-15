#!/usr/bin/env python3
"""Evaluate MO-PQUCB from cached LLM-inferred rankings."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mo_pqucb.config import load_config  # noqa: E402
from mo_pqucb.runner import run_experiment, save_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--horizon", type=int, help="override the configured horizon")
    parser.add_argument("--runs", type=int, help="override the configured run count")
    args = parser.parse_args()
    config = load_config(args.config)
    replacements = {}
    if args.horizon is not None:
        replacements["horizon"] = args.horizon
    if args.runs is not None:
        replacements["runs"] = args.runs
    if replacements:
        config = dataclasses.replace(config, **replacements)
        config.validate()
    manifest_path = args.cache_dir / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if int(manifest.get("horizon", -1)) != config.horizon:
            raise ValueError("cache horizon does not match the experiment horizon")
        if int(manifest.get("runs", -1)) != config.runs:
            raise ValueError("cache run count does not match the experiment run count")
    rankings = {}
    for run_index in range(config.runs):
        path = args.cache_dir / "run_{:03d}.json".format(run_index)
        with path.open("r", encoding="utf-8") as stream:
            rankings[run_index] = np.asarray(json.load(stream)["rankings"], dtype=int)
    results = run_experiment(config, project_root=PROJECT_ROOT, query_rankings=rankings)
    destination = args.output_dir or (PROJECT_ROOT / config.output_dir)
    output = save_results(config, results, destination)
    print("saved figures and summary to {}".format(output))


if __name__ == "__main__":
    main()
