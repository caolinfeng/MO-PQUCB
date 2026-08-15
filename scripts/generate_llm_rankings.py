#!/usr/bin/env python3
"""Generate and cache two-simulator LLM preference rankings."""

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
from mo_pqucb.environment import build_environment  # noqa: E402
from mo_pqucb.llm_pipeline import (  # noqa: E402
    TRIPADVISOR_OBJECTIVES,
    build_provider,
    generation_prompt,
    parse_queries,
    parse_rankings,
    query_response_schema,
    ranking_response_schema,
    ranking_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provider", choices=["groq", "gemini", "dashscope"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "llm_cache")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--horizon", type=int, help="override the configured horizon")
    parser.add_argument("--runs", type=int, help="override the configured run count")
    return parser.parse_args()


def generate_validated(provider, prompt, schema, parser, retries, failure_path):
    """Retry both API failures and semantically invalid model responses."""

    last_response = ""
    last_error = None
    for _ in range(retries):
        try:
            last_response = provider.generate(prompt, schema)
            return parser(last_response)
        except Exception as error:
            last_error = error
    diagnostic = last_response or repr(last_error)
    failure_path.write_text(diagnostic, encoding="utf-8")
    raise RuntimeError(
        "LLM returned an invalid structured response after {} attempts; raw "
        "response saved to {}".format(retries, failure_path)
    ) from last_error


def infer_rankings(
    provider,
    queries,
    objectives,
    dimension,
    retries,
    failure_root,
    run_index,
    offset,
):
    """Infer rankings, recursively splitting a batch after persistent failures."""

    try:
        return generate_validated(
            provider,
            ranking_prompt(queries, objectives),
            ranking_response_schema(len(queries), dimension),
            lambda text: parse_rankings(text, len(queries), dimension),
            retries,
            failure_root / "failed_run_{:03d}_rankings_{:05d}_{:03d}.txt".format(
                run_index, offset, len(queries)
            ),
        )
    except RuntimeError:
        if len(queries) <= 1:
            raise
        midpoint = len(queries) // 2
        print(
            "ranking batch at {} failed; retrying as {} + {} items".format(
                offset, midpoint, len(queries) - midpoint
            )
        )
        return infer_rankings(
            provider,
            queries[:midpoint],
            objectives,
            dimension,
            retries,
            failure_root,
            run_index,
            offset,
        ) + infer_rankings(
            provider,
            queries[midpoint:],
            objectives,
            dimension,
            retries,
            failure_root,
            run_index,
            offset + midpoint,
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    replacements = {}
    if args.horizon is not None:
        replacements["horizon"] = args.horizon
    if args.runs is not None:
        replacements["runs"] = args.runs
    if replacements:
        config = dataclasses.replace(config, **replacements)
        config.validate()
    if not config.llm:
        raise ValueError("the configuration does not contain an llm section")
    temperature = float(config.llm.get("temperature", 0.5))
    batch_size = int(config.llm.get("batch_size", 50))
    rank_corruption = float(config.llm.get("rank_corruption", 0.0))
    query_corruption = float(config.llm.get("query_corruption", 0.0))
    provider = build_provider(args.provider, args.model, temperature, args.retries)
    cache_root = args.cache_dir / config.name / args.model.replace("/", "_")
    cache_root.mkdir(parents=True, exist_ok=True)

    for run_index in range(config.runs):
        run_seed = config.seed + run_index
        environment = build_environment(
            config.environment, config.horizon, run_seed, project_root=PROJECT_ROOT
        )
        if environment.dimension != len(TRIPADVISOR_OBJECTIVES):
            raise ValueError("default LLM prompts require six TripAdvisor objectives")
        rng = np.random.default_rng(np.random.SeedSequence([run_seed, 991]))
        true_rankings = []
        for round_index in range(config.horizon):
            user = environment.user_at(round_index)
            ranking = environment.ranking_at(round_index, user, environment.dimension)
            if rng.random() < rank_corruption:
                ranking = rng.permutation(ranking)
            true_rankings.append(ranking)

        vague_flags = rng.random(config.horizon) < query_corruption
        run_path = cache_root / "run_{:03d}.json".format(run_index)
        inferred = []
        queries_all = []
        if run_path.is_file():
            with run_path.open("r", encoding="utf-8") as stream:
                checkpoint = json.load(stream)
            inferred = list(checkpoint.get("rankings", []))
            queries_all = list(checkpoint.get("queries", []))
            if len(inferred) != len(queries_all) or len(inferred) > config.horizon:
                raise ValueError("invalid LLM checkpoint: {}".format(run_path))
            print(
                "run {}: resuming from {}/{} rankings".format(
                    run_index, len(inferred), config.horizon
                )
            )

        for start in range(len(inferred), config.horizon, batch_size):
            batch = true_rankings[start : start + batch_size]
            vague = vague_flags[start : start + len(batch)]
            queries = generate_validated(
                provider,
                generation_prompt(batch, TRIPADVISOR_OBJECTIVES, vague),
                query_response_schema(len(batch)),
                lambda text: parse_queries(text, len(batch)),
                args.retries,
                cache_root / "failed_run_{:03d}_batch_{:05d}_queries.txt".format(
                    run_index, start
                ),
            )
            rankings = infer_rankings(
                provider,
                queries,
                TRIPADVISOR_OBJECTIVES,
                environment.dimension,
                args.retries,
                cache_root,
                run_index,
                start,
            )
            inferred.extend(rankings)
            queries_all.extend(queries)
            print("run {}: generated {}/{} rankings".format(run_index, len(inferred), config.horizon))
            temporary_path = run_path.with_suffix(".json.tmp")
            with temporary_path.open("w", encoding="utf-8") as stream:
                json.dump(
                    {"rankings": inferred, "queries": queries_all},
                    stream,
                    ensure_ascii=False,
                    indent=2,
                )
                stream.write("\n")
            temporary_path.replace(run_path)

    with (cache_root / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "config": config.name,
                "provider": args.provider,
                "model": args.model,
                "temperature": temperature,
                "runs": config.runs,
                "horizon": config.horizon,
                "rank_corruption": rank_corruption,
                "query_corruption": query_corruption,
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    print("saved cache to {}".format(cache_root))


if __name__ == "__main__":
    main()
