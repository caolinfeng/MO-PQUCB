"""Configuration loading and validation for experiment scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated top-level experiment configuration."""

    name: str
    seed: int
    horizon: int
    runs: int
    top_m: int
    algorithms: List[str]
    environment: Dict[str, Any]
    algorithm: Dict[str, Any] = field(default_factory=dict)
    algorithm_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    llm: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = "results"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        config = cls(
            name=str(raw["name"]),
            seed=int(raw.get("seed", 1)),
            horizon=int(raw["horizon"]),
            runs=int(raw.get("runs", 1)),
            top_m=int(raw.get("top_m", 1)),
            algorithms=[str(name) for name in raw.get("algorithms", ["mo_pqucb"])],
            environment=dict(raw["environment"]),
            algorithm=dict(raw.get("algorithm", {})),
            algorithm_overrides={
                str(name): dict(values)
                for name, values in raw.get("algorithm_overrides", {}).items()
            },
            llm=dict(raw.get("llm", {})),
            output_dir=str(raw.get("output_dir", "results")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.horizon <= 0 or self.runs <= 0:
            raise ValueError("horizon and runs must be positive")
        if self.top_m <= 0:
            raise ValueError("top_m must be positive")
        if not self.algorithms:
            raise ValueError("at least one algorithm is required")
        if "type" not in self.environment:
            raise ValueError("environment.type is required")

    def parameters_for(self, algorithm: str) -> Dict[str, Any]:
        params = dict(self.algorithm)
        params.update(self.algorithm_overrides.get(algorithm, {}))
        return params


def load_config(path: Path) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as stream:
        return ExperimentConfig.from_mapping(json.load(stream))
