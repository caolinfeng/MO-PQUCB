"""Provider-neutral LLM query generation and ranking interpretation."""

from __future__ import annotations

import json
import os
import re
import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence


TRIPADVISOR_OBJECTIVES = [
    "cleanliness",
    "location",
    "rooms",
    "service",
    "sleep quality",
    "value",
]


def query_response_schema(expected: int) -> Dict[str, Any]:
    """Strict schema for Simulator-1 output."""

    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "minItems": expected,
                "maxItems": expected,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "query": {"type": "string"},
                    },
                    "required": ["id", "query"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def ranking_response_schema(expected: int, dimension: int) -> Dict[str, Any]:
    """Strict schema for Simulator-2 output."""

    position_names = ["p{}".format(index) for index in range(dimension)]
    rank_properties = {
        name: {
            "type": "integer",
            "minimum": 0,
            "maximum": dimension - 1,
        }
        for name in position_names
    }
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "minItems": expected,
                "maxItems": expected,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "rank": {
                            "type": "object",
                            "properties": rank_properties,
                            "required": position_names,
                            "additionalProperties": False,
                        },
                    },
                    "required": ["id", "rank"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def _extract_json(text: str) -> Any:
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "")
    decoder = json.JSONDecoder()
    for position, character in enumerate(cleaned):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[position:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("model response does not contain valid JSON")


def parse_queries(text: str, expected: int) -> List[str]:
    payload = _extract_json(text)
    if isinstance(payload, dict):
        payload = payload.get("results", payload.get("queries", []))
    if not isinstance(payload, list) or len(payload) != expected:
        raise ValueError("query response has an unexpected batch size")
    queries = []
    for item in payload:
        query = item.get("query") if isinstance(item, dict) else item
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query response contains an empty query")
        queries.append(query.strip())
    return queries


def parse_rankings(text: str, expected: int, dimension: int) -> List[List[int]]:
    payload = _extract_json(text)
    if isinstance(payload, dict):
        payload = payload.get("results", payload.get("rankings", []))
    if not isinstance(payload, list) or len(payload) != expected:
        raise ValueError("ranking response has an unexpected batch size")
    rankings: List[List[int]] = []
    required = set(range(dimension))
    repaired = 0
    for item in payload:
        ranking = item.get("rank") if isinstance(item, dict) else item
        if isinstance(ranking, dict):
            position_names = ["p{}".format(index) for index in range(dimension)]
            if set(ranking) == set(position_names):
                ranking = [ranking[name] for name in position_names]
        if isinstance(ranking, str):
            compact = ranking.strip().strip("[]")
            if "," in compact:
                ranking = [part.strip() for part in compact.split(",")]
            elif compact.isdigit() and len(compact) == dimension:
                ranking = list(compact)
        if not isinstance(ranking, list):
            raise ValueError("ranking response is not a list or encoded permutation")
        parsed = [int(value) for value in ranking]
        if len(parsed) != dimension or not set(parsed).issubset(required):
            raise ValueError("ranking must be a permutation of all objective indices")
        if set(parsed) != required:
            missing = iter(sorted(required - set(parsed)))
            seen = set()
            for position, value in enumerate(parsed):
                if value in seen:
                    parsed[position] = next(missing)
                else:
                    seen.add(value)
            repaired += 1
        rankings.append(parsed)
    if repaired:
        warnings.warn(
            "repaired {} LLM ranking(s) with duplicated objective indices".format(
                repaired
            ),
            RuntimeWarning,
        )
    return rankings


def generation_prompt(
    rankings: Sequence[Sequence[int]],
    objectives: Sequence[str],
    vague_flags: Sequence[bool],
) -> str:
    records = [
        {"id": index, "rank": list(map(int, ranking)), "vague": bool(vague_flags[index])}
        for index, ranking in enumerate(rankings)
    ]
    objective_map = {str(index): name for index, name in enumerate(objectives)}
    return (
        "You are Simulator-1 in a hotel recommendation experiment. Convert each "
        "objective ranking into one natural user search query. Earlier indices in rank "
        "are more important. If vague=true, intentionally use ambiguous wording and "
        "omit some priorities. Return only a JSON object with key results, whose "
        "value is an array in the same order "
        "using objects with keys id and query. Objective map: {}. Inputs: {}"
    ).format(json.dumps(objective_map), json.dumps(records))


def ranking_prompt(queries: Sequence[str], objectives: Sequence[str]) -> str:
    records = [{"id": index, "query": query} for index, query in enumerate(queries)]
    objective_map = {str(index): name for index, name in enumerate(objectives)}
    return (
        "You are Simulator-2 in a hotel recommendation experiment. Infer a complete "
        "priority ranking of objective indices for every query. Return only a JSON "
        "object with key results, whose value is an array of objects containing keys "
        "id and rank. The rank value MUST be an object with one separate integer "
        "per position, from most to least important; for example, use "
        "{{\"rank\":{{\"p0\":4,\"p1\":1,\"p2\":2,\"p3\":0,\"p4\":5,\"p5\":3}}}}, "
        "never [412053] or \"412053\". Every objective index must occur exactly "
        "once. Preserve "
        "the input order. Objective map: {}. "
        "Queries: {}"
    ).format(json.dumps(objective_map), json.dumps(records))


class TextProvider(ABC):
    @abstractmethod
    def generate(
        self, prompt: str, schema: Optional[Mapping[str, Any]] = None
    ) -> str:
        raise NotImplementedError


class GroqProvider(TextProvider):
    def __init__(self, model: str, temperature: float) -> None:
        try:
            from groq import Groq
        except ImportError as error:
            raise RuntimeError("install the optional 'groq' package") from error
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self.client = Groq(api_key=key)
        self.model = model
        self.temperature = temperature

    def generate(
        self, prompt: str, schema: Optional[Mapping[str, Any]] = None
    ) -> str:
        options: Dict[str, Any] = {}
        if schema is not None:
            options["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "simulator_response",
                    "strict": True,
                    "schema": dict(schema),
                },
            }
            if self.model.startswith("openai/gpt-oss-"):
                options["reasoning_effort"] = "low"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            **options,
        )
        return str(response.choices[0].message.content)


class GeminiProvider(TextProvider):
    def __init__(self, model: str, temperature: float) -> None:
        try:
            from google import genai
        except ImportError as error:
            raise RuntimeError("install the optional 'google-genai' package") from error
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.client = genai.Client(api_key=key)
        self.model = model
        self.temperature = temperature

    def generate(
        self, prompt: str, schema: Optional[Mapping[str, Any]] = None
    ) -> str:
        del schema
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": self.temperature},
        )
        return str(response.text)


class DashScopeProvider(TextProvider):
    def __init__(self, model: str, temperature: float) -> None:
        try:
            import dashscope
            from dashscope import Generation
        except ImportError as error:
            raise RuntimeError("install the optional 'dashscope' package") from error
        key = os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        dashscope.api_key = key
        self.generation = Generation
        self.model = model
        self.temperature = temperature

    def generate(
        self, prompt: str, schema: Optional[Mapping[str, Any]] = None
    ) -> str:
        del schema
        response = self.generation.call(
            model=self.model, prompt=prompt, temperature=self.temperature
        )
        if getattr(response, "status_code", 200) != 200:
            raise RuntimeError("DashScope request failed: {}".format(response.message))
        return str(response.output.text)


class RetryingProvider(TextProvider):
    def __init__(self, provider: TextProvider, retries: int = 5, delay: float = 2.0) -> None:
        self.provider = provider
        self.retries = retries
        self.delay = delay

    def generate(
        self, prompt: str, schema: Optional[Mapping[str, Any]] = None
    ) -> str:
        error: Exception = RuntimeError("request was not attempted")
        for attempt in range(self.retries):
            try:
                return self.provider.generate(prompt, schema)
            except Exception as caught:  # provider SDKs expose different error types
                error = caught
                status_code = getattr(caught, "status_code", None)
                if status_code is not None and status_code < 500 and status_code != 429:
                    raise
                if attempt + 1 < self.retries:
                    time.sleep(self.delay * (2 ** attempt))
        raise RuntimeError("LLM request failed after retries") from error


def build_provider(
    provider: str, model: str, temperature: float, retries: int = 5
) -> TextProvider:
    name = provider.lower()
    if name == "groq":
        client: TextProvider = GroqProvider(model, temperature)
    elif name == "gemini":
        client = GeminiProvider(model, temperature)
    elif name in {"dashscope", "qwen"}:
        client = DashScopeProvider(model, temperature)
    else:
        raise ValueError("unsupported provider: {}".format(provider))
    return RetryingProvider(client, retries=retries)
