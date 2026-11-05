"""Explicit prediction provider interfaces and a deterministic local baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .models import GoldRecord, TraceRecord


class Predictor(Protocol):
    name: str

    def predict(self, item: GoldRecord) -> tuple[Any, str, float | None, float | None, float | None]: ...


@dataclass
class EchoBaseline:
    """Honest deterministic example baseline, useful only when input has a label key."""

    name: str = "echo-baseline"

    def predict(self, item: GoldRecord) -> tuple[Any, str, float | None, float | None, float | None]:
        if not isinstance(item.input, dict) or not isinstance(item.input.get("baseline_label"), str):
            raise RuntimeError(
                "echo baseline requires input.baseline_label; configure a real predictor "
                "instead of treating expected_label as a prediction"
            )
        label = item.input["baseline_label"]
        return {"baseline": True}, label, 0.0, 0.0, None


class MissingProviderError(RuntimeError):
    pass


def require_provider(name: str | None) -> Predictor:
    if name is None:
        raise MissingProviderError(
            "no prediction provider configured; select --provider echo only for datasets "
            "with input.baseline_label, or integrate a Predictor"
        )
    if name == "echo":
        return EchoBaseline()
    raise MissingProviderError(f"unknown provider {name!r}; no network provider is bundled")


def run_predictor(records: list[GoldRecord], predictor: Predictor, task_name: str) -> list[TraceRecord]:
    traces: list[TraceRecord] = []
    for item in records:
        output, label, latency, cost, confidence = predictor.predict(item)
        traces.append(
            TraceRecord(
                schema_version=1,
                record_id=item.id,
                task_name=task_name,
                input=item.input,
                output=output,
                predicted_label=label,
                model_version=predictor.name,
                prompt_version="none",
                latency_ms=latency,
                cost_usd=cost,
                confidence=confidence,
            )
        )
    return traces
