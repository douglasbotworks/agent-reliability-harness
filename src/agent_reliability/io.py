"""Strict, line-aware JSONL and report file operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .models import GoldRecord, TraceRecord

T = TypeVar("T", bound=BaseModel)


def load_jsonl(path: Path, model: type[T]) -> list[T]:
    records: list[T] = []
    seen: set[str] = set()
    try:
        stream = path.open(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
            try:
                raw = json.loads(line)
                item = model.model_validate(raw)
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                detail = exc.errors(include_url=False) if isinstance(exc, ValidationError) else str(exc)
                raise ValueError(f"{path}:{line_number}: invalid {model.__name__}: {detail}") from exc
            key = getattr(item, "id", getattr(item, "record_id", None))
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate record id {key!r}")
            seen.add(key)
            records.append(item)
    if not records:
        raise ValueError(f"{path}: dataset contains no records")
    return records


def load_gold(path: Path) -> list[GoldRecord]:
    return load_jsonl(path, GoldRecord)


def load_traces(path: Path) -> list[TraceRecord]:
    return load_jsonl(path, TraceRecord)


def write_json(path: Path, value: BaseModel | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
