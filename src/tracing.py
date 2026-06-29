"""Local trace logging for pipeline observability.

This JSONL trace is intentionally independent of LangSmith. That makes the repo
reproducible for reviewers who do not have external service credentials.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.schemas import PipelineTrace


def build_run_id(video_id: str) -> str:
    """Create a stable, readable run identifier."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{video_id}"


def append_trace(trace: PipelineTrace, log_path: str | Path = "eval_logs/eval_trace.jsonl") -> Path:
    """Append a trace record to a JSONL file.

    Args:
        trace: Validated trace object.
        log_path: Destination JSONL path.

    Returns:
        Path to the trace log.
    """

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace.model_dump(), ensure_ascii=False) + "\n")

    return path
