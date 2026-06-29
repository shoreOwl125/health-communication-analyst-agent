"""Shared workflow state for the analyst pipeline.

The state object is intentionally explicit. LangGraph-style workflows are easier
to debug when every node reads from and writes to a clearly documented state.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AnalystState(TypedDict, total=False):
    """Mutable state passed between workflow nodes."""

    # Inputs
    input_path: str
    metrics_path: str
    output_dir: str
    eval_log_path: str

    # Loaded objects
    video_input: Any
    metric_bundle: Any
    transcript: str

    # Retrieval outputs
    rubric_context: list[Any]
    transcript_evidence: list[Any]

    # Report and validation
    report: Any
    qa_result: Any
    retry_count: int
    final_status: str
    warnings: list[str]
    errors: list[str]
