"""Typed schemas for the Health Video Communication Analyst Agent.

The schemas in this file are intentionally strict. In scientific or regulated
workflows, schema validation is a practical guardrail because it prevents
downstream systems from silently accepting incomplete, malformed, or ambiguous
outputs.

Design principle:
    Metric scores should come from deterministic thesis interpretable metric pipeline, not from
    free-form LLM reasoning. The LLM may interpret and summarize values later,
    but it should not invent the values themselves.
"""

from __future__ import annotations

from typing import Literal, List, Optional
from pydantic import BaseModel, Field, conlist, model_validator
from src.metric_config import EXPECTED_METRICS



class VideoInput(BaseModel):
    """Input metadata for one health-video analysis run."""

    video_id: str
    title: str
    channel: str
    health_topic: str
    query_label: str
    length_seconds: int
    transcript_source: str
    model_context: dict = Field(default_factory=dict)
    audio_context: dict = Field(default_factory=dict)


class MetricScore(BaseModel):
    """One precomputed thesis-style metric.

    These are loaded from JSON or an exported thesis CSV. The agent workflow
    interprets these values; it does not generate them from scratch.
    """

    metric_name: str
    category: Literal["text", "audio", "model"]
    score: float = Field(description="Metric value from exported thesis artifacts.")
    baseline_mean: Optional[float] = Field(
        default=None,
        description="Reference value used for comparison, such as class-1 high-like mean.",
    )
    comparison_to_baseline: Literal["higher", "lower", "near_baseline"]
    interpretation_direction: Literal["positive", "negative", "neutral", "mixed"]
    source_field: str


class MetricBundle(BaseModel):
    """All metric scores for a single video."""

    video_id: str
    metric_source: str
    baseline_name: str
    metrics: conlist(MetricScore, min_length=15, max_length=15)

    @model_validator(mode="after")
    def validate_expected_metrics(self) -> "MetricBundle":
        """Ensure the metric bundle contains exactly the expected metrics."""

        provided = [m.metric_name for m in self.metrics]
        missing = [m for m in EXPECTED_METRICS if m not in provided]
        unexpected = [m for m in provided if m not in EXPECTED_METRICS]
        duplicates = sorted({m for m in provided if provided.count(m) > 1})

        errors = []
        if missing:
            errors.append(f"Missing metrics: {missing}")
        if unexpected:
            errors.append(f"Unexpected metrics: {unexpected}")
        if duplicates:
            errors.append(f"Duplicate metrics: {duplicates}")

        if errors:
            raise ValueError("; ".join(errors))
        return self


class RetrievedEvidence(BaseModel):
    """A retrieved evidence chunk used to ground an interpretation."""

    source: str
    text: str
    score: float = Field(description="Retriever relevance score.")
    rank: int


class MetricInterpretation(BaseModel):
    """Report-ready interpretation for one metric."""

    metric_name: str
    category: Literal["text", "audio", "model"]
    score: float
    comparison_to_baseline: Literal["higher", "lower", "near_baseline"]
    interpretation: str
    evidence_quote: str
    evidence_source: str


class CommunicationQualityReport(BaseModel):
    """Structured report emitted by the analyst workflow."""

    video_id: str
    title: str
    health_topic: str
    overall_summary: str
    metric_interpretations: conlist(MetricInterpretation, min_length=15, max_length=15)
    strengths: conlist(str, min_length=1, max_length=6)
    improvement_opportunities: conlist(str, min_length=1, max_length=6)
    limitations: conlist(str, min_length=1, max_length=6)
    scientist_email_draft: str


class QAResult(BaseModel):
    """Deterministic quality-control result."""

    passed: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PipelineTrace(BaseModel):
    """Local trace record written after every pipeline run."""

    run_id: str
    video_id: str
    input_path: str
    metrics_path: str
    schema_validation: str
    qa_status: str
    metric_count: int
    evidence_coverage_count: int
    revision_count: int
    final_status: str
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
