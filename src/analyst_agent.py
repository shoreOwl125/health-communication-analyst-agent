"""Report generation logic for the analyst workflow.

Day 1 uses a deterministic report generator rather than an LLM. This is
intentional: it proves the pipeline, schemas, retrieval, output writing, and
guardrails before adding model variability.

Later, this module can expose an LLM-backed `generate_report` function that uses
the same input and output schemas.
"""

from __future__ import annotations

from src.schemas import (
    CommunicationQualityReport,
    MetricBundle,
    MetricInterpretation,
    RetrievedEvidence,
    VideoInput,
)


def generate_deterministic_report(
    video_input: VideoInput,
    metric_bundle: MetricBundle,
    rubric_context: list[RetrievedEvidence],
    transcript_evidence: list[RetrievedEvidence],
) -> CommunicationQualityReport:
    """Generate a structured report from precomputed metrics and retrieved evidence.

    Args:
        video_input: Video metadata.
        metric_bundle: Validated 15-metric bundle.
        rubric_context: Retrieved metric/rubric context.
        transcript_evidence: Retrieved transcript chunks.

    Returns:
        A validated CommunicationQualityReport.
    """

    evidence_fallback = (
        transcript_evidence[0].text
        if transcript_evidence
        else "No transcript evidence was retrieved for this metric."
    )
    evidence_source_fallback = (
        transcript_evidence[0].source if transcript_evidence else "not_available"
    )

    interpretations: list[MetricInterpretation] = []
    for metric in metric_bundle.metrics:
        evidence = _select_evidence_for_metric(metric.metric_name, transcript_evidence)
        evidence_text = evidence.text if evidence else evidence_fallback
        evidence_source = evidence.source if evidence else evidence_source_fallback

        interpretations.append(
            MetricInterpretation(
                metric_name=metric.metric_name,
                category=metric.category,
                score=metric.score,
                comparison_to_baseline=metric.comparison_to_baseline,
                interpretation=_interpret_metric(metric.metric_name, metric.comparison_to_baseline),
                evidence_quote=_short_quote(evidence_text),
                evidence_source=evidence_source,
            )
        )

    strengths = [
        "The video uses a clear list-based structure that supports stepwise comprehension.",
        "The transcript includes direct viewer-oriented language and practical next steps.",
        "The metric bundle suggests above-baseline topic specificity and actionability.",
    ]

    improvement_opportunities = [
        "Maintain a cautious distinction between educational guidance and medical advice.",
        "Keep technical medical vocabulary paired with plain-language explanations.",
        "Use audio delivery patterns as supporting context rather than standalone quality evidence.",
    ]

    limitations = [
        "This report analyzes communication-quality signals and relative engagement patterns, not medical accuracy.",
        "Metric values are precomputed thesis-style artifacts and should be reviewed before production use.",
        "Audio metrics can be affected by recording conditions, microphone quality, and background noise.",
    ]

    email_draft = _build_scientist_email(video_input, strengths, improvement_opportunities)

    return CommunicationQualityReport(
        video_id=video_input.video_id,
        title=video_input.title,
        health_topic=video_input.health_topic,
        overall_summary=(
            f"This video shows several communication signals associated with stronger health-video "
            f"engagement in the thesis framework, including structured explanation, audience-oriented "
            f"language, and actionable educational framing. The report should be interpreted as a "
            f"communication-quality assessment, not a clinical review."
        ),
        metric_interpretations=interpretations,
        strengths=strengths,
        improvement_opportunities=improvement_opportunities,
        limitations=limitations,
        scientist_email_draft=email_draft,
    )


def revise_report_with_qa_feedback(
    report: CommunicationQualityReport,
    qa_errors: list[str],
) -> CommunicationQualityReport:
    """Apply a conservative deterministic revision.

    This placeholder keeps the Day 1 pipeline simple. In a later LLM-backed
    version, the QA errors would be sent back to the Analyst Agent as structured
    feedback for one controlled regeneration attempt.
    """

    if not qa_errors:
        return report

    # Conservative revision: add a limitation noting that automated QA found an issue.
    revised_limitations = list(report.limitations)
    revised_limitations.append(
        "Automated QA flagged issues during generation; this report should be manually reviewed."
    )
    report.limitations = revised_limitations[:6]
    return report


def _interpret_metric(metric_name: str, comparison: str) -> str:
    """Create a cautious interpretation sentence for one metric."""

    readable_name = metric_name.replace("_", " ")
    if comparison == "higher":
        return (
            f"The {readable_name} value is higher than the selected baseline, which may indicate "
            f"a stronger presence of this communication signal in the video."
        )
    if comparison == "lower":
        return (
            f"The {readable_name} value is lower than the selected baseline. Depending on the metric, "
            f"this may indicate either a reduced signal or a potentially favorable moderation."
        )
    return (
        f"The {readable_name} value is near the selected baseline, suggesting this signal is not a "
        f"major differentiator for this video."
    )


def _select_evidence_for_metric(
    metric_name: str,
    transcript_evidence: list[RetrievedEvidence],
) -> RetrievedEvidence | None:
    """Select a transcript chunk to attach to a metric interpretation.

    Day 1 uses a simple heuristic. Later, this can use metric-specific evidence
    retrieval or transcript span attribution.
    """

    query_terms = set(metric_name.replace("_", " ").split())
    best: RetrievedEvidence | None = None
    best_overlap = -1

    for evidence in transcript_evidence:
        tokens = set(evidence.text.lower().split())
        overlap = len(query_terms.intersection(tokens))
        if overlap > best_overlap:
            best = evidence
            best_overlap = overlap

    return best


def _short_quote(text: str, max_chars: int = 260) -> str:
    """Return a compact evidence quote suitable for markdown reports."""

    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _build_scientist_email(
    video_input: VideoInput,
    strengths: list[str],
    improvement_opportunities: list[str],
) -> str:
    """Create a concise stakeholder-facing email draft."""

    return (
        f"Subject: Communication-quality summary for {video_input.video_id}\n\n"
        f"Hi,\n\n"
        f"I completed a communication-quality analysis for the video "
        f"'{video_input.title}' in the {video_input.health_topic} topic area. "
        f"The strongest signals were: {strengths[0]} {strengths[1]}\n\n"
        f"The main review opportunity is to ensure that practical health guidance remains "
        f"clearly framed as educational information rather than medical advice. A second "
        f"opportunity is to pair medical vocabulary with plain-language explanations.\n\n"
        f"I attached the structured report with metric-level interpretations, supporting "
        f"transcript evidence, and limitations.\n\n"
        f"Best,\n"
        f"Health Video Communication Analyst"
    )
