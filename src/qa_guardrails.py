"""Deterministic QA guardrails for generated reports.

These checks are deliberately simple and transparent. In scientific workflows,
quality-control logic should not depend only on another LLM call. Code-based
checks provide a reproducible baseline.
"""

from __future__ import annotations

from src.schemas import CommunicationQualityReport, EXPECTED_METRICS, QAResult


BANNED_UNSUPPORTED_CLAIMS = [
    "improves patient outcomes",
    "is clinically accurate",
    "clinically proven",
    "proves causality",
    "guarantees engagement",
    "guarantees outcomes",
]


TECHNICAL_JARGON_TERMS = [
    "heteroscedasticity",
    "eigenvector",
    "bayesian optimization",
    "gradient boosted residualization",
    "embedding manifold",
]


def validate_report(report: CommunicationQualityReport) -> QAResult:
    """Run deterministic guardrail checks over a report.

    Args:
        report: Structured communication-quality report.

    Returns:
        QAResult with pass/fail status and actionable errors/warnings.
    """

    errors: list[str] = []
    warnings: list[str] = []

    metric_names = [m.metric_name for m in report.metric_interpretations]

    if len(metric_names) != 15:
        errors.append(f"Expected exactly 15 metric interpretations, found {len(metric_names)}.")

    missing = [m for m in EXPECTED_METRICS if m not in metric_names]
    unexpected = [m for m in metric_names if m not in EXPECTED_METRICS]

    if missing:
        errors.append(f"Missing expected metrics: {missing}")
    if unexpected:
        errors.append(f"Unexpected metric names: {unexpected}")

    missing_evidence = [
        m.metric_name
        for m in report.metric_interpretations
        if not m.evidence_quote.strip() or not m.evidence_source.strip()
    ]
    if missing_evidence:
        errors.append(f"Metrics missing evidence references: {missing_evidence}")

    report_text = report.model_dump_json().lower()
    for phrase in BANNED_UNSUPPORTED_CLAIMS:
        if phrase in report_text:
            errors.append(f"Unsupported or unsafe claim detected: '{phrase}'")

    email_lower = report.scientist_email_draft.lower()
    jargon_hits = [term for term in TECHNICAL_JARGON_TERMS if term in email_lower]
    if jargon_hits:
        warnings.append(f"Email draft may contain technical jargon: {jargon_hits}")

    if len(report.scientist_email_draft) > 2500:
        warnings.append("Email draft is long for a stakeholder-facing summary.")

    return QAResult(passed=not errors, errors=errors, warnings=warnings)


def evidence_coverage_count(report: CommunicationQualityReport) -> int:
    """Count metric interpretations with evidence quotes and source labels."""

    return sum(
        1
        for metric in report.metric_interpretations
        if metric.evidence_quote.strip() and metric.evidence_source.strip()
    )
