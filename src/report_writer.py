"""Write analyst outputs to local files."""

from __future__ import annotations

from pathlib import Path

from src.schemas import CommunicationQualityReport


def write_report_outputs(
    report: CommunicationQualityReport,
    output_dir: str | Path = "outputs",
) -> tuple[Path, Path]:
    """Write the markdown report and scientist email draft.

    Args:
        report: Structured report object.
        output_dir: Directory for generated outputs.

    Returns:
        Tuple of paths: (report_path, email_path).
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "communication_quality_report.md"
    email_path = out_dir / "scientist_email_draft.txt"

    report_path.write_text(_render_markdown_report(report), encoding="utf-8")
    email_path.write_text(report.scientist_email_draft, encoding="utf-8")

    return report_path, email_path


def _render_markdown_report(report: CommunicationQualityReport) -> str:
    """Render a structured markdown report."""

    lines: list[str] = []
    lines.append(f"# Communication Quality Report: {report.video_id}")
    lines.append("")
    lines.append(f"**Title:** {report.title}")
    lines.append(f"**Health topic:** {report.health_topic}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(report.overall_summary)
    lines.append("")
    lines.append("## Metric Interpretations")
    lines.append("")
    lines.append("| Metric | Category | Score | Baseline comparison | Interpretation | Evidence |")
    lines.append("|---|---:|---:|---|---|---|")

    for metric in report.metric_interpretations:
        evidence = metric.evidence_quote.replace("|", "\|")
        interpretation = metric.interpretation.replace("|", "\|")
        lines.append(
            f"| `{metric.metric_name}` | {metric.category} | {metric.score:.2f} | "
            f"{metric.comparison_to_baseline} | {interpretation} | {evidence} |"
        )

    lines.append("")
    lines.append("## Strengths")
    lines.extend([f"- {item}" for item in report.strengths])
    lines.append("")
    lines.append("## Improvement Opportunities")
    lines.extend([f"- {item}" for item in report.improvement_opportunities])
    lines.append("")
    lines.append("## Limitations")
    lines.extend([f"- {item}" for item in report.limitations])
    lines.append("")
    lines.append("## Scientist Email Draft")
    lines.append("")
    lines.append("```text")
    lines.append(report.scientist_email_draft)
    lines.append("```")
    lines.append("")

    return "\n".join(lines)
