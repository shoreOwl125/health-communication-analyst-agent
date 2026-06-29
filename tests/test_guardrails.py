"""Basic QA guardrail tests for Day 1."""

from src.analyst_agent import generate_deterministic_report
from src.metric_loader import load_metric_bundle, load_transcript, load_video_input
from src.qa_guardrails import validate_report
from src.retrieval import retrieve_for_metrics


def test_sample_report_passes_qa():
    video = load_video_input("data/sample_video_input.json")
    metrics = load_metric_bundle("data/sample_video_metrics.json")
    transcript = load_transcript(video)

    metric_names = [m.metric_name for m in metrics.metrics]
    rubric_context, transcript_evidence = retrieve_for_metrics(metric_names, transcript)

    report = generate_deterministic_report(video, metrics, rubric_context, transcript_evidence)
    result = validate_report(report)

    assert result.passed
    assert result.errors == []
