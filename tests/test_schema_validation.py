"""Basic schema tests for Day 1."""

from src.metric_loader import load_metric_bundle


def test_sample_metric_bundle_has_exactly_15_metrics():
    bundle = load_metric_bundle("data/sample_video_metrics.json")
    assert len(bundle.metrics) == 15
