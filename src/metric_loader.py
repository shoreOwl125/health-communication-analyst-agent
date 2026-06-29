"""Load exported thesis artifacts for one video.

In the full thesis project, metric values are produced by upstream notebooks or
scripts. This module makes the downstream analyst layer independent of GPU-heavy
training workflows by loading precomputed values from JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.schemas import VideoInput, MetricBundle


def load_video_input(input_path: str | Path) -> VideoInput:
    """Load and validate video-level metadata.

    Args:
        input_path: Path to a JSON file such as `data/sample_video_input.json`.

    Returns:
        A validated VideoInput object.
    """

    path = Path(input_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return VideoInput.model_validate(payload)


def load_metric_bundle(metrics_path: str | Path) -> MetricBundle:
    """Load and validate the 15-metric bundle for one video.

    Args:
        metrics_path: Path to a JSON file such as `data/sample_video_metrics.json`.

    Returns:
        A validated MetricBundle object.
    """

    path = Path(metrics_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return MetricBundle.model_validate(payload)


def load_transcript(video_input: VideoInput, base_dir: str | Path = ".") -> str:
    """Load transcript text referenced by the video input JSON.

    Args:
        video_input: Validated video metadata.
        base_dir: Repository root or working directory.

    Returns:
        Clean transcript text.
    """

    transcript_path = Path(base_dir) / video_input.transcript_source
    return transcript_path.read_text(encoding="utf-8").strip()
