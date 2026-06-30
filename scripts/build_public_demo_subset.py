"""Build a small public/demo subset from private thesis CSV exports.

This script reads private interpretable metric pipeline artifacts from `data/private/` and writes a
small, pipeline-ready demo dataset to `data/public/`.

Inputs:
    data/private/interpretable_feature_dataset_v3.csv
    data/private/healthVideoData_preprocessed.csv

Outputs:
    data/public/video_inputs/<video_id>.json
    data/public/metric_bundles/<video_id>.json
    data/public/transcripts/<video_id>.txt
    data/public/manifest.jsonl
    data/public/README.md

The generated files are intentionally small and structured so they can be used
by the analyst agent for the purpose of demoing the agent.

Recommended use:
    python scripts/build_public_demo_subset.py \
        --features data/private/interpretable_feature_dataset_v3.csv \
        --videos data/private/healthVideoData_preprocessed.csv \
        --output-dir data/public \
        --n 100 \
        --label 0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metric_config import METRIC_DEFINITIONS, SOURCE_COLUMNS


DEFAULT_FEATURES_PATH = "data/private/interpretable_feature_dataset_v3.csv"
DEFAULT_VIDEOS_PATH = "data/private/healthVideoData_preprocessed.csv"
DEFAULT_OUTPUT_DIR = "data/public"


VIDEO_COLUMNS = [
    "video_id",
    "title",
    "channel",
    "published_at",
    "view_count",
    "like_count",
    "comment_count",
    "query_label",
    "health_topic",
    "rank",
    "transcript",
    "transcript_readable",
    "transcript_sentences",
    "length_seconds",
    "age_days",
    "upload_year",
]


FEATURE_COLUMNS = [
    "video_id",
    "test_set",
    "length_seconds",
    "ffnn_p",
    "xgb_p",
    "label",
    *SOURCE_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Create a 100-video public/demo subset from private thesis CSV files."
    )
    parser.add_argument(
        "--features",
        default=DEFAULT_FEATURES_PATH,
        help="Path to interpretable_feature_dataset_v3.csv.",
    )
    parser.add_argument(
        "--videos",
        default=DEFAULT_VIDEOS_PATH,
        help="Path to healthVideoData_preprocessed.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated public/demo artifacts.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of videos to export.",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=0,
        help="Target class label to select. Use 0 for lower-engagement / improvement-potential examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--max-transcript-chars",
        type=int,
        default=6000,
        help=(
            "Maximum transcript characters to write per video. "
            "Use a smaller value for public GitHub demos; use a larger value for local-only experiments."
        ),
    )
    parser.add_argument(
        "--target-baseline",
        default="data/reference/class1_high_like_metric_means.json",
        help="Optional JSON file containing target reference metric means, such as class-1 high-like means.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the conversion process."""

    args = parse_args()

    features_path = Path(args.features)
    videos_path = Path(args.videos)
    output_dir = Path(args.output_dir)

    validate_input_paths(features_path, videos_path)

    create_output_dirs(output_dir)

    feature_df = load_feature_dataset(features_path)
    selected_features = select_label_subset(
        feature_df=feature_df,
        label_value=args.label,
        n=args.n,
        seed=args.seed,
    )

    selected_video_ids = set(selected_features["video_id"].astype(str))
    video_df = load_selected_video_rows(
        videos_path=videos_path,
        selected_video_ids=selected_video_ids,
    )

    merged = selected_features.merge(video_df, on="video_id", how="inner", suffixes=("_feat", ""))

    if len(merged) < args.n:
        print(
            f"Warning: Requested {args.n} videos, but only {len(merged)} had matching transcript rows."
        )

    baseline_means, baseline_name = load_or_compute_baseline_means(
        feature_df=feature_df,
        target_baseline_path=Path(args.target_baseline),
    )

    manifest_records = []
    for _, row in merged.iterrows():
        record = export_one_video(
            row=row,
            baseline_means=baseline_means,
            baseline_name=baseline_name,
            output_dir=output_dir,
            max_transcript_chars=args.max_transcript_chars,
        )
        manifest_records.append(record)

    write_manifest(output_dir, manifest_records)
    write_public_readme(output_dir, args, len(manifest_records))

    print(f"Created {len(manifest_records)} demo examples in: {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.jsonl'}")


def validate_input_paths(features_path: Path, videos_path: Path) -> None:
    """Fail early if required private inputs are missing."""

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")
    if not videos_path.exists():
        raise FileNotFoundError(f"Missing video metadata file: {videos_path}")


def create_output_dirs(output_dir: Path) -> None:
    """Create the public/demo output directory structure."""

    for subdir in ["video_inputs", "metric_bundles", "transcripts"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def load_feature_dataset(features_path: Path) -> pd.DataFrame:
    """Load only the columns needed from the interpretable feature dataset."""

    df = pd.read_csv(
        features_path,
        usecols=lambda col: col in FEATURE_COLUMNS,
        dtype={"video_id": "string"},
        low_memory=False,
    )

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    df["video_id"] = df["video_id"].astype(str)
    return df


def select_label_subset(
    feature_df: pd.DataFrame,
    label_value: int,
    n: int,
    seed: int,
) -> pd.DataFrame:
    """Select a reproducible subset of labeled videos.

    Args:
        feature_df: Interpretable feature dataset.
        label_value: Target label, usually 0 for improvement-potential examples.
        n: Number of rows to select.
        seed: Random seed.

    Returns:
        DataFrame with selected rows.
    """

    labels_numeric = pd.to_numeric(feature_df["label"], errors="coerce")
    candidates = feature_df.loc[labels_numeric == label_value].copy()

    # Require all reportable normalized metrics.
    candidates = candidates.dropna(subset=SOURCE_COLUMNS)

    if candidates.empty:
        raise ValueError(f"No candidates found for label={label_value} with complete metrics.")

    if len(candidates) < n:
        print(
            f"Warning: only {len(candidates)} complete label={label_value} candidates found; exporting all."
        )
        return candidates.sort_values("video_id").reset_index(drop=True)

    return (
        candidates.sample(n=n, random_state=seed)
        .sort_values("video_id")
        .reset_index(drop=True)
    )


def load_selected_video_rows(
    videos_path: Path,
    selected_video_ids: set[str],
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """Load matching rows from the large preprocessed video dataset.

    The preprocessed CSV can be large, so this function reads it in chunks and
    keeps only the selected video IDs.

    Args:
        videos_path: Path to healthVideoData_preprocessed.csv.
        selected_video_ids: Video IDs to retain.
        chunksize: Number of rows per pandas chunk.

    Returns:
        DataFrame with rows matching selected video IDs.
    """

    selected_chunks = []

    for chunk in pd.read_csv(
        videos_path,
        usecols=lambda col: col in VIDEO_COLUMNS,
        dtype={"video_id": "string"},
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["video_id"] = chunk["video_id"].astype(str)
        matched = chunk[chunk["video_id"].isin(selected_video_ids)].copy()
        if not matched.empty:
            selected_chunks.append(matched)

    if not selected_chunks:
        raise ValueError("No selected video IDs were found in the video metadata file.")

    video_df = pd.concat(selected_chunks, ignore_index=True)

    # Prefer transcript_readable, but keep transcript as fallback.
    if "transcript_readable" not in video_df.columns and "transcript" not in video_df.columns:
        raise ValueError("Video dataset must contain transcript_readable or transcript.")

    return video_df

def load_or_compute_baseline_means(
    feature_df: pd.DataFrame,
    target_baseline_path: Path,
) -> tuple[dict[str, float], str]:
    """Load target baseline means from JSON or compute dataset means as fallback.

    The preferred baseline for this MVP is the class-1 high-like reference group.
    This lets the report compare lower-engagement videos against the average
    metric profile of higher-engagement videos.

    Args:
        feature_df: Interpretable feature dataset.
        target_baseline_path: Path to JSON file with source-column metric means.

    Returns:
        Tuple of baseline dictionary and baseline name.
    """

    if target_baseline_path.exists():
        with target_baseline_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        metric_payload = payload.get("metrics", {})
        missing = [col for col in SOURCE_COLUMNS if col not in metric_payload]
        if missing:
            raise ValueError(
                f"Target baseline file is missing required metric columns: {missing}"
            )

        baselines = {
            col: float(metric_payload[col])
            for col in SOURCE_COLUMNS
        }

        baseline_name = payload.get("baseline_name", "target_reference_baseline")
        return baselines, baseline_name

    print(
        f"Warning: target baseline file not found at {target_baseline_path}. "
        "Falling back to dataset-wide metric means."
    )
    return compute_baseline_means(feature_df), "dataset_wide_mean"

def compute_baseline_means(feature_df: pd.DataFrame) -> dict[str, float]:
    """Compute metric baselines from the labeled feature dataset.

    Baselines are used only for comparison labels such as higher/lower/near
    baseline. They are not causal or normative thresholds.
    """

    baselines: dict[str, float] = {}
    for metric in METRIC_DEFINITIONS:
        values = pd.to_numeric(feature_df[metric.source_column], errors="coerce")
        baselines[metric.source_column] = float(values.mean(skipna=True))
    return baselines


def export_one_video(
    row: pd.Series,
    baseline_means: dict[str, float],
    baseline_name: str,
    output_dir: Path,
    max_transcript_chars: int,
) -> dict[str, Any]:
    """Export one video into transcript, input JSON, and metric-bundle JSON."""

    video_id = str(row["video_id"])
    safe_id = safe_filename(video_id)

    transcript_text = choose_transcript_text(row)
    transcript_text = trim_transcript(transcript_text, max_chars=max_transcript_chars)

    transcript_rel_path = f"data/public/transcripts/{safe_id}.txt"
    transcript_out_path = output_dir / "transcripts" / f"{safe_id}.txt"
    transcript_out_path.write_text(transcript_text, encoding="utf-8")

    video_input = build_video_input_json(
        row=row,
        video_id=video_id,
        transcript_rel_path=transcript_rel_path,
    )

    metric_bundle = build_metric_bundle_json(
        row=row,
        video_id=video_id,
        baseline_means=baseline_means,
        baseline_name=baseline_name,
    )

    input_path = output_dir / "video_inputs" / f"{safe_id}.json"
    metrics_path = output_dir / "metric_bundles" / f"{safe_id}.json"

    write_json(input_path, video_input)
    write_json(metrics_path, metric_bundle)

    return {
        "video_id": video_id,
        "title": as_str(row.get("title", "")),
        "health_topic": as_str(row.get("health_topic", "")),
        "query_label": as_str(row.get("query_label", "")),
        "label": safe_int(row.get("label")),
        "xgb_p": safe_float(row.get("xgb_p")),
        "ffnn_p": safe_float(row.get("ffnn_p")),
        "input_path": str(input_path),
        "metrics_path": str(metrics_path),
        "transcript_path": str(transcript_out_path),
    }


def build_video_input_json(
    row: pd.Series,
    video_id: str,
    transcript_rel_path: str,
) -> dict[str, Any]:
    """Build the video input JSON consumed by `run_pipeline.py`."""

    length_seconds = safe_int(row.get("length_seconds"))
    if length_seconds is None:
        length_seconds = safe_int(row.get("length_seconds_feat")) or 0

    return {
        "video_id": video_id,
        "title": as_str(row.get("title", "Untitled health video")),
        "channel": as_str(row.get("channel", "Unknown channel")),
        "health_topic": as_str(row.get("health_topic", "unknown")),
        "query_label": as_str(row.get("query_label", "unknown")),
        "length_seconds": length_seconds,
        "transcript_source": transcript_rel_path,
        "model_context": {
            "label_type": "thesis_interpretable_feature_label",
            "true_label": safe_int(row.get("label")),
            "target_subset": "label_0_lower_engagement_or_improvement_potential",
            "xgb_p": safe_float(row.get("xgb_p")),
            "ffnn_p": safe_float(row.get("ffnn_p")),
            "test_set": safe_bool(row.get("test_set")),
            "note": (
                "Prediction fields are exported thesis artifacts. "
                "They are provided as model evidence, not recalculated by this MVP."
            ),
        },
        "audio_context": {
            "audio_feature_source": "interpretable_feature_dataset_v3.csv",
            "note": "Audio metrics are normalized thesis-derived summary features.",
        },
    }


def build_metric_bundle_json(
    row: pd.Series,
    video_id: str,
    baseline_means: dict[str, float],
    baseline_name: str,
) -> dict[str, Any]:
    """Build the metric bundle JSON consumed by the analyst workflow."""

    metrics = []

    for metric_def in METRIC_DEFINITIONS:
        source_col = metric_def.source_column
        score = safe_float(row.get(source_col))
        baseline_mean = safe_float(baseline_means[source_col])
        metrics.append(
            {
                "metric_name": metric_def.metric_name,
                "category": metric_def.category,
                "score": score,
                "baseline_mean": baseline_mean,
                "comparison_to_baseline": compare_to_baseline(score, baseline_mean),
                "interpretation_direction": metric_def.interpretation_direction,
                "source_field": source_col,
            }
        )

    return {
        "video_id": video_id,
        "metric_source": "interpretable_feature_dataset_v3.csv",
        "baseline_name": baseline_name,
        "metrics": metrics,
    }


def choose_transcript_text(row: pd.Series) -> str:
    """Choose transcript_readable with fallback to transcript."""

    transcript_readable = row.get("transcript_readable")
    transcript = row.get("transcript")

    if isinstance(transcript_readable, str) and transcript_readable.strip():
        return transcript_readable.strip()
    if isinstance(transcript, str) and transcript.strip():
        return transcript.strip()

    return "Transcript unavailable for this demo example."


def trim_transcript(text: str, max_chars: int) -> str:
    """Trim transcript for public/demo output.

    For a public GitHub repo, avoid committing unnecessary large transcript files.
    A shorter transcript excerpt is enough to demonstrate retrieval and evidence
    grounding.
    """

    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars].rstrip()
        + "\n\n[Transcript excerpt truncated for public/demo repository.]"
    )


def compare_to_baseline(score: float, baseline: float, tolerance: float = 0.05) -> str:
    """Convert a score and baseline into higher/lower/near_baseline."""

    if score > baseline + tolerance:
        return "higher"
    if score < baseline - tolerance:
        return "lower"
    return "near_baseline"


def safe_filename(value: str) -> str:
    """Create a filesystem-safe filename from a video ID."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float with a default fallback."""

    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any) -> int | None:
    """Convert a value to int if possible."""

    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe_bool(value: Any) -> bool | None:
    """Convert common truthy/falsy values to bool."""

    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def as_str(value: Any) -> str:
    """Convert a possibly missing dataframe value to a clean string."""

    if value is None or pd.isna(value):
        return ""
    return str(value).strip()

def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a dictionary as pretty JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_manifest(output_dir: Path, records: list[dict[str, Any]]) -> None:
    """Write a JSONL manifest for the exported demo subset."""

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_public_readme(output_dir: Path, args: argparse.Namespace, count: int) -> None:
    """Write documentation for the generated public/demo subset."""
    
    readme = f"""# Public Demo Data

This folder contains a small generated demo subset for the Health Video Communication Analyst Agent.

## Source files

The files were generated locally from private thesis artifacts:

- `{args.features}`
- `{args.videos}`

The full thesis datasets are not included in this repository.

## Selection rule

- Target label: `{args.label}`
- Requested sample size: `{args.n}`
- Exported sample size: `{count}`
- Random seed: `{args.seed}`
- Maximum transcript characters per video: `{args.max_transcript_chars}`

## Contents

```text
video_inputs/      Per-video metadata JSON files
metric_bundles/    Per-video 15-metric JSON files
transcripts/       Transcript excerpts for retrieval/evidence grounding
manifest.jsonl     Index of generated examples
```

## Notes
This demo subset is intended to show the analyst workflow. It should not be treated as
a replacement for the full thesis dataset or as a clinical/medical-quality benchmark.
"""

    (output_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()

