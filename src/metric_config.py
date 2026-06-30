"""Metric configuration for thesis-derived communication-quality reports.

This module defines the 15 normalized metrics used by the MVP analyst workflow.
The values are loaded from `interpretable_feature_dataset_v3.csv`.

Design principle:
    These values are computed from the video's transcript, audio, and model outputs. The LLM or analyst workflow may interpret
    them, but it should not invent or recalculate them during report generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MetricCategory = Literal["text", "audio", "model"]
InterpretationDirection = Literal["positive", "negative", "neutral", "mixed"]


@dataclass(frozen=True)
class MetricDefinition:
    """Definition for one reportable communication-quality metric."""

    metric_name: str
    source_column: str
    category: MetricCategory
    interpretation_direction: InterpretationDirection
    display_name: str
    description: str


METRIC_DEFINITIONS: list[MetricDefinition] = [
    MetricDefinition(
        metric_name="first_person_rate",
        source_column="pov_1",
        category="text",
        interpretation_direction="neutral",
        display_name="First-person language",
        description="Normalized rate of first-person language such as I, we, or our.",
    ),
    MetricDefinition(
        metric_name="second_person_address_rate",
        source_column="pov_2",
        category="text",
        interpretation_direction="positive",
        display_name="Second-person viewer address",
        description="Normalized rate of direct viewer-oriented language such as you or your.",
    ),
    MetricDefinition(
        metric_name="third_person_informational_rate",
        source_column="pov_3",
        category="text",
        interpretation_direction="neutral",
        display_name="Third-person informational framing",
        description="Normalized rate of third-person informational language such as patients, people, or clinicians.",
    ),
    MetricDefinition(
        metric_name="structured_explanation_score",
        source_column="counting_score",
        category="text",
        interpretation_direction="positive",
        display_name="Structured explanation",
        description="Normalized score for list-based, stepwise, or signposted explanation structure.",
    ),
    MetricDefinition(
        metric_name="engagement_request_score",
        source_column="req_eng",
        category="text",
        interpretation_direction="positive",
        display_name="Engagement request",
        description="Normalized signal for direct engagement prompts such as like, subscribe, comment, save, or share.",
    ),
    MetricDefinition(
        metric_name="formality_score",
        source_column="style_formality",
        category="text",
        interpretation_direction="mixed",
        display_name="Formality",
        description="Normalized style signal for formal versus casual language patterns.",
    ),
    MetricDefinition(
        metric_name="medical_vocabulary_rate",
        source_column="med_vocab_rate",
        category="text",
        interpretation_direction="positive",
        display_name="Medical vocabulary rate",
        description="Normalized rate of topic-relevant medical vocabulary.",
    ),
    MetricDefinition(
        metric_name="youth_style_rate",
        source_column="youth_style_rate",
        category="text",
        interpretation_direction="mixed",
        display_name="Youth-oriented style",
        description="Normalized signal for casual or youth-oriented phrasing.",
    ),
    MetricDefinition(
        metric_name="emotional_intensity",
        source_column="emotion_intensity",
        category="text",
        interpretation_direction="mixed",
        display_name="Emotional intensity",
        description="Normalized emotional intensity signal in transcript language.",
    ),
    MetricDefinition(
        metric_name="readability_grade_level",
        source_column="txt_fk_grade",
        category="text",
        interpretation_direction="mixed",
        display_name="Readability grade level",
        description="Normalized Flesch-Kincaid-style readability complexity score.",
    ),
    MetricDefinition(
        metric_name="lexical_diversity_mtld",
        source_column="txt_mtld",
        category="text",
        interpretation_direction="positive",
        display_name="Lexical diversity",
        description="Normalized MTLD lexical diversity score.",
    ),
    MetricDefinition(
        metric_name="sentiment_polarity",
        source_column="txt_vader_compound",
        category="text",
        interpretation_direction="neutral",
        display_name="Sentiment polarity",
        description="Normalized VADER compound sentiment signal.",
    ),
    MetricDefinition(
        metric_name="pitch_variability",
        source_column="aud_pitch_var",
        category="audio",
        interpretation_direction="mixed",
        display_name="Pitch variability",
        description="Normalized vocal pitch variability signal.",
    ),
    MetricDefinition(
        metric_name="loudness_variability",
        source_column="aud_loudness_var",
        category="audio",
        interpretation_direction="neutral",
        display_name="Loudness variability",
        description="Normalized vocal loudness variability signal.",
    ),
    MetricDefinition(
        metric_name="voice_clarity_hnr",
        source_column="aud_hnr_mean",
        category="audio",
        interpretation_direction="positive",
        display_name="Voice clarity / HNR",
        description="Normalized harmonics-to-noise ratio voice clarity signal.",
    ),
]


EXPECTED_METRICS: list[str] = [metric.metric_name for metric in METRIC_DEFINITIONS]

SOURCE_COLUMNS: list[str] = [metric.source_column for metric in METRIC_DEFINITIONS]

METRIC_BY_NAME: dict[str, MetricDefinition] = {
    metric.metric_name: metric for metric in METRIC_DEFINITIONS
}

METRIC_BY_SOURCE_COLUMN: dict[str, MetricDefinition] = {
    metric.source_column: metric for metric in METRIC_DEFINITIONS
}