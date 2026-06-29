# Thesis Metric Rubric

This file defines the communication-quality metrics used by the analyst workflow. In the full thesis project, these metrics are derived from transcript text, audio summaries, or model outputs. In this Day 1 MVP, the metric values are loaded from `sample_video_metrics.json`.

## Engagement request score

Measures whether the video explicitly prompts audience engagement, such as asking viewers to like, subscribe, comment, share, save, or follow. In the thesis interpretation framework, engagement requests were treated as a potentially positive signal when evaluating relative engagement.

## Lexical diversity / MTLD

Measures variety in transcript language. Higher lexical diversity may indicate richer explanations and less repetitive content.

## Medical vocabulary rate

Measures use of health-relevant terminology. Higher values may indicate more topic-specific communication, though overly technical language can reduce accessibility.

## Structured explanation score

Measures whether the transcript uses clear structure, such as steps, lists, sequence markers, summaries, and signposting.

## Readability grade level

Measures approximate reading complexity. This should be interpreted cautiously because more complex medical topics may require more complex language.

## Sentiment polarity

Measures overall emotional tone. This is not inherently good or bad and should be interpreted in context.

## Emotional intensity

Measures emotionally loaded wording. Higher intensity may increase engagement but may also risk sensational framing.

## Second-person address rate

Measures direct address to the viewer, such as "you" or "your." This can increase perceived relevance and conversational tone.

## Third-person informational rate

Measures informational framing, such as references to "patients," "people," or "clinicians." This can support educational tone.

## Explanation density

Measures the concentration of explanatory statements, causal phrases, and definitions.

## Actionability score

Measures whether the video gives clear, practical next steps without overclaiming.

## Topic specificity score

Measures whether the transcript stays focused on the target health topic.

## Pitch variability

Audio feature measuring variation in vocal pitch. In the thesis interpretation framework, very high pitch variability was treated cautiously because it may reflect unstable delivery or overly dramatic presentation.

## Voice clarity / HNR

Audio feature approximating harmonic clarity. Higher values may suggest clearer voice quality.

## Loudness variability

Audio feature measuring delivery dynamics. This should be interpreted cautiously because recording conditions can affect the value.
