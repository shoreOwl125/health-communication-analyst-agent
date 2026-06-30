# Public Demo Data

This folder contains a small generated demo subset for the Health Video Communication Analyst Agent.

## Source files

The files were generated locally from private thesis artifacts:

- `data/private/interpretable_feature_dataset_v3.csv`
- `data/private/healthVideoData_preprocessed.csv`

The full thesis datasets are not included in this repository.

## Selection rule

- Target label: `0`
- Requested sample size: `100`
- Exported sample size: `100`
- Random seed: `42`
- Maximum transcript characters per video: `6000`

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
