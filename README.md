# Health Video Communication Analyst Agent

A thesis-extension project that converts health-video transcript, audio, and model-derived evidence into a structured communication-quality report.

This repository is designed as a software-engineered extension of a larger thesis multimodal data pipeline. The original thesis pipeline performs the heavy data collection, preprocessing, embedding extraction, model training, and interpretability workflows. This repo focuses on the downstream communication analyst layer:

1. Load exported thesis artifacts.
2. Retrieve metric definitions and transcript evidence.
3. Generate a structured communication-quality report.
4. Validate the report with guardrail checks.
5. Write a scientist-facing report, email draft, and evaluation trace.

The goal is not to rerun the thesis. The goal is to demonstrate a reusable, auditable, human-in-the-loop workflow pattern that can transfer to scientific AI workflows where domain context, structured outputs, quality control, and traceability matter.

---

## Why this project exists

The thesis project was originally developed using Google Colab because multimodal embedding and modeling workflows required GPU-backed notebooks. For deployment-style work, however, the downstream interpretation layer should be separated into a clean Python package.

This repo demonstrates that separation. It consumes exported artifacts such as:

- cleaned transcript text
- interpretable feature scores
- audio summary metrics
- model prediction outputs
- metric definitions and reporting guidelines

It then produces:

- `outputs/communication_quality_report.md`
- `outputs/scientist_email_draft.txt`
- `eval_logs/eval_trace.jsonl`

---

## Current status

The current version is intentionally deterministic and local-first. It does **not** require an LLM API key to run.

```bash
python run_pipeline.py --input data/sample_video_input.json --metrics data/sample_video_metrics.json
```

This current skeleton provides the repo architecture, schemas, retrieval layer, report generation, guardrail checks, and local trace logging. Later milestones can replace the deterministic report generator with an LLM-backed Analyst Agent while preserving the same schemas and validation layer.

---

## Optional future integrations

This repo is structured to support small additions later:

- LangGraph orchestration for conditional routing
- optional LangSmith tracing
- LLM structured outputs
- richer RAG over thesis report sections and transcript evidence
- human review forms

---

## Repository layout

```text
health-video-analyst-agent/
│
├── run_pipeline.py
├── requirements.txt
├── .env.example
│
├── data/
│   ├── sample_video_input.json
│   ├── sample_video_metrics.json
│   └── sample_transcript.txt
│
├── knowledge_base/
│   ├── thesis_metric_rubric.md
│   ├── communication_quality_guidelines.md
│   └── report_template.md
│
├── src/
│   ├── schemas.py
│   ├── state.py
│   ├── metric_loader.py
│   ├── retrieval.py
│   ├── analyst_agent.py
│   ├── qa_guardrails.py
│   ├── report_writer.py
│   ├── tracing.py
│   └── graph.py
│
├── outputs/
├── eval_logs/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CURSOR_SETUP.md
│   └── FILES_TO_UPLOAD.md
│
└── tests/
```

---

## Installation

On macOS, the system does not provide a `python` command by default. Install Python via [Homebrew](https://brew.sh) (`brew install python`) and use `python3` for the steps below. After activating the virtual environment, `python` works inside the venv.

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## Run the sample pipeline

```bash
python run_pipeline.py --input data/sample_video_input.json --metrics data/sample_video_metrics.json
```

Expected outputs:

```text
outputs/communication_quality_report.md
outputs/scientist_email_draft.txt
eval_logs/eval_trace.jsonl
```

---

## Project design principle

The LLM should not be the source of truth for metric values.

Metric scores should come from deterministic thesis artifacts or precomputed sample files. The analyst workflow may later use an LLM to interpret scores, retrieve evidence, format a report, and respond to QA feedback, but the core values should remain grounded in exported data.
