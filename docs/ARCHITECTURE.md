# Architecture

This project separates a notebook-heavy thesis workflow from a downstream analyst workflow.

## Upstream thesis pipeline

The upstream thesis pipeline is responsible for:

- YouTube metadata collection
- transcript cleaning and normalization
- audio extraction
- text/audio embedding extraction
- model training
- residual or quantile labeling
- interpretable feature extraction
- prediction and interpretability outputs

Those steps may remain in Colab notebooks because they can require GPU compute and large intermediate artifacts.

## Downstream analyst workflow

This repo begins after the upstream artifacts already exist.

```text
Exported thesis artifacts
  ├── video metadata
  ├── cleaned transcript
  ├── interpretable feature scores
  ├── audio summaries
  └── model prediction output

        ↓

Health Video Communication Analyst
  ├── load artifacts
  ├── retrieve rubric/guidelines
  ├── retrieve transcript evidence
  ├── generate structured report
  ├── validate report
  └── write outputs + trace
```

## Why the LLM should not generate metric values

In a scientific workflow, model-derived or analysis-derived values should be treated as source data. An LLM can help interpret, summarize, and format those values, but it should not invent them.

This design mirrors higher-stakes scientific workflows where generated text must be grounded in retrieved evidence, structured schemas, and reproducible validation checks.

## Why use a graph workflow

The project is organized as nodes:

1. load inputs
2. retrieve context
3. generate report
4. validate report
5. revise if needed
6. write outputs

This graph structure creates a small but real agentic workflow because QA results can conditionally route the pipeline back to revision before final output.
