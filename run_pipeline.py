"""Command-line entrypoint for the Health Video Communication Analyst Agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.graph import build_graph, run_sequential
from src.state import AnalystState


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate a communication-quality report from thesis-derived video artifacts."
    )
    parser.add_argument(
        "--input",
        default="data/sample_video_input.json",
        help="Path to video input JSON.",
    )
    parser.add_argument(
        "--metrics",
        default="data/sample_video_metrics.json",
        help="Path to precomputed metric JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where report outputs will be written.",
    )
    parser.add_argument(
        "--eval-log",
        default="eval_logs/eval_trace.jsonl",
        help="Path to local JSONL evaluation trace.",
    )
    parser.add_argument(
        "--no-langgraph",
        action="store_true",
        help="Run sequentially instead of using LangGraph.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the report generation pipeline."""

    load_dotenv()
    args = parse_args()

    state: AnalystState = {
        "input_path": args.input,
        "metrics_path": args.metrics,
        "output_dir": args.output_dir,
        "eval_log_path": args.eval_log,
        "retry_count": 0,
        "warnings": [],
        "errors": [],
    }

    if args.no_langgraph:
        final_state = run_sequential(state)
    else:
        try:
            app = build_graph()
            final_state = app.invoke(state)
        except ImportError:
            print("LangGraph not installed. Falling back to sequential execution.")
            final_state = run_sequential(state)

    status = final_state.get("final_status", "unknown")
    print(f"Pipeline completed with status: {status}")
    print(f"Report: {Path(args.output_dir) / 'communication_quality_report.md'}")
    print(f"Email draft: {Path(args.output_dir) / 'scientist_email_draft.txt'}")
    print(f"Trace log: {args.eval_log}")


if __name__ == "__main__":
    main()
