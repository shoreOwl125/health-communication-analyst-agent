"""Workflow orchestration for the analyst pipeline.

The pipeline is written as small nodes to make it easy to migrate from a simple
sequential script to a LangGraph workflow. If LangGraph is installed, `build_graph`
returns a compiled graph. If not, `run_sequential` still executes the same nodes.

Day 1 keeps the workflow deterministic. Later versions can replace
`generate_deterministic_report` with an LLM-backed Analyst Agent while preserving
retrieval, schemas, QA, and trace logging.
"""

from __future__ import annotations

from src.analyst_agent import generate_deterministic_report, revise_report_with_qa_feedback
from src.metric_loader import load_metric_bundle, load_transcript, load_video_input
from src.qa_guardrails import evidence_coverage_count, validate_report
from src.report_writer import write_report_outputs
from src.retrieval import retrieve_for_metrics
from src.schemas import PipelineTrace
from src.state import AnalystState
from src.tracing import append_trace, build_run_id


def load_inputs_node(state: AnalystState) -> AnalystState:
    """Load video metadata, metric bundle, and transcript."""

    video_input = load_video_input(state["input_path"])
    metric_bundle = load_metric_bundle(state["metrics_path"])
    transcript = load_transcript(video_input)

    state["video_input"] = video_input
    state["metric_bundle"] = metric_bundle
    state["transcript"] = transcript
    state.setdefault("retry_count", 0)
    state.setdefault("warnings", [])
    state.setdefault("errors", [])
    return state


def retrieve_context_node(state: AnalystState) -> AnalystState:
    """Retrieve rubric context and transcript evidence."""

    metric_names = [m.metric_name for m in state["metric_bundle"].metrics]
    rubric_context, transcript_evidence = retrieve_for_metrics(
        metric_names=metric_names,
        transcript=state["transcript"],
        knowledge_dir="knowledge_base",
    )
    state["rubric_context"] = rubric_context
    state["transcript_evidence"] = transcript_evidence
    return state


def generate_report_node(state: AnalystState) -> AnalystState:
    """Generate the structured report."""

    state["report"] = generate_deterministic_report(
        video_input=state["video_input"],
        metric_bundle=state["metric_bundle"],
        rubric_context=state["rubric_context"],
        transcript_evidence=state["transcript_evidence"],
    )
    return state


def validate_report_node(state: AnalystState) -> AnalystState:
    """Run deterministic QA guardrails."""

    qa_result = validate_report(state["report"])
    state["qa_result"] = qa_result
    state["errors"] = qa_result.errors
    state["warnings"] = qa_result.warnings
    return state


def revise_report_node(state: AnalystState) -> AnalystState:
    """Apply one revision attempt based on QA feedback."""

    state["retry_count"] = int(state.get("retry_count", 0)) + 1
    state["report"] = revise_report_with_qa_feedback(
        report=state["report"],
        qa_errors=state.get("errors", []),
    )
    return state


def should_revise_or_finish(state: AnalystState) -> str:
    """Conditional router for QA pass/fail handling."""

    if not state.get("errors"):
        return "write_outputs"
    if int(state.get("retry_count", 0)) < 2:
        return "revise_report"
    return "write_outputs"


def write_outputs_node(state: AnalystState) -> AnalystState:
    """Write report artifacts and local trace log."""

    output_dir = state.get("output_dir", "outputs")
    eval_log_path = state.get("eval_log_path", "eval_logs/eval_trace.jsonl")

    write_report_outputs(state["report"], output_dir=output_dir)

    qa_result = state["qa_result"]
    final_status = "passed" if qa_result.passed else "failed"
    state["final_status"] = final_status

    trace = PipelineTrace(
        run_id=build_run_id(state["video_input"].video_id),
        video_id=state["video_input"].video_id,
        input_path=state["input_path"],
        metrics_path=state["metrics_path"],
        schema_validation="passed",
        qa_status="passed" if qa_result.passed else "failed",
        metric_count=len(state["report"].metric_interpretations),
        evidence_coverage_count=evidence_coverage_count(state["report"]),
        revision_count=int(state.get("retry_count", 0)),
        final_status=final_status,
        warnings=state.get("warnings", []),
        errors=state.get("errors", []),
    )
    append_trace(trace, log_path=eval_log_path)
    return state


def run_sequential(state: AnalystState) -> AnalystState:
    """Run the pipeline without LangGraph.

    This fallback keeps the project runnable even if LangGraph is not installed.
    """

    state = load_inputs_node(state)
    state = retrieve_context_node(state)
    state = generate_report_node(state)
    state = validate_report_node(state)

    while should_revise_or_finish(state) == "revise_report":
        state = revise_report_node(state)
        state = validate_report_node(state)

    state = write_outputs_node(state)
    return state


def build_graph():
    """Build a minimal LangGraph workflow.

    Returns:
        A compiled LangGraph workflow.

    Raises:
        ImportError: If LangGraph is not installed.
    """

    from langgraph.graph import END, StateGraph

    graph = StateGraph(AnalystState)

    graph.add_node("load_inputs", load_inputs_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("validate_report", validate_report_node)
    graph.add_node("revise_report", revise_report_node)
    graph.add_node("write_outputs", write_outputs_node)

    graph.set_entry_point("load_inputs")
    graph.add_edge("load_inputs", "retrieve_context")
    graph.add_edge("retrieve_context", "generate_report")
    graph.add_edge("generate_report", "validate_report")

    graph.add_conditional_edges(
        "validate_report",
        should_revise_or_finish,
        {
            "revise_report": "revise_report",
            "write_outputs": "write_outputs",
        },
    )

    graph.add_edge("revise_report", "validate_report")
    graph.add_edge("write_outputs", END)

    return graph.compile()
