"""
LangGraph pipeline graph definition.

Pipeline:
  ingest → classify_call_type → [llm_classify_fallback?] → embed_transcripts
         → cluster_topics → label_clusters → analyze_sentiment
         → extract_insights → compile_report → END
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from langgraph.graph import END, StateGraph
from rich.console import Console

from transcript_intelligence.models import PipelineState
from transcript_intelligence.pipeline.nodes import (
    classify,
    cluster,
    embed,
    ingest,
    insights,
    sentiment,
)

console = Console()


def _needs_llm_classification(state: PipelineState) -> str:
    needed = state.get("llm_classification_needed", [])
    return "llm_classify_fallback" if needed else "embed_transcripts"


def _compile_report(state: PipelineState) -> PipelineState:
    """Serialize results and generate artifacts."""
    from transcript_intelligence.analysis import report, visualizations

    outputs_dir = Path(os.getenv("OUTPUTS_DIR", "outputs"))
    charts_dir = outputs_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    console.print("[cyan]Generating visualizations...[/cyan]")
    chart_paths = visualizations.generate_all_charts(state, charts_dir)

    console.print("[cyan]Building slide deck...[/cyan]")
    pptx_path = report.build_pptx(state, chart_paths, outputs_dir)

    console.print("[cyan]Saving results.json...[/cyan]")
    results_path = outputs_dir / "results.json"
    _save_results(state, results_path)

    console.print(f"[bold green]✓ Pipeline complete![/bold green]")
    console.print(f"  Slide deck: {pptx_path}")
    console.print(f"  Results:    {results_path}")
    console.print(f"  Charts:     {charts_dir} ({len(chart_paths)} files)")

    return {**state, "report_path": str(pptx_path), "chart_paths": chart_paths, "results_json_path": str(results_path)}


def build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    g.add_node("ingest", ingest.run)
    g.add_node("classify_call_type", classify.run)
    g.add_node("llm_classify_fallback", classify.run_llm_fallback)
    g.add_node("embed_transcripts", embed.run)
    g.add_node("cluster_topics", cluster.run_clustering)
    g.add_node("label_clusters", cluster.run_labeling)
    g.add_node("analyze_sentiment", sentiment.run)
    g.add_node("extract_insights", insights.run)
    g.add_node("compile_report", _compile_report)

    g.set_entry_point("ingest")
    g.add_edge("ingest", "classify_call_type")
    g.add_conditional_edges("classify_call_type", _needs_llm_classification)
    g.add_edge("llm_classify_fallback", "embed_transcripts")
    g.add_edge("embed_transcripts", "cluster_topics")
    g.add_edge("cluster_topics", "label_clusters")
    g.add_edge("label_clusters", "analyze_sentiment")
    g.add_edge("analyze_sentiment", "extract_insights")
    g.add_edge("extract_insights", "compile_report")
    g.add_edge("compile_report", END)

    return g.compile()


def _save_results(state: PipelineState, path: Path) -> None:
    transcripts = state.get("classified_transcripts", [])
    cluster_results = state.get("cluster_results", [])
    churn_signals = state.get("churn_signals", [])
    escalation_flags = state.get("escalation_flags", [])
    action_items = state.get("action_item_ownership", [])
    speaker_engagements = state.get("speaker_engagements", [])
    sentiment_aggregates = state.get("sentiment_aggregates", [])

    results = {
        "summary": {
            "total_transcripts": len(transcripts),
            "classification_stats": state.get("classification_stats", {}),
            "num_clusters": state.get("num_clusters", 0),
            "noise_count": state.get("noise_count", 0),
            "total_action_items": len(action_items),
            "escalations_detected": sum(1 for e in escalation_flags if e.escalation_detected),
            "high_churn_risk_calls": sum(1 for c in churn_signals if c.risk_tier == "high"),
        },
        "transcripts": [
            {
                "meeting_id": r.meeting_id,
                "title": r.title,
                "call_type": r.call_type,
                "call_type_confidence": r.call_type_confidence,
                "cluster_id": r.cluster_id,
                "sentiment_score": r.sentiment_score,
                "overall_sentiment": r.overall_sentiment,
                "umap_x": r.umap_x,
                "umap_y": r.umap_y,
                "duration_minutes": r.duration_minutes,
                "start_time": r.start_time,
            }
            for r in transcripts
        ],
        "clusters": [cr.model_dump() for cr in cluster_results],
        "churn_signals": [cs.model_dump() for cs in churn_signals],
        "escalation_flags": [ef.model_dump() for ef in escalation_flags if ef.escalation_detected],
        "action_item_ownership": [a.model_dump() for a in action_items],
        "sentiment_aggregates": [s.model_dump() for s in sentiment_aggregates if s.cluster_label is None],
        "global_sentiment_trend": state.get("global_sentiment_trend", {}),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
