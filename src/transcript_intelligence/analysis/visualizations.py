"""
Generates all 10 visualizations for the Transcript Intelligence pipeline.

Charts produced:
  1.  cluster_map          — UMAP 2D scatter, colored by cluster
  2.  sentiment_by_call_type — Box + strip plot
  3.  sentiment_trend       — Weekly line chart per call type
  4.  churn_risk_heatmap    — Risk tier × cluster heatmap
  5.  action_items_treemap  — Sunburst of ownership hierarchy
  6.  sentiment_by_cluster  — Heatmap of pos/neutral/neg % per cluster
  7.  escalation_flags      — Stacked bar by call type × severity
  8.  speaker_talk_time     — Violin plot of host talk % by call type
  9.  key_moment_types      — Stacked bar of key moment types by call type
  10. cluster_composition   — Sunburst of cluster → call type
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from transcript_intelligence.models import (
    ChurnSignal,
    ClusterResult,
    EscalationFlag,
    PipelineState,
    SpeakerEngagement,
    TranscriptRecord,
)

pio.kaleido.scope.mathjax = None  # suppress MathJax warning

CALL_TYPE_COLORS = {"internal": "#4C9BE8", "support": "#E86B4C", "external": "#6BCB77"}
CLUSTER_PALETTE = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel


def generate_all_charts(state: PipelineState, charts_dir: Path) -> list[str]:
    transcripts = state.get("classified_transcripts", [])
    cluster_results = state.get("cluster_results", [])
    churn_signals = state.get("churn_signals", [])
    escalation_flags = state.get("escalation_flags", [])
    action_items = state.get("action_item_ownership", [])
    speaker_engagements = state.get("speaker_engagements", [])
    sentiment_aggregates = state.get("sentiment_aggregates", [])
    global_trend = state.get("global_sentiment_trend", {})

    cluster_label_map = {cr.cluster_id: cr.label for cr in cluster_results}
    paths: list[str] = []

    generators = [
        ("cluster_map", lambda: _chart_cluster_map(transcripts, cluster_label_map, charts_dir)),
        ("sentiment_by_call_type", lambda: _chart_sentiment_box(transcripts, charts_dir)),
        ("sentiment_trend", lambda: _chart_sentiment_trend(sentiment_aggregates, global_trend, charts_dir)),
        ("churn_risk_heatmap", lambda: _chart_churn_heatmap(churn_signals, cluster_label_map, charts_dir)),
        ("action_items_treemap", lambda: _chart_action_treemap(action_items, charts_dir)),
        ("sentiment_by_cluster", lambda: _chart_sentiment_cluster_heatmap(transcripts, cluster_label_map, charts_dir)),
        ("escalation_flags", lambda: _chart_escalation_bars(escalation_flags, charts_dir)),
        ("speaker_talk_time", lambda: _chart_speaker_engagement(speaker_engagements, transcripts, charts_dir)),
        ("key_moment_types", lambda: _chart_key_moment_types(transcripts, charts_dir)),
        ("cluster_composition", lambda: _chart_cluster_composition(transcripts, cluster_label_map, charts_dir)),
    ]

    for name, fn in generators:
        try:
            result = fn()
            if isinstance(result, list):
                paths.extend(result)
            elif result:
                paths.append(result)
        except Exception as e:
            print(f"  Warning: chart '{name}' failed: {e}")

    return paths


# ---------------------------------------------------------------------------
# Chart 1: UMAP 2D Cluster Map
# ---------------------------------------------------------------------------

def _chart_cluster_map(
    transcripts: list[TranscriptRecord],
    cluster_label_map: dict[int, str],
    out: Path,
) -> list[str]:
    rows = []
    for r in transcripts:
        if r.umap_x is None:
            continue
        label = cluster_label_map.get(r.cluster_id or -1, "Uncategorized")
        rows.append({
            "x": r.umap_x,
            "y": r.umap_y,
            "cluster": label,
            "call_type": r.call_type or "unknown",
            "title": r.title[:50],
            "sentiment": r.sentiment_score,
            "meeting_id": r.meeting_id,
        })

    if not rows:
        return []

    df = pd.DataFrame(rows)
    unique_clusters = df["cluster"].unique().tolist()
    color_map = {c: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i, c in enumerate(unique_clusters)}

    symbol_map = {"internal": "circle", "support": "square", "external": "diamond", "unknown": "x"}

    fig = go.Figure()
    for ct in df["call_type"].unique():
        sub = df[df["call_type"] == ct]
        for cluster in sub["cluster"].unique():
            csub = sub[sub["cluster"] == cluster]
            fig.add_trace(go.Scatter(
                x=csub["x"],
                y=csub["y"],
                mode="markers",
                name=f"{cluster} ({ct})",
                marker=dict(
                    color=color_map.get(cluster, "#999"),
                    symbol=symbol_map.get(ct, "circle"),
                    size=csub["sentiment"] * 3 + 4,
                    opacity=0.85,
                    line=dict(width=0.5, color="white"),
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Cluster: " + cluster + "<br>"
                    "Call type: " + ct + "<br>"
                    "Sentiment: %{customdata[1]:.1f}/5<br>"
                    "<extra></extra>"
                ),
                customdata=csub[["title", "sentiment"]].values,
            ))

    fig.update_layout(
        title="Topic Landscape: All 100 Transcripts (UMAP 2D)",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fa",
        legend=dict(title="Cluster (Call Type)", font=dict(size=9)),
        width=900,
        height=650,
    )

    html_path = str(out / "cluster_map.html")
    png_path = str(out / "cluster_map.png")
    fig.write_html(html_path)
    fig.write_image(png_path, scale=2)
    return [html_path, png_path]


# ---------------------------------------------------------------------------
# Chart 2: Sentiment by Call Type (Box + Strip)
# ---------------------------------------------------------------------------

def _chart_sentiment_box(transcripts: list[TranscriptRecord], out: Path) -> str:
    rows = [{"call_type": r.call_type or "unknown", "score": r.sentiment_score} for r in transcripts]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    call_types = ["internal", "support", "external"]
    data_by_type = [df[df["call_type"] == ct]["score"].tolist() for ct in call_types]

    bp = ax.boxplot(data_by_type, patch_artist=True, widths=0.4,
                    medianprops=dict(color="white", linewidth=2))
    colors = [CALL_TYPE_COLORS[ct] for ct in call_types]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Strip (jitter)
    for i, (ct, data) in enumerate(zip(call_types, data_by_type)):
        jitter = np.random.uniform(-0.12, 0.12, len(data))
        ax.scatter([i + 1 + j for j in jitter], data,
                   alpha=0.5, s=20, color=CALL_TYPE_COLORS[ct], zorder=3)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Internal", "Support", "External"], fontsize=12)
    ax.set_ylabel("Sentiment Score (1–5)", fontsize=11)
    ax.set_ylim(1, 5.2)
    ax.set_title("Sentiment Distribution by Call Type", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = str(out / "sentiment_by_call_type.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Chart 3: Sentiment Trend Over Time
# ---------------------------------------------------------------------------

def _chart_sentiment_trend(sentiment_aggregates, global_trend: dict, out: Path) -> list[str]:
    # Per call_type weekly trends
    type_aggs = {
        a.call_type: a
        for a in sentiment_aggregates
        if a.cluster_label is None and a.call_type in ("internal", "support", "external")
    }

    fig = go.Figure()
    for ct, agg in type_aggs.items():
        weeks = sorted(agg.weekly_trend.keys())
        scores = [agg.weekly_trend[w] for w in weeks]
        if not weeks:
            continue
        fig.add_trace(go.Scatter(
            x=weeks, y=scores,
            mode="lines+markers",
            name=ct.capitalize(),
            line=dict(color=CALL_TYPE_COLORS.get(ct, "#999"), width=2.5),
            marker=dict(size=7),
        ))

    # Global trend as dashed reference
    if global_trend:
        weeks = sorted(global_trend.keys())
        scores = [global_trend[w] for w in weeks]
        fig.add_trace(go.Scatter(
            x=weeks, y=scores,
            mode="lines",
            name="Overall",
            line=dict(color="#888", width=1.5, dash="dash"),
        ))

    fig.update_layout(
        title="Sentiment Score Trend Over Time (by Call Type)",
        xaxis_title="ISO Week",
        yaxis_title="Avg Sentiment Score (1–5)",
        yaxis=dict(range=[1, 5]),
        legend=dict(title="Call Type"),
        width=900,
        height=450,
        paper_bgcolor="white",
    )

    html_path = str(out / "sentiment_trend.html")
    png_path = str(out / "sentiment_trend.png")
    fig.write_html(html_path)
    fig.write_image(png_path, scale=2)
    return [html_path, png_path]


# ---------------------------------------------------------------------------
# Chart 4: Churn Risk Heatmap
# ---------------------------------------------------------------------------

def _chart_churn_heatmap(
    churn_signals: list[ChurnSignal],
    cluster_label_map: dict[int, str],
    out: Path,
) -> list[str]:
    if not churn_signals:
        return []

    tiers = ["high", "medium", "low"]
    # Get unique cluster labels from churn signals (we'll use a flat view)
    risk_counts = Counter(c.risk_tier for c in churn_signals)
    data = {
        "Risk Tier": tiers,
        "Count": [risk_counts.get(t, 0) for t in tiers],
        "Color": ["#E74C3C", "#F39C12", "#27AE60"],
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(df["Risk Tier"], df["Count"], color=df["Color"], alpha=0.85, height=0.5)

    for bar, count in zip(bars, df["Count"]):
        if count > 0:
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    str(count), va="center", fontsize=12, fontweight="bold")

    ax.set_xlabel("Number of Customer Calls", fontsize=11)
    ax.set_title("Churn Risk Distribution\n(External & Support Calls)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(df["Count"]) + 3)

    # Annotate high-risk call titles
    high_risk = [c for c in churn_signals if c.risk_tier == "high"][:5]
    if high_risk:
        note = "High-risk calls:\n" + "\n".join(f"• {c.title[:45]}" for c in high_risk)
        ax.text(0.98, 0.05, note, transform=ax.transAxes,
                fontsize=7.5, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#FDECEA", alpha=0.8))

    path = str(out / "churn_risk_heatmap.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return [path]


# ---------------------------------------------------------------------------
# Chart 5: Action Item Ownership Sunburst
# ---------------------------------------------------------------------------

def _chart_action_treemap(action_items, out: Path) -> list[str]:
    if not action_items:
        return []

    rows = [{"call_type": a.call_type or "unknown", "speaker": a.speaker_name, "count": 1} for a in action_items]
    df = pd.DataFrame(rows)
    agg = df.groupby(["call_type", "speaker"]).sum().reset_index()

    # Top speakers per call type
    top = agg.sort_values("count", ascending=False).head(30)

    fig = px.sunburst(
        top,
        path=["call_type", "speaker"],
        values="count",
        color="call_type",
        color_discrete_map=CALL_TYPE_COLORS,
        title="Action Item Ownership by Call Type & Speaker",
    )
    fig.update_traces(textinfo="label+percent entry")
    fig.update_layout(width=700, height=600)

    html_path = str(out / "action_items_treemap.html")
    png_path = str(out / "action_items_treemap.png")
    fig.write_html(html_path)
    fig.write_image(png_path, scale=2)
    return [html_path, png_path]


# ---------------------------------------------------------------------------
# Chart 6: Sentiment by Cluster Heatmap
# ---------------------------------------------------------------------------

def _chart_sentiment_cluster_heatmap(
    transcripts: list[TranscriptRecord],
    cluster_label_map: dict[int, str],
    out: Path,
) -> str:
    rows = []
    for r in transcripts:
        cid = r.cluster_id if r.cluster_id is not None else -1
        label = cluster_label_map.get(cid, "Uncategorized")
        # Bin score
        s = r.sentiment_score
        bucket = "Positive (4–5)" if s >= 4 else "Mixed (3–4)" if s >= 3 else "Negative (<3)"
        rows.append({"cluster": label, "bucket": bucket})

    df = pd.DataFrame(rows)
    pivot = pd.crosstab(df["cluster"], df["bucket"])
    # Normalize to %
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    # Order columns
    cols = [c for c in ["Negative (<3)", "Mixed (3–4)", "Positive (4–5)"] if c in pivot_pct.columns]
    pivot_pct = pivot_pct[cols]

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot_pct) * 0.55)))
    im = ax.imshow(pivot_pct.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="% of calls in cluster")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=10)
    ax.set_yticks(range(len(pivot_pct)))
    ax.set_yticklabels(pivot_pct.index.tolist(), fontsize=9)
    ax.set_title("Sentiment Distribution by Topic Cluster", fontsize=13, fontweight="bold")

    # Annotate cells
    for i in range(len(pivot_pct)):
        for j in range(len(cols)):
            val = pivot_pct.values[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, color="black" if 20 < val < 80 else "white")

    path = str(out / "sentiment_by_cluster.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Chart 7: Escalation Flags (Stacked Bar)
# ---------------------------------------------------------------------------

def _chart_escalation_bars(escalation_flags: list[EscalationFlag], out: Path) -> str:
    call_types = ["internal", "support", "external"]
    severities = ["critical", "moderate", "none"]
    colors = {"critical": "#E74C3C", "moderate": "#F39C12", "none": "#95A5A6"}

    data = {ct: Counter(f.severity for f in escalation_flags if f.call_type == ct) for ct in call_types}

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(call_types))
    width = 0.5
    bottom = np.zeros(len(call_types))

    for sev in severities:
        vals = np.array([data[ct].get(sev, 0) for ct in call_types])
        ax.bar(x, vals, width, bottom=bottom, label=sev.capitalize(),
               color=colors[sev], alpha=0.85)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(["Internal", "Support", "External"], fontsize=12)
    ax.set_ylabel("Number of Calls", fontsize=11)
    ax.set_title("Escalation Risk by Call Type", fontsize=13, fontweight="bold")
    ax.legend(title="Severity", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = str(out / "escalation_flags.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Chart 8: Speaker Talk Time (Violin)
# ---------------------------------------------------------------------------

def _chart_speaker_engagement(
    speaker_engagements: list[SpeakerEngagement],
    transcripts: list[TranscriptRecord],
    out: Path,
) -> str:
    # Internal (Aegis Cloud) host talk % by call type
    rows = []
    for e in speaker_engagements:
        if e.is_internal:
            rows.append({"call_type": _get_call_type(e.meeting_id, transcripts), "talk_pct": e.talk_time_pct})

    df = pd.DataFrame(rows)
    call_types = ["internal", "support", "external"]

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = range(1, len(call_types) + 1)

    for i, ct in enumerate(call_types):
        data = df[df["call_type"] == ct]["talk_pct"].tolist()
        if not data:
            continue
        vp = ax.violinplot([data], positions=[i + 1], showmedians=True, widths=0.6)
        for pc in vp["bodies"]:
            pc.set_facecolor(CALL_TYPE_COLORS[ct])
            pc.set_alpha(0.7)
        vp["cmedians"].set_color("white")
        vp["cmedians"].set_linewidth(2)

    ax.axhline(70, color="#E74C3C", linestyle="--", alpha=0.7, linewidth=1.5, label="70% monologue threshold")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Internal", "Support", "External"], fontsize=12)
    ax.set_ylabel("Internal Speaker Talk Time %", fontsize=11)
    ax.set_title("Aegis Cloud Speaker Talk Time Distribution\n(Internal speakers in each call type)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = str(out / "speaker_talk_time.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Chart 9: Key Moment Types (Stacked Bar)
# ---------------------------------------------------------------------------

def _chart_key_moment_types(transcripts: list[TranscriptRecord], out: Path) -> str:
    km_types = ["concern", "feature_gap", "positive_pivot", "action_item", "technical_issue", "churn_signal"]
    call_types = ["internal", "support", "external"]
    km_colors = {
        "concern": "#E74C3C",
        "feature_gap": "#F39C12",
        "positive_pivot": "#27AE60",
        "action_item": "#3498DB",
        "technical_issue": "#8E44AD",
        "churn_signal": "#C0392B",
    }

    data = {ct: Counter() for ct in call_types}
    for r in transcripts:
        ct = r.call_type or "unknown"
        if ct in call_types:
            for km in r.key_moments:
                data[ct][km.type] += 1

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(call_types))
    bottom = np.zeros(len(call_types))

    for km_type in km_types:
        vals = np.array([data[ct].get(km_type, 0) for ct in call_types])
        ax.bar(x, vals, 0.55, bottom=bottom, label=km_type.replace("_", " ").title(),
               color=km_colors.get(km_type, "#999"), alpha=0.85)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(["Internal", "Support", "External"], fontsize=12)
    ax.set_ylabel("Key Moment Count", fontsize=11)
    ax.set_title("Key Moment Types by Call Type", fontsize=13, fontweight="bold")
    ax.legend(title="Moment type", fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = str(out / "key_moment_types.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Chart 10: Cluster Composition (Sunburst)
# ---------------------------------------------------------------------------

def _chart_cluster_composition(
    transcripts: list[TranscriptRecord],
    cluster_label_map: dict[int, str],
    out: Path,
) -> list[str]:
    rows = []
    for r in transcripts:
        cid = r.cluster_id if r.cluster_id is not None else -1
        label = cluster_label_map.get(cid, "Uncategorized")
        rows.append({"cluster": label, "call_type": r.call_type or "unknown"})

    df = pd.DataFrame(rows)
    agg = df.groupby(["cluster", "call_type"]).size().reset_index(name="count")

    fig = px.sunburst(
        agg,
        path=["cluster", "call_type"],
        values="count",
        color="call_type",
        color_discrete_map={**CALL_TYPE_COLORS, "unknown": "#ccc"},
        title="Topic Cluster Composition by Call Type",
    )
    fig.update_traces(textinfo="label+percent parent")
    fig.update_layout(width=700, height=600)

    html_path = str(out / "cluster_composition.html")
    png_path = str(out / "cluster_composition.png")
    fig.write_html(html_path)
    fig.write_image(png_path, scale=2)
    return [html_path, png_path]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_call_type(meeting_id: str, transcripts: list[TranscriptRecord]) -> str:
    for r in transcripts:
        if r.meeting_id == meeting_id:
            return r.call_type or "unknown"
    return "unknown"
