"""
Node 6: Sentiment Analysis

Aggregates pre-extracted sentiment data:
- Per call_type: avg score, overallSentiment distribution
- Per cluster: avg score and distribution
- Sentence-level: pos/neutral/neg ratio per call_type
- Temporal trend: weekly avg sentiment per call_type
- Sentiment arc: first-half vs second-half negativity ratio per call
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from statistics import mean

from rich.console import Console

from transcript_intelligence.models import (
    PipelineState,
    SentimentAggregate,
    TranscriptRecord,
)

console = Console()

SENTIMENT_NUMERIC = {"positive": 1, "neutral": 0, "negative": -1}


def run(state: PipelineState) -> PipelineState:
    transcripts = state["classified_transcripts"]
    cluster_results = state.get("cluster_results", [])

    cluster_label_map = {cr.cluster_id: cr.label for cr in cluster_results}

    aggregates: list[SentimentAggregate] = []

    for call_type in ("internal", "support", "external"):
        subset = [r for r in transcripts if r.call_type == call_type]
        if not subset:
            continue
        agg = _compute_aggregate(subset, call_type=call_type, cluster_label=None)
        aggregates.append(agg)

        # Per cluster within this call_type
        cluster_groups: dict[int, list[TranscriptRecord]] = defaultdict(list)
        for r in subset:
            cid = r.cluster_id if r.cluster_id is not None else -1
            cluster_groups[cid].append(r)

        for cid, group in cluster_groups.items():
            label = cluster_label_map.get(cid, "Uncategorized")
            agg_c = _compute_aggregate(group, call_type=call_type, cluster_label=label)
            aggregates.append(agg_c)

    # Global weekly trend across all calls
    global_trend = _compute_weekly_trend(transcripts)

    console.print("[green]✓ Sentiment analysis complete[/green]")
    for call_type in ("internal", "support", "external"):
        top = next((a for a in aggregates if a.call_type == call_type and a.cluster_label is None), None)
        if top:
            console.print(f"  {call_type}: avg={top.avg_score:.2f}, dist={top.score_distribution}")

    return {**state, "sentiment_aggregates": aggregates, "global_sentiment_trend": global_trend}


def _compute_aggregate(
    records: list[TranscriptRecord],
    call_type: str | None,
    cluster_label: str | None,
) -> SentimentAggregate:
    avg_score = mean(r.sentiment_score for r in records)
    score_dist = dict(Counter(r.overall_sentiment for r in records))
    weekly = _compute_weekly_trend(records)

    # Sentence-level ratios
    total_sentences = sum(len(r.sentences) for r in records)
    pos = sum(sum(1 for s in r.sentences if s.sentimentType == "positive") for r in records)
    neg = sum(sum(1 for s in r.sentences if s.sentimentType == "negative") for r in records)

    pos_ratio = pos / total_sentences if total_sentences else 0.0
    neg_ratio = neg / total_sentences if total_sentences else 0.0

    return SentimentAggregate(
        call_type=call_type,
        cluster_label=cluster_label,
        avg_score=round(avg_score, 3),
        score_distribution=score_dist,
        weekly_trend=weekly,
        sentence_pos_ratio=round(pos_ratio, 3),
        sentence_neg_ratio=round(neg_ratio, 3),
    )


def _compute_weekly_trend(records: list[TranscriptRecord]) -> dict[str, float]:
    by_week: dict[str, list[float]] = defaultdict(list)
    for r in records:
        try:
            dt = datetime.fromisoformat(r.start_time.replace("Z", "+00:00"))
            # ISO week key: "YYYY-Www"
            week_key = f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}"
            by_week[week_key].append(r.sentiment_score)
        except (ValueError, AttributeError):
            pass
    return {week: round(mean(scores), 3) for week, scores in sorted(by_week.items())}


def compute_sentiment_arc(record: TranscriptRecord) -> str:
    """Compare first-half vs second-half sentence negativity ratio."""
    sentences = sorted(record.sentences, key=lambda s: s.time)
    if len(sentences) < 4:
        return "stable"
    mid = len(sentences) // 2
    first_neg = sum(1 for s in sentences[:mid] if s.sentimentType == "negative") / mid
    second_neg = sum(1 for s in sentences[mid:] if s.sentimentType == "negative") / (len(sentences) - mid)
    if second_neg > first_neg + 0.15:
        return "declining"
    if second_neg < first_neg - 0.15:
        return "improving"
    return "stable"
