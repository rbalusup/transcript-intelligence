"""
Node 7: Extract Insights

Implements all 4 bonus insight analyses:
  A. Churn Risk Scoring (external calls)
  B. Action Item Ownership (all calls)
  C. Escalation Detection (all calls)
  D. Speaker Engagement Analysis (all calls)
"""
from __future__ import annotations

import json
import re
import time
from statistics import mean

import anthropic
from rich.console import Console
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transcript_intelligence.models import (
    ActionItemOwnership,
    ChurnSignal,
    EscalationFlag,
    PipelineState,
    SpeakerEngagement,
    TranscriptRecord,
)
from transcript_intelligence.pipeline.nodes.sentiment import compute_sentiment_arc
from transcript_intelligence.prompts import CHURN_ENRICHMENT_PROMPT, ESCALATION_PROMPT

console = Console()

SENTIMENT_NUMERIC = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def run(state: PipelineState) -> PipelineState:
    transcripts = state["classified_transcripts"]
    cluster_results = state.get("cluster_results", [])
    cluster_label_map = {cr.cluster_id: cr.label for cr in cluster_results}

    console.print("[cyan]Extracting insights...[/cyan]")

    churn_signals = _compute_churn_risk(transcripts, cluster_label_map)
    action_items = _compute_action_ownership(transcripts, cluster_label_map)
    escalation_flags = _compute_escalations(transcripts, cluster_label_map)
    speaker_engagements = _compute_speaker_engagement(transcripts)

    _log_summary(churn_signals, escalation_flags, action_items, speaker_engagements)
    return {
        **state,
        "churn_signals": churn_signals,
        "action_item_ownership": action_items,
        "escalation_flags": escalation_flags,
        "speaker_engagements": speaker_engagements,
    }


# ---------------------------------------------------------------------------
# A. Churn Risk Scoring
# ---------------------------------------------------------------------------

def _compute_churn_risk(
    transcripts: list[TranscriptRecord],
    cluster_label_map: dict[int, str],
) -> list[ChurnSignal]:
    client = anthropic.Anthropic(max_retries=5)
    results: list[ChurnSignal] = []

    external_calls = [r for r in transcripts if r.call_type in ("external", "support")]

    for record in external_calls:
        churn_moments = [km.text for km in record.key_moments if km.type == "churn_signal"]
        feature_gaps = [km.text for km in record.key_moments if km.type == "feature_gap"]
        concerns = [km for km in record.key_moments if km.type == "concern"]

        score = 0.0
        score += min(len(churn_moments) * 0.25, 0.50)
        score += min(len(feature_gaps) * 0.10, 0.25)
        # Sentiment penalty: full 0.25 at score=1.0, zero at score>=3.5
        # Scores are 1–5; normalize against the negative half of the scale
        score += max(0.0, (3.5 - record.sentiment_score) / 2.5) * 0.25

        arc = compute_sentiment_arc(record)
        if arc == "declining":
            score += 0.10

        score = min(score, 1.0)
        risk_tier = "high" if score >= 0.60 else "medium" if score >= 0.30 else "low"

        signal = ChurnSignal(
            meeting_id=record.meeting_id,
            title=record.title,
            churn_score=round(score, 3),
            churn_moments=churn_moments,
            feature_gaps=feature_gaps,
            sentiment_trend=arc,
            risk_tier=risk_tier,
        )

        # LLM enrichment for high-risk calls
        if risk_tier == "high" and churn_moments:
            try:
                km_text = "\n".join(f"- [{km.type}] {km.text}" for km in record.key_moments[:8])
                prompt = CHURN_ENRICHMENT_PROMPT.format(
                    title=record.title,
                    start_time=record.start_time[:10],
                    duration=record.duration_minutes,
                    sentiment_score=record.sentiment_score,
                    sentiment_trend=arc,
                    summary=record.summary_text[:500],
                    key_moments=km_text,
                    action_items="\n".join(f"- {a}" for a in record.action_items[:5]),
                )
                resp = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )
                parsed = json.loads(_extract_json(resp.content[0].text))
                signal = signal.model_copy(update={
                    "churn_risk_reasoning": parsed.get("churn_risk_reasoning"),
                    "top_concerns": parsed.get("top_concerns", []),
                    "recommended_actions": parsed.get("recommended_actions", []),
                    "urgency": parsed.get("urgency"),
                })
            except Exception as e:
                console.print(f"[yellow]  Churn enrichment skipped for {record.meeting_id}: {e}[/yellow]")
            else:
                time.sleep(1)

        results.append(signal)

    results.sort(key=lambda x: x.churn_score, reverse=True)
    high = sum(1 for r in results if r.risk_tier == "high")
    console.print(f"[green]✓ Churn risk: {high} high-risk calls out of {len(results)} external/support[/green]")
    return results


# ---------------------------------------------------------------------------
# B. Action Item Ownership
# ---------------------------------------------------------------------------

def _compute_action_ownership(
    transcripts: list[TranscriptRecord],
    cluster_label_map: dict[int, str],
) -> list[ActionItemOwnership]:
    results: list[ActionItemOwnership] = []

    for record in transcripts:
        cluster_label = cluster_label_map.get(record.cluster_id or -1)

        for action_item in record.action_items:
            owner = _infer_action_owner(record, action_item)
            results.append(ActionItemOwnership(
                meeting_id=record.meeting_id,
                title=record.title,
                speaker_name=owner,
                action_item=action_item,
                call_type=record.call_type,
                cluster_label=cluster_label,
            ))

    console.print(f"[green]✓ Action items: {len(results)} total across {len(transcripts)} calls[/green]")
    return results


def _infer_action_owner(record: TranscriptRecord, action_item: str) -> str:
    """
    Try to infer who owns an action item:
    1. Parse "Name:" prefix pattern (common format in this dataset)
    2. TF-IDF cosine similarity against speaker sentences
    3. Fallback to organizer
    """
    # Pattern: "Name: task description"
    match = re.match(r"^([A-Z][a-z]+(?: [A-Z][a-z]+)?)\s*:\s*", action_item)
    if match:
        candidate = match.group(1)
        known_speakers = {s.speaker_name for s in record.sentences}
        # Fuzzy match: check if first name appears in any speaker name
        first_name = candidate.split()[0]
        for speaker in known_speakers:
            if first_name.lower() in speaker.lower():
                return speaker
        if candidate:
            return candidate

    # TF-IDF similarity against speaker sentences
    if record.sentences:
        speakers: list[str] = []
        speaker_texts: list[str] = []
        speaker_groups: dict[str, list[str]] = {}

        for s in record.sentences:
            speaker_groups.setdefault(s.speaker_name, []).append(s.sentence)

        if len(speaker_groups) >= 2:
            for name, texts in speaker_groups.items():
                speakers.append(name)
                speaker_texts.append(" ".join(texts))

            try:
                vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
                all_texts = speaker_texts + [action_item]
                tfidf = vectorizer.fit_transform(all_texts)
                action_vec = tfidf[-1]
                speaker_vecs = tfidf[:-1]
                sims = cosine_similarity(action_vec, speaker_vecs)[0]
                best_idx = int(sims.argmax())
                return speakers[best_idx]
            except Exception:
                pass

    return record.organizer_email.split("@")[0].replace(".", " ").title()


# ---------------------------------------------------------------------------
# C. Escalation Detection
# ---------------------------------------------------------------------------

def _compute_escalations(
    transcripts: list[TranscriptRecord],
    cluster_label_map: dict[int, str],  # noqa: ARG001 — reserved for future cluster-aware routing
) -> list[EscalationFlag]:
    client = anthropic.Anthropic(max_retries=5)
    results: list[EscalationFlag] = []

    for record in transcripts:
        signals: list[str] = []
        km_types = [km.type for km in record.key_moments]

        if "churn_signal" in km_types:
            signals.append("churn signal detected in call")
        if record.sentiment_score < 2.5:
            signals.append(f"very low sentiment score ({record.sentiment_score:.1f}/5)")
        concern_count = km_types.count("concern")
        if concern_count >= 3:
            signals.append(f"{concern_count} concern moments flagged")
        tech_count = km_types.count("technical_issue")
        if tech_count >= 2:
            signals.append(f"{tech_count} technical issues raised")
        arc = compute_sentiment_arc(record)
        if arc == "declining":
            signals.append("sentiment declined throughout the call")

        escalation = len(signals) >= 2
        severity = "critical" if len(signals) >= 4 else "moderate" if escalation else "none"

        flag = EscalationFlag(
            meeting_id=record.meeting_id,
            title=record.title,
            call_type=record.call_type,
            escalation_detected=escalation,
            signals=signals,
            severity=severity,
        )

        # LLM recommendation for critical cases
        if severity == "critical":
            try:
                prompt = ESCALATION_PROMPT.format(
                    call_type=record.call_type or "unknown",
                    signals_list="\n".join(f"- {s}" for s in signals),
                    sentiment_score=record.sentiment_score,
                    duration=record.duration_minutes,
                    has_churn_signals="churn_signal" in km_types,
                )
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=150,
                    messages=[{"role": "user", "content": prompt}],
                )
                parsed = json.loads(_extract_json(resp.content[0].text))
                flag = flag.model_copy(update={
                    "recommended_owner": parsed.get("recommended_owner"),
                })
            except Exception as e:
                console.print(f"[yellow]  Escalation LLM skipped {record.meeting_id}: {e}[/yellow]")

        results.append(flag)

    critical = sum(1 for f in results if f.severity == "critical")
    moderate = sum(1 for f in results if f.severity == "moderate")
    console.print(f"[green]✓ Escalations: {critical} critical, {moderate} moderate[/green]")
    return results


# ---------------------------------------------------------------------------
# D. Speaker Engagement Analysis
# ---------------------------------------------------------------------------

def _compute_speaker_engagement(
    transcripts: list[TranscriptRecord],
) -> list[SpeakerEngagement]:
    results: list[SpeakerEngagement] = []
    internal_domain = "@aegiscloud.com"

    for record in transcripts:
        total_seconds = record.duration_minutes * 60.0

        # Group sentences by speaker
        by_speaker: dict[str, list] = {}
        for s in record.sentences:
            by_speaker.setdefault(s.speaker_name, []).append(s)

        # Determine internal speakers from email list
        internal_names: set[str] = set()
        for email in record.all_emails:
            if email.endswith(internal_domain):
                name_part = email.split("@")[0].replace(".", " ").replace("-", " ").title()
                internal_names.add(name_part)

        # Organizer / host
        host_email = record.organizer_email.lower()
        host_name_part = host_email.split("@")[0].replace(".", " ").title()

        for speaker_name, sentences in by_speaker.items():
            talk_time = sum(max(0, s.endTime - s.time) for s in sentences)
            talk_pct = (talk_time / total_seconds * 100) if total_seconds > 0 else 0.0

            sentiment_vals = [SENTIMENT_NUMERIC.get(s.sentimentType, 0.0) for s in sentences]
            avg_sentiment = mean(sentiment_vals) if sentiment_vals else 0.0

            is_host = host_name_part.lower() in speaker_name.lower()
            is_internal = any(n.lower() in speaker_name.lower() for n in internal_names) or is_host

            results.append(SpeakerEngagement(
                meeting_id=record.meeting_id,
                title=record.title,
                speaker_name=speaker_name,
                talk_time_seconds=round(talk_time, 1),
                talk_time_pct=round(talk_pct, 1),
                sentence_count=len(sentences),
                avg_sentiment_numeric=round(avg_sentiment, 3),
                is_host=is_host,
                is_internal=is_internal,
            ))

    console.print(f"[green]✓ Speaker engagement: {len(results)} speaker-call records[/green]")
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_summary(churn, escalations, actions, speakers):
    console.print(f"[bold]Insight summary:[/bold]")
    console.print(f"  Churn signals: {len(churn)}")
    console.print(f"  Action items: {len(actions)}")
    console.print(f"  Escalation flags: {sum(1 for e in escalations if e.escalation_detected)}")
    console.print(f"  Speaker records: {len(speakers)}")


def _extract_json(text: str) -> str:
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else "{}"
