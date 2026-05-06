"""
Node 2: Classify Call Type

Hybrid approach:
  1. Rule-based pass: email domain analysis, title/topic keyword matching
  2. LLM fallback (claude-haiku-4-5): for ambiguous calls
"""
from __future__ import annotations

import json
import re
from collections import Counter

import anthropic
from rich.console import Console

from transcript_intelligence.models import PipelineState, TranscriptRecord
from transcript_intelligence.prompts import CLASSIFICATION_PROMPT

console = Console()

INTERNAL_DOMAIN = "@aegiscloud.com"

SUPPORT_KEYWORDS = [
    "support", "ticket", "help desk", "helpdesk", "incident", "bug",
    "troubleshoot", "outage", "issue", "problem", "error", "failure",
    "root cause", "rca", "post-mortem", "postmortem", "escalat",
]

EXTERNAL_KEYWORDS = [
    "renewal", "renew", "onboarding", "onboard", "implementation", "deploy",
    "qbr", "business review", "quarterly", "demo", "discovery", "kickoff",
    "kick-off", "adoption", "account review", "customer success",
    "check-in", "check in", "handoff", "hand-off", "go-live", "golive",
]


def run(state: PipelineState) -> PipelineState:
    """Rule-based classification pass."""
    transcripts = state["transcripts"]
    classified: list[TranscriptRecord] = []
    llm_needed: list[str] = []

    for record in transcripts:
        call_type = _rule_classify(record)
        if call_type:
            record = record.model_copy(update={"call_type": call_type, "call_type_confidence": "rule"})
        else:
            llm_needed.append(record.meeting_id)
        classified.append(record)

    stats = dict(Counter(r.call_type for r in classified if r.call_type))
    console.print(f"[cyan]Classification (rule-based):[/cyan] {stats}")
    console.print(f"[yellow]  {len(llm_needed)} calls need LLM classification[/yellow]")

    return {
        **state,
        "classified_transcripts": classified,
        "llm_classification_needed": llm_needed,
        "classification_stats": stats,
    }


def run_llm_fallback(state: PipelineState) -> PipelineState:
    """LLM fallback for ambiguous calls using claude-haiku-4-5."""
    classified = state["classified_transcripts"]
    llm_needed = state["llm_classification_needed"]

    if not llm_needed:
        return state

    client = anthropic.Anthropic()
    record_map = {r.meeting_id: r for r in classified}
    updated: list[TranscriptRecord] = []

    console.print(f"[cyan]LLM classifying {len(llm_needed)} ambiguous calls...[/cyan]")

    for meeting_id in llm_needed:
        record = record_map[meeting_id]
        key_moments_sample = "; ".join(km.text[:80] for km in record.key_moments[:3])

        prompt = CLASSIFICATION_PROMPT.format(
            title=record.title,
            duration=record.duration_minutes,
            participants=", ".join(record.all_emails[:8]),
            topics=", ".join(record.topics[:10]),
            summary=record.summary_text[:600],
            key_moments_sample=key_moments_sample or "none",
        )

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=60,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            parsed = json.loads(_extract_json(raw))
            call_type = parsed.get("call_type", "internal")
            # Validate
            if call_type not in ("internal", "support", "external"):
                call_type = "internal"
            record = record.model_copy(update={"call_type": call_type, "call_type_confidence": "llm"})
        except Exception as e:
            console.print(f"[red]  LLM classify failed for {meeting_id}: {e}[/red]")
            # Safe default: if no external emails, internal; else external
            has_external = any(not e.endswith(INTERNAL_DOMAIN) for e in record.all_emails)
            record = record.model_copy(update={
                "call_type": "external" if has_external else "internal",
                "call_type_confidence": "fallback",
            })

        record_map[meeting_id] = record

    updated = [record_map.get(r.meeting_id, r) for r in classified]
    stats = dict(Counter(r.call_type for r in updated if r.call_type))
    console.print(f"[green]✓ Final classification:[/green] {stats}")

    return {**state, "classified_transcripts": updated, "classification_stats": stats}


def _rule_classify(record: TranscriptRecord) -> str | None:
    """Returns call_type string or None if ambiguous."""
    emails = set(record.all_emails)
    internal_emails = {e for e in emails if e.endswith(INTERNAL_DOMAIN)}
    external_emails = emails - internal_emails

    title_lower = record.title.lower()
    topics_lower = " ".join(record.topics).lower()
    combined = f"{title_lower} {topics_lower}"

    km_types = {km.type for km in record.key_moments}

    # Rule 1: all internal emails — definitely internal
    if emails and not external_emails:
        return "internal"

    # Rule 2: explicit support signals in title
    if any(kw in title_lower for kw in SUPPORT_KEYWORDS):
        return "support"

    # Rule 3: external keywords in title or topics
    if any(kw in combined for kw in EXTERNAL_KEYWORDS):
        return "external"

    # Rule 4: churn signal in key moments AND external participants present
    if "churn_signal" in km_types and external_emails:
        return "external"

    # Rule 5: any external emails without clearer signal → external
    if external_emails:
        return "external"

    return None


def _extract_json(text: str) -> str:
    """Extract JSON object from LLM response that may have extra text or markdown fences."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    return match.group(0) if match else '{"call_type": "internal"}'
