"""
Node 1: Ingest

Walks the data directory, loads and validates all 6 JSON files per transcript,
and merges them into TranscriptRecord objects.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console

from transcript_intelligence.models import (
    EventRecord,
    MeetingInfo,
    PipelineState,
    SpeakerRecord,
    SummaryData,
    TranscriptRecord,
    TranscriptSentence,
)

console = Console()


def run(state: PipelineState) -> PipelineState:
    # data_dir = Path(os.getenv("DATA_DIR", "data/transcripts"))
    data_dir = Path(os.getenv("DATA_DIR", "dataset"))
    transcripts: list[TranscriptRecord] = []
    errors: list[str] = []

    dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    console.print(f"[bold cyan]Ingesting {len(dirs)} transcript directories...[/bold cyan]")

    for d in dirs:
        meeting_id = d.name
        try:
            meeting_info = _load_json(d / "meeting-info.json")
            summary_raw = _load_json(d / "summary.json")
            transcript_raw = _load_json(d / "transcript.json")
            events_raw = _load_json(d / "events.json")
            speakers_raw = _load_json(d / "speakers.json")
            speaker_meta_raw = _load_json(d / "speaker-meta.json")

            info = MeetingInfo.model_validate(meeting_info)
            summary = SummaryData.model_validate(summary_raw)

            # transcript.json wraps the array under "data"
            raw_sentences = transcript_raw.get("data", transcript_raw) if isinstance(transcript_raw, dict) else transcript_raw
            sentences = [TranscriptSentence.model_validate(s) for s in raw_sentences]

            raw_events = events_raw if isinstance(events_raw, list) else []
            events = [EventRecord.model_validate(e) for e in raw_events]

            raw_speakers = speakers_raw if isinstance(speakers_raw, list) else []
            speakers = [SpeakerRecord.model_validate(sp) for sp in raw_speakers]

            speaker_meta: dict[str, str] = speaker_meta_raw if isinstance(speaker_meta_raw, dict) else {}

            record = TranscriptRecord(
                meeting_id=info.meetingId,
                title=info.title,
                organizer_email=info.organizerEmail,
                all_emails=info.allEmails,
                duration_minutes=info.duration,
                start_time=info.startTime,
                end_time=info.endTime,
                summary_text=summary.summary,
                action_items=summary.actionItems,
                topics=summary.topics,
                overall_sentiment=summary.overallSentiment,
                sentiment_score=summary.sentimentScore,
                key_moments=summary.keyMoments,
                sentences=sentences,
                events=events,
                speakers=speakers,
                speaker_meta=speaker_meta,
            )
            transcripts.append(record)

        except (FileNotFoundError, json.JSONDecodeError, ValidationError, KeyError) as e:
            console.print(f"[yellow]  Warning: skipping {meeting_id} — {type(e).__name__}: {e}[/yellow]")
            errors.append(meeting_id)

    console.print(f"[green]✓ Loaded {len(transcripts)} transcripts ({len(errors)} errors)[/green]")
    return {**state, "transcripts": transcripts, "ingest_errors": errors}


def _load_json(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
