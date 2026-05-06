"""
Node 3: Embed Transcripts

Generates 384-dim sentence embeddings using all-MiniLM-L6-v2 (local model).
Embedding text is a rich combination of summary, topics, and key moments.
"""
from __future__ import annotations

import numpy as np
from rich.console import Console
from sentence_transformers import SentenceTransformer

from transcript_intelligence.models import PipelineState, TranscriptRecord

console = Console()

_MODEL_NAME = "all-MiniLM-L6-v2"


def run(state: PipelineState) -> PipelineState:
    transcripts = state["classified_transcripts"]

    console.print(f"[cyan]Generating embeddings with {_MODEL_NAME}...[/cyan]")
    model = SentenceTransformer(_MODEL_NAME)

    texts = [_build_embedding_text(r) for r in transcripts]
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine-similarity ready
    )

    updated: list[TranscriptRecord] = []
    for i, record in enumerate(transcripts):
        updated.append(record.model_copy(update={"embedding": embeddings[i].tolist()}))

    console.print(f"[green]✓ Embedded {len(updated)} transcripts (dim={embeddings.shape[1]})[/green]")
    return {**state, "classified_transcripts": updated, "embeddings_complete": True}


def _build_embedding_text(record: TranscriptRecord) -> str:
    """Combine summary, topics, key moments, and action items for richer signal."""
    parts = [record.summary_text]

    if record.topics:
        parts.append("TOPICS: " + ", ".join(record.topics))

    km_texts = [km.text for km in record.key_moments[:5]]
    if km_texts:
        parts.append("KEY MOMENTS: " + " | ".join(km_texts))

    if record.action_items:
        parts.append("ACTIONS: " + "; ".join(record.action_items[:4]))

    return " ".join(parts)
