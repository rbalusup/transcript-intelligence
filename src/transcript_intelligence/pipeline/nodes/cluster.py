"""
Node 4 & 5: Cluster Topics + Label Clusters

Step 1: UMAP (10D) + HDBSCAN for clustering
Step 2: UMAP (2D) for visualization
Step 3: Claude claude-sonnet-4-6 assigns human-readable labels to each cluster
"""
from __future__ import annotations

import json
import re
from collections import Counter
from statistics import mean

import hdbscan
import numpy as np
import umap
from rich.console import Console

import anthropic
from transcript_intelligence.models import (
    ClusterResult,
    PipelineState,
    TranscriptRecord,
)
from transcript_intelligence.prompts import CLUSTER_LABEL_PROMPT

console = Console()


def run_clustering(state: PipelineState) -> PipelineState:
    """UMAP + HDBSCAN clustering pass."""
    transcripts = state["classified_transcripts"]
    embeddings = np.array([r.embedding for r in transcripts], dtype=np.float32)

    console.print(f"[cyan]Clustering {len(transcripts)} transcripts...[/cyan]")

    # Step 1: UMAP 10D for HDBSCAN (tight, low min_dist)
    reducer_10d = umap.UMAP(
        n_components=10,
        n_neighbors=10,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    reduced_10d = reducer_10d.fit_transform(embeddings)

    # Step 2: HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=4,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(reduced_10d)

    # Step 3: UMAP 2D for visualization (spread out, readable)
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=10,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords_2d = reducer_2d.fit_transform(embeddings)

    # Store back on records
    updated: list[TranscriptRecord] = []
    cluster_assignments: dict[str, int] = {}
    umap_coords: dict[str, tuple[float, float]] = {}

    for i, record in enumerate(transcripts):
        cid = int(labels[i])
        x, y = float(coords_2d[i, 0]), float(coords_2d[i, 1])
        updated.append(record.model_copy(update={"cluster_id": cid, "umap_x": x, "umap_y": y}))
        cluster_assignments[record.meeting_id] = cid
        umap_coords[record.meeting_id] = (x, y)

    unique_clusters = set(labels) - {-1}
    noise_count = int((labels == -1).sum())

    console.print(f"[green]✓ Found {len(unique_clusters)} clusters, {noise_count} noise points[/green]")
    cluster_dist = dict(Counter(labels.tolist()))
    console.print(f"  Distribution: {cluster_dist}")

    return {
        **state,
        "classified_transcripts": updated,
        "cluster_assignments": cluster_assignments,
        "umap_coords_2d": umap_coords,
        "num_clusters": len(unique_clusters),
        "noise_count": noise_count,
    }


def run_labeling(state: PipelineState) -> PipelineState:
    """Claude claude-sonnet-4-6 labels each cluster with a human-readable name."""
    transcripts = state["classified_transcripts"]
    num_clusters = state["num_clusters"]

    client = anthropic.Anthropic()
    cluster_results: list[ClusterResult] = []

    # Group records by cluster
    by_cluster: dict[int, list[TranscriptRecord]] = {}
    for r in transcripts:
        cid = r.cluster_id if r.cluster_id is not None else -1
        by_cluster.setdefault(cid, []).append(r)

    console.print(f"[cyan]Labeling {num_clusters} clusters with Claude...[/cyan]")

    for cluster_id, members in sorted(by_cluster.items()):
        if cluster_id == -1:
            # Noise cluster — no LLM call needed
            cluster_results.append(ClusterResult(
                cluster_id=-1,
                label="Uncategorized",
                description="Transcripts that did not fit a clear topic cluster.",
                size=len(members),
                representative_meeting_ids=[m.meeting_id for m in members[:3]],
                dominant_call_types=dict(Counter(m.call_type for m in members)),
                avg_sentiment_score=mean(m.sentiment_score for m in members),
            ))
            continue

        reps = members[:5]
        examples_parts = []
        for r in reps:
            examples_parts.append(
                f"Title: {r.title}\n"
                f"Summary: {r.summary_text[:300]}\n"
                f"Topics: {', '.join(r.topics[:8])}\n"
                f"Action items: {'; '.join(r.action_items[:3])}\n"
                "---"
            )
        examples_text = "\n".join(examples_parts)

        prompt = CLUSTER_LABEL_PROMPT.format(
            cluster_size=len(members),
            num_examples=len(reps),
            examples=examples_text,
        )

        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            parsed = json.loads(_extract_json(raw))
            label = parsed.get("label", f"Cluster {cluster_id}")
            description = parsed.get("description", "")
        except Exception as e:
            console.print(f"[red]  Label failed for cluster {cluster_id}: {e}[/red]")
            label = f"Cluster {cluster_id}"
            description = ""

        cluster_results.append(ClusterResult(
            cluster_id=cluster_id,
            label=label,
            description=description,
            size=len(members),
            representative_meeting_ids=[m.meeting_id for m in reps],
            dominant_call_types=dict(Counter(m.call_type for m in members)),
            avg_sentiment_score=mean(m.sentiment_score for m in members),
        ))
        console.print(f"  Cluster {cluster_id} ({len(members)} calls): [bold]{label}[/bold]")

    return {**state, "cluster_results": cluster_results}


def _extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text
