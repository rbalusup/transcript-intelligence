# Transcript Intelligence

A production-grade LLM-powered pipeline for analyzing B2B SaaS call transcripts. It ingests structured meeting data, performs hybrid classification, semantic clustering, sentiment analysis, and generates executive-ready insights — including churn risk scoring, escalation detection, action item ownership, and speaker engagement analytics.

## Features

- **Hybrid Call Classification** — Rule-based routing with Claude Haiku fallback for ambiguous calls (internal / support / external)
- **Semantic Clustering** — Local MiniLM embeddings + UMAP + HDBSCAN to discover emergent topics across all calls
- **Automated Cluster Labeling** — Claude Sonnet generates concise, human-readable cluster labels and descriptions
- **Churn Risk Scoring** — Multi-signal scoring (churn moments, feature gaps, sentiment, trend) with LLM enrichment for high-risk calls
- **Escalation Detection** — Flags critical and moderate escalations with recommended owner (account manager / engineering / executive)
- **Action Item Ownership** — Infers speaker ownership via pattern matching and TF-IDF similarity
- **Speaker Engagement Analytics** — Talk time, sentiment per speaker, host/internal identification
- **Sentiment Aggregation** — Distributions and weekly trends per call type and cluster; sentiment arc (improving / stable / declining)
- **10 Interactive Visualizations** — Plotly charts: cluster maps, sentiment heatmaps, churn risk, escalation flags, action ownership, and more
- **Automated 12-Slide Deck** — Executive-ready PowerPoint generated at the end of every run

## Architecture

The pipeline is built as a **LangGraph state machine** with 9 sequential nodes:

```
ingest
  ↓
classify_call_type          ← rule-based
  ↓ (conditional)
llm_classify_fallback       ← claude-haiku-4-5 (ambiguous calls only)
  ↓
embed_transcripts           ← all-MiniLM-L6-v2 (local, no API cost)
  ↓
cluster_topics              ← UMAP + HDBSCAN
  ↓
label_clusters              ← claude-sonnet-4-6
  ↓
analyze_sentiment
  ↓
extract_insights            ← churn risk, escalations, action ownership, speaker engagement
  ↓
compile_report              ← charts (Plotly) + PPTX (python-pptx) + results.json
```

### Source Layout

```
src/transcript_intelligence/
├── __main__.py             # CLI entry point
├── models.py               # Pydantic input/output schemas + LangGraph PipelineState
├── prompts.py              # LLM prompt templates
├── pipeline/
│   ├── graph.py            # LangGraph state graph definition
│   └── nodes/
│       ├── ingest.py       # Load and validate 6-file JSON transcripts
│       ├── classify.py     # Rule-based + LLM fallback classification
│       ├── embed.py        # Sentence embeddings (batch=32, normalized)
│       ├── cluster.py      # UMAP dimensionality reduction + HDBSCAN clustering
│       ├── sentiment.py    # Sentiment aggregation and trend analysis
│       └── insights.py     # Churn risk, escalations, action ownership, speaker engagement
└── analysis/
    ├── visualizations.py   # 10 Plotly/Matplotlib charts
    └── report.py           # 12-slide PowerPoint generation
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Anthropic API key

## Installation

```bash
git clone <repo-url>
cd transcript-intelligence

# Install dependencies
uv sync

# Copy and configure environment
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | _(required)_ | Anthropic API key |
| `DATA_DIR` | `dataset` | Path to directory containing transcript folders |
| `OUTPUTS_DIR` | `outputs` | Path where results, charts, and PPTX are written |

## Input Format

Each transcript lives in its own directory under `DATA_DIR`, named by meeting ID. Six JSON files are expected per transcript:

```
dataset/
└── <meeting-id>/
    ├── meeting-info.json      # Title, organizer, attendees, timestamps
    ├── summary.json           # AI summary, topics, key moments, action items, sentiment score
    ├── transcript.json        # Sentence-level transcript with speaker & sentiment
    ├── events.json            # Join/leave events
    ├── speakers.json          # Speaker timing metadata
    └── speaker-meta.json      # Speaker name mappings
```

## Running the Pipeline

```bash
# Using the installed script
transcript-intelligence

# Or directly
uv run python -m transcript_intelligence
```

## Outputs

All artifacts are written to `OUTPUTS_DIR`:

```
outputs/
├── transcript_intelligence.pptx    # 12-slide executive presentation
├── results.json                    # Complete structured results
└── charts/
    ├── cluster_map.html
    ├── sentiment_by_call_type.html
    ├── sentiment_trend.html
    ├── churn_risk_heatmap.html
    ├── action_items_treemap.html
    ├── sentiment_by_cluster.html
    ├── escalation_flags.html
    ├── speaker_talk_time.html
    ├── key_moment_types.html
    └── cluster_composition.html
```

### results.json Structure

```json
{
  "summary": {
    "total_transcripts": 102,
    "classification_stats": { "internal": 12, "support": 28, "external": 62 },
    "num_clusters": 15,
    "noise_count": 4,
    "total_action_items": 340,
    "escalations_detected": 8,
    "high_churn_risk_calls": 11
  },
  "transcripts": [...],
  "clusters": [...],
  "churn_signals": [...],
  "escalation_flags": [...],
  "action_item_ownership": [...],
  "sentiment_aggregates": [...],
  "global_sentiment_trend": { ... }
}
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Local MiniLM embeddings | No API cost or latency; 384-dim vectors sufficient for ~100 transcript scale |
| Rules-first classification | Covers ~95% of calls in milliseconds; LLM fallback reserved for ambiguous edge cases |
| HDBSCAN over k-means | Discovers emergent topic clusters without requiring a pre-specified number of clusters; handles noise gracefully |
| LLM enrichment only for high-risk | Targets Claude API calls to churn (≥0.65 score) and critical escalations, minimizing cost |
| LangGraph state machine | Provides debuggable, observable, and easily extensible pipeline structure |
| Automated PPTX | Eliminates manual slide creation; enables rapid stakeholder communication after each pipeline run |

## Models Used

| Task | Model |
|---|---|
| Ambiguous call classification | `claude-haiku-4-5` |
| Cluster labeling, churn enrichment, escalation owner | `claude-sonnet-4-6` |
| Sentence embeddings | `all-MiniLM-L6-v2` (local) |

## Dependencies

Key packages (see `pyproject.toml` for full list):

- **Pipeline**: `langgraph`, `langchain-core`, `anthropic`
- **Embeddings**: `sentence-transformers`
- **Clustering**: `umap-learn`, `hdbscan`, `scikit-learn`
- **Data**: `numpy`, `pandas`, `pydantic`
- **Visualization**: `plotly`, `matplotlib`, `kaleido`
- **Reporting**: `python-pptx`
- **Utilities**: `python-dotenv`, `tqdm`, `rich`