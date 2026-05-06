"""
Data models for the Transcript Intelligence pipeline.

Pydantic models validate raw JSON at ingest time.
TypedDict PipelineState is used by LangGraph's StateGraph.
"""
from __future__ import annotations

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Literal types
# ---------------------------------------------------------------------------

CallType = Literal["internal", "support", "external"]
SentimentType = Literal["positive", "neutral", "negative"]
KeyMomentType = Literal[
    "concern", "feature_gap", "positive_pivot",
    "action_item", "technical_issue", "churn_signal",
]
RiskTier = Literal["high", "medium", "low"]
EscalationSeverity = Literal["critical", "moderate", "none"]


# ---------------------------------------------------------------------------
# Raw JSON schemas (Pydantic — used only at ingest time)
# ---------------------------------------------------------------------------

class KeyMoment(BaseModel):
    time: float
    text: str
    type: str  # keep as str — some values may not match our literal exactly
    speaker: str


class MeetingInfo(BaseModel):
    meetingId: str
    title: str
    organizerEmail: str
    host: str
    startTime: str
    endTime: str
    duration: float
    allEmails: list[str] = Field(default_factory=list)
    invitees: list[str] = Field(default_factory=list)


class SummaryData(BaseModel):
    summary: str
    actionItems: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    overallSentiment: str = "unknown"
    sentimentScore: float = 3.0
    keyMoments: list[KeyMoment] = Field(default_factory=list)
    meetingId: str = ""


class TranscriptSentence(BaseModel):
    sentence: str
    speaker_name: str
    sentimentType: str = "neutral"
    speaker_id: int | str = 0
    time: float = 0.0
    endTime: float = 0.0
    averageConfidence: float = 0.9
    index: int = 0

    @field_validator("speaker_id", mode="before")
    @classmethod
    def coerce_speaker_id(cls, v):
        return int(v) if isinstance(v, str) and v.isdigit() else v


class EventRecord(BaseModel):
    participantName: str
    timestamp: int | str = 0
    type: str = "Join"
    time: float = 0.0


class SpeakerRecord(BaseModel):
    speakerName: str
    timestamp: float | str = 0.0
    endTimeTs: float | str = 0.0


# ---------------------------------------------------------------------------
# Unified record (one per transcript directory)
# ---------------------------------------------------------------------------

class TranscriptRecord(BaseModel):
    meeting_id: str
    title: str
    organizer_email: str
    all_emails: list[str]
    duration_minutes: float
    start_time: str
    end_time: str

    summary_text: str
    action_items: list[str]
    topics: list[str]
    overall_sentiment: str
    sentiment_score: float  # 1–5
    key_moments: list[KeyMoment]

    sentences: list[TranscriptSentence]
    events: list[EventRecord]
    speakers: list[SpeakerRecord]
    speaker_meta: dict[str, str]

    # Filled during classify node
    call_type: Optional[CallType] = None
    call_type_confidence: Optional[str] = None  # "rule" | "llm"

    # Filled during embed node
    embedding: Optional[list[float]] = None  # 384-dim MiniLM vector

    # Filled during cluster node
    cluster_id: Optional[int] = None  # -1 = noise
    umap_x: Optional[float] = None
    umap_y: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Insight result models
# ---------------------------------------------------------------------------

class ChurnSignal(BaseModel):
    meeting_id: str
    title: str
    churn_score: float  # 0.0 – 1.0
    churn_moments: list[str]
    feature_gaps: list[str]
    sentiment_trend: str  # "declining" | "stable" | "improving"
    risk_tier: RiskTier
    # Filled by claude-sonnet-4-6 for high-risk calls
    churn_risk_reasoning: Optional[str] = None
    top_concerns: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    urgency: Optional[str] = None


class ActionItemOwnership(BaseModel):
    meeting_id: str
    title: str
    speaker_name: str
    action_item: str
    call_type: Optional[CallType]
    cluster_label: Optional[str]


class EscalationFlag(BaseModel):
    meeting_id: str
    title: str
    call_type: Optional[CallType]
    escalation_detected: bool
    signals: list[str]
    severity: EscalationSeverity
    recommended_owner: Optional[str] = None  # filled by LLM for critical


class SpeakerEngagement(BaseModel):
    meeting_id: str
    title: str
    speaker_name: str
    talk_time_seconds: float
    talk_time_pct: float
    sentence_count: int
    avg_sentiment_numeric: float  # positive=1, neutral=0, negative=-1
    is_host: bool = False
    is_internal: bool = False


# ---------------------------------------------------------------------------
# Cluster and sentiment aggregation results
# ---------------------------------------------------------------------------

class ClusterResult(BaseModel):
    cluster_id: int
    label: str
    description: str
    size: int
    representative_meeting_ids: list[str]
    dominant_call_types: dict[str, int]
    avg_sentiment_score: float


class SentimentAggregate(BaseModel):
    call_type: Optional[str]
    cluster_label: Optional[str]
    avg_score: float
    score_distribution: dict[str, int]  # overallSentiment value -> count
    weekly_trend: dict[str, float]       # iso_week -> avg score
    sentence_pos_ratio: float = 0.0
    sentence_neg_ratio: float = 0.0


# ---------------------------------------------------------------------------
# LangGraph TypedDict state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    # ingest
    transcripts: list[TranscriptRecord]
    ingest_errors: list[str]

    # classify
    classified_transcripts: list[TranscriptRecord]
    llm_classification_needed: list[str]
    classification_stats: dict[str, int]

    # embed
    embeddings_complete: bool

    # cluster
    cluster_assignments: dict[str, int]
    umap_coords_2d: dict[str, tuple[float, float]]
    num_clusters: int
    noise_count: int

    # label
    cluster_results: list[ClusterResult]

    # sentiment
    sentiment_aggregates: list[SentimentAggregate]
    global_sentiment_trend: dict[str, float]

    # insights
    churn_signals: list[ChurnSignal]
    action_item_ownership: list[ActionItemOwnership]
    escalation_flags: list[EscalationFlag]
    speaker_engagements: list[SpeakerEngagement]

    # report
    report_path: str
    chart_paths: list[str]
    results_json_path: str
