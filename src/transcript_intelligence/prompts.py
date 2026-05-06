"""
All LLM prompt templates for the Transcript Intelligence pipeline.

Each prompt instructs the model to return ONLY valid JSON — no preamble.
"""

CLASSIFICATION_PROMPT = """\
You are classifying a B2B SaaS call recording into one of three categories.

Company context: Aegis Cloud is the internal company. Email domain @aegiscloud.com belongs to employees.
All other email domains are external partners or customers.

Call details:
Title: {title}
Duration: {duration:.1f} minutes
Participants: {participants}
Topics discussed: {topics}
Summary excerpt: {summary}
Key moments: {key_moments_sample}

Classify this call into EXACTLY one category:
- "internal": All participants are @aegiscloud.com employees. Team meetings, standups, planning, retrospectives, engineering syncs.
- "support": A customer/partner is reporting bugs, requesting troubleshooting, or seeking help resolving a technical or product issue.
- "external": A customer-facing call that is NOT primarily support. Includes demos, onboarding, implementation, QBRs, renewals, customer success check-ins, discovery calls.

Respond with ONLY valid JSON, no explanation:
{{"call_type": "internal"}}
or
{{"call_type": "support"}}
or
{{"call_type": "external"}}
"""

CLUSTER_LABEL_PROMPT = """\
You are analyzing a group of {cluster_size} B2B SaaS call transcripts that were automatically \
clustered together based on semantic similarity.

Here are {num_examples} representative examples from this cluster:

{examples}

Based on these examples, identify the single unifying theme or topic for this cluster.

Requirements for the label:
- 4-7 words, specific and actionable
- Describes what these calls ARE ABOUT, not their outcome
- Good examples: "SSO Integration Troubleshooting", "Q4 Contract Renewal Negotiations", \
"Post-Outage Engineering Retrospective", "Customer Data Migration Onboarding"
- Bad examples: "Technical Issues", "Customer Calls", "Internal Meetings"

Respond with ONLY valid JSON:
{{
  "label": "<4-7 word concise topic label>",
  "description": "<1-2 sentence explanation of what unifies these calls and why they matter to stakeholders>"
}}
"""

CHURN_ENRICHMENT_PROMPT = """\
You are a senior customer success analyst. Analyze this B2B SaaS customer call to assess churn \
risk and provide specific recommended actions.

Call details:
Title: {title}
Date: {start_time}
Duration: {duration:.1f} minutes
Sentiment score: {sentiment_score:.1f}/5 (1=very negative, 5=very positive)
Sentiment trend during call: {sentiment_trend}

Meeting summary:
{summary}

Key moments flagged (concerns, feature gaps, churn signals):
{key_moments}

Action items from the call:
{action_items}

Provide a concise, actionable analysis as valid JSON:
{{
  "churn_risk_reasoning": "<2-3 sentence explanation of the specific risk signals you see>",
  "top_concerns": ["<specific concern 1>", "<specific concern 2>", "<specific concern 3>"],
  "recommended_actions": ["<concrete action 1>", "<concrete action 2>"],
  "urgency": "immediate"
}}

For urgency, use one of: "immediate" (act within 24h), "this_week", "this_month", "monitor"
Respond with ONLY valid JSON.
"""

ESCALATION_PROMPT = """\
You are a customer success operations manager. Review these escalation signals from a customer call \
and recommend the appropriate owner.

Call type: {call_type}
Escalation signals detected:
{signals_list}

Additional context:
- Sentiment score: {sentiment_score:.1f}/5
- Duration: {duration:.1f} minutes
- Has churn signals: {has_churn_signals}

Who should own this escalation?
- "account_manager": Relationship issue, renewal risk, dissatisfaction — CS team handles
- "engineering": Technical bug, outage impact, integration failure — eng team handles
- "executive": Strategic relationship at risk, large account churn threat — exec sponsor needed
- "none": No escalation needed

Respond with ONLY valid JSON:
{{
  "requires_escalation": true,
  "escalation_reason": "<one sentence describing the primary risk>",
  "recommended_owner": "account_manager"
}}
"""
