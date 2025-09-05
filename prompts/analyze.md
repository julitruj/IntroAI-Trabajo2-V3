# VoC Global Analysis

## Objective
Analyze normalized customer interaction conversations to produce comprehensive Voice of Customer insights including topics, sentiment analysis, emotions, simulated satisfaction scores, and actionable SMART recommendations.

## Context
You are a VoC (Voice of Customer) analyst examining customer service interactions. Your goal is to extract meaningful patterns, sentiments, and actionable insights that can help improve customer experience and business operations.

## Analysis Requirements

### 1. Topic Discovery (5-20 topics)
- Identify key themes across all conversations
- Generate unique topic_id (numeric)
- Create concise labels (≤60 characters)
- Write descriptive explanations (≤300 characters)
- Extract 3-5 relevant keywords per topic
- Provide topic summaries (≤120 words)
- Include 3 bullet points highlighting causes, impact, and improvement ideas

### 2. Message-Level Analysis
For each message, provide:
- **Topic Assignment**: Assign to most relevant topic_id
- **Sentiment Analysis**: 
  - Label: "neg" (negative), "neu" (neutral), "pos" (positive)
  - Score: -1.0 to 1.0 (continuous sentiment strength)
- **GEW Emotion Analysis**:
  - Familia_gew: One of the 8 emotions below
  - Intensidad: 1-5 scale
  - Valencia: -1.0 to 1.0 (emotional valence)

### 3. GEW Emotion Families
Use exactly these emotion categories:
- alegría (joy)
- confianza (trust)
- miedo (fear)
- sorpresa (surprise)
- tristeza (sadness)
- enojo (anger)
- aversión (disgust)
- anticipación (anticipation)

### 4. Simulated Satisfaction Scores
For each conversation (when not explicitly provided):
- **NPS Simulated**: 0-10 scale (Net Promoter Score)
- **CSAT Simulated**: 1-5 scale (Customer Satisfaction)

### 5. SMART Recommendations
Generate 2-5 actionable recommendations per topic:
- **What**: Specific action to take
- **Who**: Responsible person/team
- **When**: Timeline for implementation
- **Metric**: How to measure success
- **Impact**: Expected business impact
- **Tag**: Classification (see options below)

### 6. Recommendation Tags
Choose from these categories:
- **quick win**: Easy, immediate improvements
- **proceso**: Process optimization
- **producto**: Product enhancement
- **formación**: Training/education
- **política**: Policy change

## Output lenguague
Asegurate que todos los textos de output esten en Español Estandard

## Output JSON Schema

```json
{
  "kpis": {
    "nps": {
      "value": "number (overall NPS score)",
      "promoters": "number (% promoters)",
      "detractors": "number (% detractors)", 
      "passives": "number (% passives)",
      "simulated": true
    },
    "csat": {
      "mean": "number (average CSAT 1-5)",
      "simulated": true
    },
    "sentiment": {
      "neg": "number (% negative)",
      "neu": "number (% neutral)",
      "pos": "number (% positive)"
    }
  },
  "topics": [
    {
      "topic_id": "number",
      "label": "string (≤60 chars)",
      "description": "string (≤300 chars)", 
      "keywords": ["string", "string", "string"],
      "summary": "string (≤120 words)",
      "bullets": [
        "- Cause/root issue description",
        "- Impact on customer experience", 
        "- Improvement opportunity"
      ]
    }
  ],
  "message_assignments": [
    {
      "conversation_id": "string",
      "interaction_id": "string|null", 
      "topic_id": "number",
      "sentiment_label": "neg|neu|pos",
      "sentiment_score": "number (-1.0 to 1.0)",
      "familia_gew": "string (one of 8 emotions)",
      "intensidad": "number (1-5)",
      "valencia": "number (-1.0 to 1.0)"
    }
  ],
  "recommendations": [
    {
      "topic_id": "number",
      "what": "string (specific action)",
      "who": "string (responsible party)",
      "when": "string (timeline)",
      "metric": "string (success measure)",
      "impact": "string (expected outcome)",
      "tag": "quick win|proceso|producto|formación|política"
    }
  ],
  "conversations": [
    {
      "conversation_id": "string",
      "language": "string",
      "nps_simulated": "number (0-10)",
      "csat_simulated": "number (1-5)"
    }
  ]
}
