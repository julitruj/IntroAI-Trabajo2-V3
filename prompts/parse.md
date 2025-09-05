# VoC Conversation Parser

## Objective
Transform raw text from a single customer interaction file into structured JSON format with normalized messages and PII redaction.

## Context
You are analyzing customer service interaction transcripts. Each file contains exactly one conversation between a customer (cliente/customer/user) and an agent (agente/agent/employee/support). Your task is to parse, normalize, and redact PII from the conversation.

## Requirements

### 1. Conversation Identification
- Extract or derive conversation_id (use filename if not present in text)
- Detect language (es, en, etc.)
- Ensure this is a single interaction (error if multiple conversations detected)

### 2. Message Parsing
- Identify each turn in the conversation
- Extract timestamps if present
- Determine speaker_role for each message:
  - "cliente" for customer/user
  - "agente" for agent/employee/support
  - "unknown" if unclear
- Extract channel information if mentioned (call, chat, email, etc.)

### 3. PII Redaction
Replace personally identifiable information with bracketed tags:
- Email addresses → [EMAIL]
- Phone numbers → [PHONE] 
- Credit card numbers → [CARD]
- ID numbers (DNI, SSN, etc.) → [ID]

### 4. Metadata Extraction
Look for metadata in headers, footers, or explicit meta tags:
- interaction_id
- agent_id
- channel
- duration

### 5. Error Detection
Return appropriate error codes for:
- MULTI_INTERACTION_DETECTED: Multiple separate conversations found
- UNCLEAR_ROLES: Cannot distinguish customer from agent
- EMPTY_CONTENT: No meaningful conversation content

## Output JSON Schema

```json
{
  "conversation_id": "string",
  "language": "string (ISO 2-letter code)", 
  "messages": [
    {
      "interaction_id": "string|null",
      "timestamp": "string|null (preserve original format)",
      "channel": "string|null",
      "speaker_role": "cliente|agente|unknown",
      "text": "string (with PII redacted)"
    }
  ],
  "metadata": {
    "interaction_id": "string|null",
    "agent_id": "string|null", 
    "channel": "string|null",
    "duration": "string|null"
  },
  "error_code": "string|null",
  "error_message": "string|null"
}

