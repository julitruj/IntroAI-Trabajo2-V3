import json
import logging
import os
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import re

# LLM SDK imports
import openai
from openai import OpenAI
import anthropic
from anthropic import Anthropic
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for LLM model selection"""
    provider: str  # 'openai', 'anthropic', 'gemini'
    model: str
    api_key: str
    max_retries: int = 3
    retry_delay: float = 1.0

class LLMBackend:
    """Backend service for LLM-based VoC analysis"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = self._initialize_client()
        self.parse_prompt = self._load_parse_prompt()
        self.analyze_prompt = self._load_analyze_prompt()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        if self.config.provider == 'openai':
            return OpenAI(api_key=self.config.api_key)
        elif self.config.provider == 'anthropic':
            return Anthropic(api_key=self.config.api_key)
        elif self.config.provider == 'gemini':
            return genai.Client(api_key=self.config.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _load_parse_prompt(self) -> str:
        """Load parsing prompt template"""
        try:
            with open('prompts/parse.md', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("Parse prompt file not found, using embedded prompt")
            return self._get_embedded_parse_prompt()
    
    def _load_analyze_prompt(self) -> str:
        """Load analysis prompt template"""
        try:
            with open('prompts/analyze.md', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("Analyze prompt file not found, using embedded prompt")
            return self._get_embedded_analyze_prompt()
    
    def _get_embedded_parse_prompt(self) -> str:
        """Embedded parsing prompt as fallback"""
        return """
# VoC Conversation Parser

## Objective
Transform raw text from a single customer interaction file into structured JSON format with normalized messages and PII redaction.

## Requirements
1. Extract conversation_id (derive from filename if not present)
2. Detect language (es, en, etc.)
3. Parse messages with clear speaker roles (cliente/customer vs agente/agent)
4. Redact PII: [EMAIL], [PHONE], [CARD], [ID]
5. Extract metadata if present
6. Ensure 1 file = 1 interaction (error if multiple detected)

## Output JSON Schema
```json
{
  "conversation_id": "string",
  "language": "string", 
  "messages": [
    {
      "interaction_id": "string",
      "timestamp": "string|null",
      "channel": "string",
      "speaker_role": "cliente|agente|unknown",
      "text": "string (PII redacted)"
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
```

Return only valid JSON in your response.
"""

    def _get_embedded_analyze_prompt(self) -> str:
        """Embedded analysis prompt as fallback"""
        return """
# VoC Global Analysis

## Objective
Analyze customer interactions to produce comprehensive VoC insights.

## Requirements
1. Topic Discovery (5-20 topics)
2. Sentiment Analysis (-1.0 to 1.0)
3. GEW Emotion Analysis (8 categories)
4. Simulated NPS/CSAT scores
5. SMART Recommendations

Return only valid JSON matching the expected schema.
"""

    def _call_llm(self, prompt: str, system_prompt: str = "") -> str:
        """Generic LLM call with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                if self.config.provider == 'openai':
                    # The newest OpenAI model is "gpt-5" which was released August 7, 2025.
                    # do not change this unless explicitly requested by the user
                    messages = [{"role": "user", "content": prompt}]
                    if system_prompt:
                        messages.insert(0, {"role": "system", "content": system_prompt})
                    
                    response = self.client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        response_format={"type": "json_object"}
                    )
                    return response.choices[0].message.content or ""
                
                elif self.config.provider == 'anthropic':
                    # The newest Anthropic model is "claude-sonnet-4-20250514", not "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022" nor "claude-3-sonnet-20240229". 
                    # If the user doesn't specify a model, always prefer using "claude-sonnet-4-20250514" as it is the latest model. However, if the user has already selected "claude-3-7-sonnet-20250219", keep that selection unless they explicitly request a change.
                    final_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                    response = self.client.messages.create(
                        model=self.config.model,
                        max_tokens=4000,
                        messages=[{"role": "user", "content": final_prompt}]
                    )
                    return response.content[0].text if response.content else ""
                
                elif self.config.provider == 'gemini':
                    # Note that the newest Gemini model series is "gemini-2.5-flash" or gemini-2.5-pro"
                    # - do not change this unless explicitly requested by the user
                    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                    response = self.client.models.generate_content(
                        model=self.config.model,
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json"
                        )
                    )
                    return response.text or ""
                
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    raise e
        
        return ""  # Fallback return
    
    def analyze_conversations_batch(self, file_contents: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze multiple files in a single LLM call for better performance"""
        try:
            # Prepare all files for batch processing
            files_text = ""
            for i, file_data in enumerate(file_contents):
                files_text += f"\n\n=== FILE {i+1}: {file_data['filename']} ===\n"
                files_text += file_data['content']
            
            # Single optimized prompt that handles both parsing AND analysis
            batch_prompt = f"""
You are a VoC analyst processing customer service conversations. Analyze ALL files in this batch and return comprehensive insights.

## TASK
1. Parse each conversation (detect roles: cliente/agente, redact PII with [EMAIL], [PHONE], [CARD], [ID])
2. Discover 3-10 topics across all conversations
3. Analyze sentiment and emotions for key messages
4. Generate 2-5 SMART recommendations
5. Calculate simulated NPS/CSAT scores

## INPUT FILES
{files_text}

## OUTPUT SCHEMA
Return ONLY valid JSON with this structure:
{{
  "kpis": {{
    "nps": {{"value": "number", "promoters": "number", "detractors": "number", "passives": "number", "simulated": true}},
    "csat": {{"mean": "number (1-5)", "simulated": true}},
    "sentiment": {{"neg": "decimal (0-1)", "neu": "decimal (0-1)", "pos": "decimal (0-1)"}}
  }},
  "topics": [
    {{"topic_id": "number", "label": "string", "description": "string", "keywords": ["string"], "summary": "string", "bullets": ["string"]}}
  ],
  "message_assignments": [
    {{"conversation_id": "string", "topic_id": "number", "sentiment_label": "neg|neu|pos", "sentiment_score": "number", "familia_gew": "string", "intensidad": "number", "valencia": "number"}}
  ],
  "recommendations": [
    {{"topic_id": "number", "what": "string", "who": "string", "when": "string", "metric": "string", "impact": "string", "tag": "quick win|proceso|producto|formación|política"}}
  ],
  "conversations": [
    {{"conversation_id": "string", "language": "string", "nps_simulated": "number", "csat_simulated": "number"}}
  ]
}}
"""
            
            system_prompt = "You are a VoC analyst. Process all files in batch and return comprehensive analysis in a single JSON response."
            
            response = self._call_llm(batch_prompt, system_prompt)
            
            # Extract and parse JSON from response
            json_text = self._extract_json_from_response(response)
            result = json.loads(json_text)
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze conversations in batch: {str(e)}")
            return {
                "kpis": {
                    "nps": {"value": 0, "promoters": 0, "detractors": 0, "passives": 0, "simulated": True},
                    "csat": {"mean": 0, "simulated": True},
                    "sentiment": {"neg": 0.33, "neu": 0.34, "pos": 0.33}
                },
                "topics": [],
                "message_assignments": [],
                "recommendations": [],
                "conversations": []
            }
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from response text, handling markdown and other formatting"""
        if not response_text:
            return "{}"
        
        # Try to find JSON content within markdown code blocks
        import re
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response_text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Try to find JSON object directly
        json_pattern = r'(\{.*\})'
        match = re.search(json_pattern, response_text, re.DOTALL)
        if match:
            return match.group(1)
        
        # If no JSON found, return the original response
        return response_text.strip()
    
    def parse_conversation(self, text_content: str, filename: str) -> Dict[str, Any]:
        """Parse a single conversation file"""
        try:
            prompt = f"""
{self.parse_prompt}

## Input File
Filename: {filename}
Content:
{text_content}

Parse this conversation and return ONLY valid JSON following the schema above. Do not include any explanations, markdown formatting, or additional text. Return only the JSON object.
"""
            
            system_prompt = "You are a VoC conversation parser. Parse the given conversation text into structured JSON format with PII redaction."
            
            response = self._call_llm(prompt, system_prompt)
            
            # Extract and parse JSON from response
            json_text = self._extract_json_from_response(response)
            result = json.loads(json_text)
            
            # Set conversation_id from filename if not present
            if not result.get('conversation_id'):
                result['conversation_id'] = os.path.splitext(filename)[0]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse conversation {filename}: {str(e)}")
            return {
                "conversation_id": os.path.splitext(filename)[0],
                "language": "unknown",
                "messages": [],
                "metadata": {},
                "error_code": "PARSING_FAILED",
                "error_message": str(e)
            }
    
    def analyze_conversations(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze multiple conversations to extract VoC insights"""
        try:
            # Prepare conversation data for analysis
            conversations_text = json.dumps(conversations, indent=2, ensure_ascii=False)
            
            prompt = f"""
{self.analyze_prompt}

## Input Conversations
{conversations_text}

Analyze these conversations and return ONLY valid JSON following the schema above. Do not include any explanations, markdown formatting, or additional text. Return only the JSON object.
"""
            
            system_prompt = "You are a VoC analyst. Analyze customer conversations to extract topics, sentiment, emotions, and recommendations."
            
            response = self._call_llm(prompt, system_prompt)
            
            # Extract and parse JSON from response
            json_text = self._extract_json_from_response(response)
            result = json.loads(json_text)
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze conversations: {str(e)}")
            return {
                "kpis": {
                    "nps": {"value": 0, "promoters": 0, "detractors": 0, "passives": 0, "simulated": True},
                    "csat": {"mean": 0, "simulated": True},
                    "sentiment": {"neg": 0, "neu": 1.0, "pos": 0}
                },
                "topics": [],
                "message_assignments": [],
                "recommendations": [],
                "conversations": []
            }
