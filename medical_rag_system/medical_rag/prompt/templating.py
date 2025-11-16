"""Prompt templates and response parsing."""
import re, json
from typing import List, Dict, Any, Optional
from ..schema.response_schema import LLMResponse, EvidenceItem
from ..utils.logging import get_logger

logger = get_logger(__name__)

class PromptTemplate:
    @staticmethod
    def create_reasoning_prompt(question: str, context: str, corrections: List[Dict], conversation_history: str = "") -> str:
        corrections_str = ""
        if corrections:
            corrections_str = "\n\n### Important Past Corrections:\n"
            for corr in corrections[:3]:
                corrections_str += f"Q: {corr['question']}\nCorrect Answer: {corr['correct_answer']}\n\n"
        conv_str = ""
        if conversation_history:
            conv_str = f"\n\n### Recent Conversation:\n{conversation_history}\n"
        prompt = f"""You are a medical expert AI. Think step-by-step to answer the question.

{corrections_str}

### Medical Evidence:
{context}
{conv_str}

### Question: {question}

### Internal Reasoning (step-by-step analysis):
"""
        return prompt

    @staticmethod
    def create_final_answer_prompt(question: str, reasoning: str, evidence_passages: List[Dict]) -> str:
        evidence_str = "\n".join([f"[{i+1}] {p['text'][:200]}... (score: {p.get('score', 0):.2f})" for i, p in enumerate(evidence_passages[:5])])
        prompt = f"""Based on the reasoning and evidence, provide a structured JSON response.

### Internal Reasoning:
{reasoning}

### Evidence:
{evidence_str}

### Question: {question}

Respond with ONLY a JSON object matching this schema:
{{"final_answer": "2-3 sentence concise answer", "confidence": 0.85, "evidence_ids": ["passage_1", "passage_2"], "key_points": ["point 1", "point 2"]}}

JSON Response:
"""
        return prompt

class ResponseParser:
    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        json_match = re.search(r'\{[^{{}}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("json_parse_failed", text=text[:200])
            return None

    @staticmethod
    def parse_response(generated_text: str, prompt: str, evidence_items: List[EvidenceItem], confidence: float) -> LLMResponse:
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text[len(prompt):].strip()
        answer = re.sub(r'\[.*?\]', '', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()
        sentences = [s.strip() + '.' for s in answer.split('.') if s.strip() and len(s.strip()) > 20]
        final_answer = ' '.join(sentences[:4]) if sentences else answer
        provenance = []
        for item in evidence_items:
            if item.url:
                provenance.append(item.url)
            else:
                provenance.append(f"{item.source}:{item.doc_id}")
        return LLMResponse(final_answer=final_answer if final_answer else "Unable to generate answer.", evidence=evidence_items, confidence=confidence, provenance=list(set(provenance)))
