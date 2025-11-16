"""Pydantic models for structured LLM responses."""
from typing import List, Optional
from pydantic import BaseModel, Field, validator

class EvidenceItem(BaseModel):
    doc_id: str = Field(..., description="Document/passage identifier")
    source: str = Field(..., description="Source name or type")
    excerpt: str = Field(..., description="Relevant text excerpt")
    score: float = Field(..., ge=0.0, le=1.0)
    url: Optional[str] = Field(None)

class LLMResponse(BaseModel):
    final_answer: str = Field(...)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    provenance: List[str] = Field(default_factory=list)
    method: str = Field(default="unknown")
    reasoning: Optional[str] = Field(None)

    @validator('final_answer')
    def validate_answer_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Answer cannot be empty")
        return v.strip()

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    style: Optional[str] = Field("professional")
    context: Optional[List[dict]] = Field(None)

class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=3)
    wrong_answer: str = Field(..., min_length=1)
    correct_answer: str = Field(..., min_length=3)
    tags: Optional[List[str]] = Field(default_factory=list)
