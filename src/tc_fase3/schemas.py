from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["baixo", "moderado", "alto"]
SafetyStatus = Literal["approved", "blocked", "needs_completion"]


class AskRequest(BaseModel):
    patient_id: str = Field(..., examples=["PACIENTE_001"])
    question: str = Field(..., examples=["Quais sinais indicam risco de sepse neste paciente?"])


class RiskPayload(BaseModel):
    risk_level: RiskLevel
    risk_score: float
    used_model: str
    factors: List[str] = []


class SafetyPayload(BaseModel):
    status: SafetyStatus
    human_validation_required: bool = True
    limitations: List[str] = []


class AssistantResponse(BaseModel):
    patient_id: str
    question: str
    answer: str
    risk: RiskPayload
    sources: List[str]
    safety: SafetyPayload
    executed_nodes: List[str] = []
    generation_mode: Literal["fine_tuned_langchain", "template_fallback"] = "template_fallback"
    generation_fallback_reason: str | None = None


JsonDict = Dict[str, Any]
