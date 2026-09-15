from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class SepsisGraphState(TypedDict, total=False):
    patient_id: str
    question: str
    patient_data: Dict[str, Any]
    pending_exams: List[str]
    protocols: List[Dict[str, Any]]
    risk: Dict[str, Any]
    draft_answer: str
    final_answer: str
    sources: List[str]
    safety: Dict[str, Any]
    executed_nodes: List[str]
    generation_mode: str
    generation_fallback_reason: str | None
