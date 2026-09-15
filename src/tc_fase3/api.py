from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .assistant import SepsisAssistant
from .audit_logger import read_latest_logs
from .patient_repository import PatientRepository
from .protocol_retriever import ProtocolRetriever
from .schemas import AskRequest, AssistantResponse
from .langgraph_flow import run_sepsis_flow

app = FastAPI(
    title="API Fase 3 - Assistente Médico de Sepse",
    description="Assistente acadêmico para apoio à triagem de sepse com fontes, segurança e auditoria.",
    version="0.1.0",
)

repository = PatientRepository()
retriever = ProtocolRetriever()
assistant = SepsisAssistant(repository=repository, retriever=retriever)


@app.get("/fase3/health")
def health() -> dict:
    return {
        "status": "ok",
        "patients_loaded": repository.count(),
        "protocols_loaded": retriever.count(),
        "mode": "academic_synthetic",
    }


@app.post("/fase3/assistant/ask", response_model=AssistantResponse)
def ask(payload: AskRequest) -> dict:
    try:
        return assistant.answer(payload.patient_id, payload.question)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/fase3/assistant/flow", response_model=AssistantResponse)
def flow(payload: AskRequest) -> dict:
    try:
        return run_sepsis_flow(payload.patient_id, payload.question)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/fase3/logs/latest")
def latest_logs(limit: int = 20) -> dict:
    safe_limit = max(1, min(int(limit), 100))
    return {"lines": read_latest_logs(limit=safe_limit)}
