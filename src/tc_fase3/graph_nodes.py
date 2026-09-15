from __future__ import annotations

from typing import Any, Dict, List

from .audit_logger import write_audit_log
from .patient_repository import PatientRepository
from .phase2_risk_tool import estimate_sepsis_risk
from .protocol_retriever import ProtocolRetriever
from .safety import SAFETY_LIMITATIONS, complete_safe_answer, validate_question


def _mark(state: Dict[str, Any], node: str) -> Dict[str, Any]:
    state.setdefault("executed_nodes", []).append(node)
    return state


def validate_input_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state["safety"] = validate_question(state.get("question", ""))
    return _mark(state, "validate_input_node")


def load_patient_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state["patient_data"] = PatientRepository().get_patient(state["patient_id"])
    state.setdefault("sources", []).append("pacientes_sinteticos.jsonl")
    return _mark(state, "load_patient_node")


def check_pending_exams_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state["pending_exams"] = list(state.get("patient_data", {}).get("exames_pendentes", []))
    return _mark(state, "check_pending_exams_node")


def retrieve_protocols_node(state: Dict[str, Any]) -> Dict[str, Any]:
    retriever = ProtocolRetriever()
    patient = state.get("patient_data", {})
    query = f"{state.get('question', '')} {patient.get('queixa_principal', '')} sepse exames alerta segurança"
    protocols = retriever.search(query, limit=3)
    if state.get("safety", {}).get("status") == "blocked":
        limits = [doc for doc in retriever.documents if doc.source == "protocolo_limites_assistente.md"]
        if limits:
            protocols = [{"source": limits[0].source, "content": limits[0].content, "score": 99}] + protocols[:2]
    state["protocols"] = protocols
    state["sources"] = list(dict.fromkeys(state.get("sources", []) + [item["source"] for item in protocols]))
    return _mark(state, "retrieve_protocols_node")


def estimate_risk_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state["risk"] = estimate_sepsis_risk(state.get("patient_data", {}))
    return _mark(state, "estimate_risk_node")


def generate_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    patient = state.get("patient_data", {})
    risk = state.get("risk", {})
    factors = ", ".join(risk.get("factors", [])) or "sem fatores críticos claros"
    pending = ", ".join(state.get("pending_exams", [])) or "não informados"
    source_text = ", ".join(state.get("sources", [])) or "protocolos internos sintéticos"
    if state.get("safety", {}).get("status") == "blocked":
        state["draft_answer"] = (
            "Solicitação bloqueada por segurança: o assistente não prescreve, não informa dose e não fecha diagnóstico definitivo. "
            f"Pode apoiar a triagem listando fatores ({factors}) e exames pendentes ({pending}). "
            "A validação humana obrigatória é necessária. "
            f"Fontes/protocolos usados: {source_text}."
        )
    else:
        state["draft_answer"] = (
            f"Paciente sintético {patient.get('patient_id')} com queixa de {patient.get('queixa_principal')}. "
            f"O risco estimado é {risk.get('risk_level')} (score {risk.get('risk_score')}, método {risk.get('used_model')}). "
            f"Fatores de risco: {factors}. Exames pendentes: {pending}. "
            f"Fontes/protocolos usados: {source_text}. "
            "Esta resposta não é diagnóstico definitivo e exige validação humana obrigatória."
        )
    return _mark(state, "generate_answer_node")


def safety_validation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state["final_answer"] = complete_safe_answer(state.get("draft_answer", ""), state.get("sources", []))
    state["safety"] = {
        "status": state.get("safety", {}).get("status", "approved"),
        "human_validation_required": True,
        "limitations": SAFETY_LIMITATIONS,
    }
    return _mark(state, "safety_validation_node")


def audit_log_node(state: Dict[str, Any]) -> Dict[str, Any]:
    write_audit_log({
        "patient_id": state.get("patient_id"),
        "question": state.get("question"),
        "executed_nodes": state.get("executed_nodes", []),
        "sources": state.get("sources", []),
        "risk": state.get("risk", {}),
        "safety": state.get("safety", {}),
    })
    return _mark(state, "audit_log_node")


SEQUENTIAL_NODES = [
    validate_input_node,
    load_patient_node,
    check_pending_exams_node,
    retrieve_protocols_node,
    estimate_risk_node,
    generate_answer_node,
    safety_validation_node,
    audit_log_node,
]
