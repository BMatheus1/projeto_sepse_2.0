from __future__ import annotations

from typing import Any, Dict

from .config import REPORTS_FASE3_DIR
from .graph_nodes import SEQUENTIAL_NODES
from .graph_state import SepsisGraphState

FLOW_DESCRIPTION = """# Fluxo LangGraph - Assistente de Sepse Fase 3

```mermaid
flowchart TD
    A[Entrada] --> B[Validação de segurança]
    B --> C[Consulta ao paciente sintético]
    C --> D[Verificação de exames pendentes]
    D --> E[Busca em protocolos internos]
    E --> F[Estimativa de risco com modelo Fase 2 ou fallback]
    F --> G[Geração da resposta]
    G --> H[Validação final de segurança]
    H --> I[Log de auditoria]
    I --> J[Resposta final com fontes]
```

Quando `langgraph` está disponível, a estrutura pode ser compilada como grafo. Em ambiente local sem essa dependência, o projeto usa fallback sequencial com os mesmos nós e a mesma ordem de execução.
"""


def save_flow_description() -> None:
    REPORTS_FASE3_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_FASE3_DIR / "langgraph_flow.md").write_text(FLOW_DESCRIPTION, encoding="utf-8")


def _to_response(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "patient_id": state["patient_id"],
        "question": state["question"],
        "answer": state.get("final_answer", state.get("draft_answer", "")),
        "risk": state.get("risk", {}),
        "sources": state.get("sources", []),
        "safety": state.get("safety", {}),
        "executed_nodes": state.get("executed_nodes", []),
    }


def _run_sequential(initial_state: SepsisGraphState) -> Dict[str, Any]:
    state: Dict[str, Any] = dict(initial_state)
    for node in SEQUENTIAL_NODES:
        state = node(state)
    return _to_response(state)


def _run_langgraph(initial_state: SepsisGraphState) -> Dict[str, Any]:
    from langgraph.graph import END, StateGraph

    from .graph_nodes import (
        audit_log_node,
        check_pending_exams_node,
        estimate_risk_node,
        generate_answer_node,
        load_patient_node,
        retrieve_protocols_node,
        safety_validation_node,
        validate_input_node,
    )

    builder = StateGraph(SepsisGraphState)
    builder.add_node("validate_input", validate_input_node)
    builder.add_node("load_patient", load_patient_node)
    builder.add_node("check_pending_exams", check_pending_exams_node)
    builder.add_node("retrieve_protocols", retrieve_protocols_node)
    builder.add_node("estimate_risk", estimate_risk_node)
    builder.add_node("generate_answer", generate_answer_node)
    builder.add_node("safety_validation", safety_validation_node)
    builder.add_node("audit_log", audit_log_node)
    builder.set_entry_point("validate_input")
    builder.add_edge("validate_input", "load_patient")
    builder.add_edge("load_patient", "check_pending_exams")
    builder.add_edge("check_pending_exams", "retrieve_protocols")
    builder.add_edge("retrieve_protocols", "estimate_risk")
    builder.add_edge("estimate_risk", "generate_answer")
    builder.add_edge("generate_answer", "safety_validation")
    builder.add_edge("safety_validation", "audit_log")
    builder.add_edge("audit_log", END)
    app = builder.compile()
    return _to_response(app.invoke(initial_state))


def run_sepsis_flow(patient_id: str, question: str) -> Dict[str, Any]:
    save_flow_description()
    initial_state: SepsisGraphState = {"patient_id": patient_id, "question": question, "executed_nodes": []}
    try:
        return _run_langgraph(initial_state)
    except ImportError:
        return _run_sequential(initial_state)
    except Exception:
        return _run_sequential(initial_state)
