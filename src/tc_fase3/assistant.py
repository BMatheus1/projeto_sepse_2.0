from __future__ import annotations

from typing import Any, Dict, List

from .audit_logger import write_audit_log
from .patient_repository import PatientRepository
from .phase2_risk_tool import estimate_sepsis_risk
from .protocol_retriever import ProtocolRetriever
from .safety import SAFETY_LIMITATIONS, complete_safe_answer, validate_question

PATIENT_SOURCE = "pacientes_sinteticos.jsonl"


class SepsisAssistant:
    def __init__(self, repository: PatientRepository | None = None, retriever: ProtocolRetriever | None = None) -> None:
        self.repository = repository or PatientRepository()
        self.retriever = retriever or ProtocolRetriever()

    def _build_answer(
        self,
        patient: Dict[str, Any],
        question: str,
        risk: Dict[str, Any],
        protocols: List[Dict[str, Any]],
        question_status: Dict[str, Any],
    ) -> str:
        sources = list(dict.fromkeys([item["source"] for item in protocols] + [PATIENT_SOURCE]))
        if question_status["status"] == "blocked":
            return complete_safe_answer(
                "Não posso prescrever dose, fechar diagnóstico definitivo ou orientar conduta sem médico responsável. "
                "Posso apoiar a triagem organizando sinais de risco, exames pendentes e fontes sintéticas para revisão da equipe. "
                f"Neste caso, fatores observados: {', '.join(risk['factors']) or 'sem fatores críticos claros'}. "
                f"Exames pendentes registrados: {', '.join(patient.get('exames_pendentes', [])) or 'não informados'}. "
                "A validação humana obrigatória deve ser realizada pela equipe assistencial. "
                f"Fontes/protocolos usados: {', '.join(sources)}.",
                sources,
            )

        protocol_sources = ", ".join(item["source"] for item in protocols) or "protocolos sintéticos não encontrados"
        factors = ", ".join(risk["factors"]) if risk["factors"] else "sem fatores de alto risco pela regra local"
        pending = ", ".join(patient.get("exames_pendentes", [])) or "não há exames pendentes registrados"
        answer = (
            f"Dados do paciente sintético {patient['patient_id']}: queixa principal de {patient.get('queixa_principal')}, "
            f"sinais vitais {patient.get('sinais_vitais')} e exames {patient.get('exames')}. "
            f"Inferência do assistente: o nível de risco estimado é {risk['risk_level']} "
            f"(score {risk['risk_score']}, método {risk['used_model']}). "
            f"Fatores de risco identificados: {factors}. "
            f"Exames pendentes relevantes: {pending}. "
            f"Protocolos internos sintéticos consultados: {protocol_sources}. "
            "Esta resposta não é diagnóstico definitivo, não prescreve condutas e exige validação humana obrigatória."
        )
        return complete_safe_answer(answer, sources)

    def answer(self, patient_id: str, question: str) -> Dict[str, Any]:
        executed_nodes = ["validate_question"]
        question_status = validate_question(question)

        patient = self.repository.get_patient(patient_id)
        executed_nodes.append("load_patient")

        query = f"{question} {patient.get('queixa_principal', '')} sepse alerta exames validação"
        protocols = self.retriever.search(query, limit=3)
        if question_status["status"] == "blocked":
            limit_protocols = [doc for doc in self.retriever.documents if doc.source == "protocolo_limites_assistente.md"]
            if limit_protocols:
                protocols = [{"source": limit_protocols[0].source, "content": limit_protocols[0].content, "score": 99}] + protocols[:2]
        executed_nodes.append("retrieve_protocols")

        risk = estimate_sepsis_risk(patient)
        executed_nodes.append("estimate_risk")

        answer = self._build_answer(patient, question, risk, protocols, question_status)
        executed_nodes.append("generate_answer")

        sources = list(dict.fromkeys([item["source"] for item in protocols] + [PATIENT_SOURCE]))
        safety = {
            "status": question_status["status"],
            "human_validation_required": True,
            "limitations": SAFETY_LIMITATIONS,
        }
        executed_nodes.append("safety_validation")

        result = {
            "patient_id": patient_id,
            "question": question,
            "answer": answer,
            "risk": risk,
            "sources": sources,
            "safety": safety,
            "executed_nodes": executed_nodes,
        }
        write_audit_log(result)
        result["executed_nodes"].append("audit_log")
        return result

