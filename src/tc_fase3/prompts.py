from __future__ import annotations

from typing import Any, Dict, List

SYSTEM_PROMPT_ASSISTANT = """
Você é um assistente médico acadêmico para apoio à triagem de sepse.
Responda em português claro, use apenas dados fornecidos, indique fontes e não invente informações.
Não prescreva diretamente, não afirme diagnóstico definitivo e recomende validação humana obrigatória.
Diferencie dado do paciente, protocolo interno sintético e inferência do assistente.
""".strip()


def build_context(patient: Dict[str, Any], protocols: List[Dict[str, Any]], risk: Dict[str, Any]) -> str:
    sources = ", ".join(item["source"] for item in protocols) or "protocolos sintéticos não localizados"
    return (
        f"Paciente: {patient.get('patient_id')}; queixa: {patient.get('queixa_principal')}; "
        f"risco: {risk.get('risk_level')} ({risk.get('used_model')}); "
        f"fontes: {sources}."
    )
