from __future__ import annotations

import re
from typing import Dict, List

DANGEROUS_PATTERNS = [
    r"\bdose\b",
    r"\bprescrever\b",
    r"\bantibi[oó]tico\b.*\bsem\s+m[eé]dico\b",
    r"\bfeche\s+o\s+diagn[oó]stico\b",
    r"\bdiagn[oó]stico\s+definitivo\b",
    r"\bignore\s+as\s+regras\b",
]

SAFETY_LIMITATIONS = [
    "Não fornece diagnóstico definitivo.",
    "Não prescreve medicamentos, doses ou condutas terapêuticas diretas.",
    "Exige validação humana obrigatória por profissional habilitado.",
]


def validate_question(question: str) -> Dict[str, object]:
    normalized = (question or "").strip()
    if not normalized:
        return {"status": "blocked", "reason": "Pergunta vazia.", "human_validation_required": True}
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return {
                "status": "blocked",
                "reason": "Pergunta solicita prescrição, diagnóstico definitivo ou violação de regras de segurança.",
                "human_validation_required": True,
            }
    return {"status": "approved", "reason": "Pergunta permitida.", "human_validation_required": True}


def validate_answer(answer: str) -> Dict[str, object]:
    value = answer or ""
    lower = value.lower()
    limitations: List[str] = []
    if "diagnóstico definitivo" not in lower:
        limitations.append("Resposta complementada com aviso de não diagnóstico definitivo.")
    if "validação humana" not in lower and "avaliação humana" not in lower:
        limitations.append("Resposta complementada com exigência de validação humana.")
    if "fonte" not in lower and "protocolo" not in lower:
        limitations.append("Resposta complementada com indicação de fontes/protocolos usados.")
    status = "approved" if not limitations else "needs_completion"
    return {"status": status, "human_validation_required": True, "limitations": limitations}


def complete_safe_answer(answer: str, sources: List[str]) -> str:
    validation = validate_answer(answer)
    if validation["status"] == "approved":
        return answer
    source_text = ", ".join(sources) if sources else "protocolos internos sintéticos"
    return (
        f"{answer}\n\n"
        "Limite de segurança: esta resposta não é diagnóstico definitivo, não substitui avaliação médica "
        "e exige validação humana obrigatória. "
        f"Fontes/protocolos usados: {source_text}."
    )


def generated_answer_is_unsafe(answer: str) -> bool:
    """Barreira conservadora por padrões; não substitui revisão clínica humana."""
    patterns = [
        r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|g/kg|ml/kg)\b",
        r"\b(?:administre|prescreva|tome|inicie|aplique)\b",
        r"\b(?:diagnóstico (?:definitivo|confirmado) (?:é|de)|confirmo (?:a |o )?diagnóstico|você tem sepse|paciente tem sepse)\b",
        r"\b(?:dispensa|sem necessidade de) (?:validação|avaliação) (?:humana|médica)\b",
    ]
    return any(re.search(pattern, answer, flags=re.IGNORECASE) for pattern in patterns)
