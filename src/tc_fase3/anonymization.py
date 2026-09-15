from __future__ import annotations

import re

CPF_PATTERN = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\(?\d{2}\)?\s?)?(?:9\s?)?\d{4}[-\s]?\d{4}(?!\d)")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
NAME_PATTERNS = [
    re.compile(r"\bNome\s*:\s*[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç]+){0,3}", re.IGNORECASE),
    re.compile(r"\bPaciente\s+(?:João|Maria|José|Ana|Carlos|Mariana|Pedro|Paula|Roberto|Fernanda)\b", re.IGNORECASE),
]


def anonymize_text(text: str) -> str:
    """Remove identificadores pessoais simulados de textos clínicos sintéticos."""
    if text is None:
        return ""
    value = str(text)
    value = CPF_PATTERN.sub("[CPF_REMOVIDO]", value)
    value = EMAIL_PATTERN.sub("[EMAIL_REMOVIDO]", value)
    value = PHONE_PATTERN.sub("[TELEFONE_REMOVIDO]", value)
    for pattern in NAME_PATTERNS:
        value = pattern.sub("[NOME_REMOVIDO]", value)
    return value
