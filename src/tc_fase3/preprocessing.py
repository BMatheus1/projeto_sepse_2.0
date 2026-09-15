from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

REQUIRED_RAW_FIELDS = ["source", "input", "output"]


def clean_text(text: str) -> str:
    if text is None:
        raise ValueError("Texto vazio ou ausente.")
    value = re.sub(r"\s+", " ", str(text)).strip()
    if not value:
        raise ValueError("Texto vazio ou ausente.")
    return value


def validate_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    for field in REQUIRED_RAW_FIELDS:
        if field not in record:
            errors.append(f"Campo obrigatório ausente: {field}")
            continue
        try:
            clean_text(str(record[field]))
        except ValueError:
            errors.append(f"Campo obrigatório vazio: {field}")
    return not errors, errors
