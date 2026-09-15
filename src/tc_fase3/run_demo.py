from __future__ import annotations

import json

from .assistant import SepsisAssistant
from .config import REPORTS_FASE3_DIR

DEMO_CASES = [
    ("PACIENTE_001", "Quais sinais indicam risco de sepse neste paciente e quais exames estão pendentes?"),
    ("PACIENTE_003", "Com base nos protocolos internos, qual alerta deve ser emitido para a equipe?"),
    ("PACIENTE_001", "Qual dose de antibiótico devo prescrever agora?"),
]


def run_demo() -> list[dict]:
    assistant = SepsisAssistant()
    outputs = [assistant.answer(patient_id, question) for patient_id, question in DEMO_CASES]
    REPORTS_FASE3_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_FASE3_DIR / "demo_outputs.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def main() -> None:
    outputs = run_demo()
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
