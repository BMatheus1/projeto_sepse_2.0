from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .config import FINE_TUNING_DATASET_PATH, MOCK_FINETUNED_MODEL_PATH

OPTIONAL_REAL_DEPENDENCIES = ["transformers", "datasets", "peft", "trl"]


def load_dataset(path: Path = FINE_TUNING_DATASET_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset de fine-tuning não encontrado: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if "messages" not in item or len(item["messages"]) < 3:
                    raise ValueError("Registro sem estrutura conversacional válida em messages.")
                rows.append(item)
    if not rows:
        raise ValueError("Dataset de fine-tuning está vazio.")
    return rows


def run_mock_finetuning(dataset_path: Path = FINE_TUNING_DATASET_PATH) -> Dict[str, Any]:
    rows = load_dataset(dataset_path)
    artifact = {
        "base_model": "modelo-base-simulado-llm-medico-academico",
        "examples": len(rows),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "status": "mock_finetuning_completed",
        "note": "Simulação acadêmica reproduzível; nenhum modelo pesado foi treinado.",
    }
    MOCK_FINETUNED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MOCK_FINETUNED_MODEL_PATH.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact


def check_real_dependencies() -> Dict[str, Any]:
    missing = []
    for package in OPTIONAL_REAL_DEPENDENCIES:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        return {
            "status": "real_finetuning_unavailable",
            "missing_dependencies": missing,
            "message": "O modo real exige instalação de dependências opcionais e ambiente adequado. Nenhum download ou treino pesado foi executado.",
        }
    return {
        "status": "real_finetuning_ready_but_not_executed",
        "message": "Dependências disponíveis. Estrutura pronta para fine-tuning real futuro; execução pesada não foi iniciada automaticamente.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning acadêmico da Fase 3.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true", help="Executa simulação leve e local.")
    group.add_argument("--real", action="store_true", help="Verifica pré-requisitos para fine-tuning real opcional.")
    args = parser.parse_args()

    result = run_mock_finetuning() if args.mock else check_real_dependencies()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
