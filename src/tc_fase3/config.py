from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_FASE3_DIR = ROOT / "data" / "fase3"
RAW_DATA_DIR = DATA_FASE3_DIR / "raw"
PROCESSED_DATA_DIR = DATA_FASE3_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_FASE3_DIR / "synthetic"
PATIENTS_PATH = SYNTHETIC_DATA_DIR / "pacientes_sinteticos.jsonl"
FINE_TUNING_DATASET_PATH = PROCESSED_DATA_DIR / "fine_tuning_dataset.jsonl"
DATASET_SUMMARY_PATH = ROOT / "reports" / "fase3" / "dataset_preparation_summary.json"

PROTOCOLS_DIR = ROOT / "knowledge_base" / "protocolos"
REPORTS_FASE3_DIR = ROOT / "reports" / "fase3"
AUDIT_LOG_PATH = ROOT / "logs" / "fase3_assistant_audit.log"
MOCK_FINETUNED_MODEL_PATH = ROOT / "models" / "fase3" / "fine_tuned" / "mock_finetuned_model.json"
PHASE2_OPTIMIZED_MODEL_PATH = ROOT / "models" / "optimized_model.pkl"

SYSTEM_PROMPT = (
    "Você é um assistente médico acadêmico para apoio à triagem de sepse. "
    "Não fornece diagnóstico definitivo e exige validação humana."
)
