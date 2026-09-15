import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for path in [ROOT, SRC]:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


import pytest


@pytest.fixture(autouse=True)
def fase3_offline_models(monkeypatch, tmp_path):
    """Mesmo com adapter instalado localmente, pytest não baixa/carrega pesos."""
    from src.tc_fase3 import audit_logger
    monkeypatch.setattr(audit_logger, "AUDIT_LOG_PATH", tmp_path / "test_audit.log")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("FASE3_ADAPTER_PATH", str(tmp_path / "no_adapter"))
