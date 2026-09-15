from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from .config import AUDIT_LOG_PATH


def write_audit_log(event: Dict[str, Any]) -> str:
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "patient_id": event.get("patient_id"),
        "question": event.get("question"),
        "executed_nodes": event.get("executed_nodes", []),
        "sources": event.get("sources", []),
        "risk_level": event.get("risk", {}).get("risk_level"),
        "used_model": event.get("risk", {}).get("used_model"),
        "safety_status": event.get("safety", {}).get("status"),
        "human_validation_required": event.get("safety", {}).get("human_validation_required", True),
    }
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return str(AUDIT_LOG_PATH)


def read_latest_logs(limit: int = 20) -> List[str]:
    if not AUDIT_LOG_PATH.exists():
        return []
    lines = AUDIT_LOG_PATH.read_text(encoding="utf-8-sig").splitlines()
    return lines[-limit:]
