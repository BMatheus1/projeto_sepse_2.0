from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .config import PATIENTS_PATH


class PatientRepository:
    def __init__(self, path: Path = PATIENTS_PATH) -> None:
        self.path = path
        self._patients = self._load_patients()

    def _load_patients(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(f"Base de pacientes sintéticos não encontrada: {self.path}")
        patients: Dict[str, Dict[str, Any]] = {}
        with self.path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                patient = json.loads(line)
                patients[str(patient["patient_id"])] = patient
        return patients

    def get_patient(self, patient_id: str) -> Dict[str, Any]:
        try:
            return self._patients[patient_id]
        except KeyError as exc:
            raise KeyError(f"Paciente sintético não encontrado: {patient_id}") from exc

    def list_patients(self) -> List[Dict[str, Any]]:
        return list(self._patients.values())

    def count(self) -> int:
        return len(self._patients)
