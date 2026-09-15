from fastapi.testclient import TestClient

from src.tc_fase3.api import app


client = TestClient(app)


def test_fase3_health_endpoint():
    response = client.get("/fase3/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["patients_loaded"] >= 10
    assert payload["protocols_loaded"] >= 4


def test_fase3_assistant_ask_endpoint():
    response = client.post(
        "/fase3/assistant/ask",
        json={"patient_id": "PACIENTE_001", "question": "Quais sinais indicam risco de sepse neste paciente?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["sources"]
    assert payload["safety"]["human_validation_required"] is True
