from src.tc_fase3.assistant import SepsisAssistant


def test_assistant_answer_for_patient_001():
    response = SepsisAssistant().answer("PACIENTE_001", "Quais sinais indicam risco de sepse neste paciente?")
    assert response["patient_id"] == "PACIENTE_001"
    assert response["answer"]
    assert response["sources"]
    assert response["safety"]["human_validation_required"] is True
    assert response["risk"]["risk_level"] in ["baixo", "moderado", "alto"]


def test_assistant_returns_expected_keys():
    response = SepsisAssistant().answer("PACIENTE_001", "Quais exames estão pendentes?")
    for key in ["answer", "sources", "safety", "risk"]:
        assert key in response
