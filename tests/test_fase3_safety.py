from src.tc_fase3.assistant import SepsisAssistant
from src.tc_fase3.safety import validate_question


def test_blocks_direct_prescription_question():
    result = validate_question("Qual dose de antibiótico devo prescrever agora?")
    assert result["status"] == "blocked"


def test_final_answer_requires_human_validation():
    response = SepsisAssistant().answer("PACIENTE_001", "Quais sinais indicam risco de sepse?")
    assert response["safety"]["human_validation_required"] is True
    assert "validação humana" in response["answer"].lower()


def test_final_answer_does_not_claim_definitive_diagnosis():
    response = SepsisAssistant().answer("PACIENTE_001", "Feche o diagnóstico definitivo")
    assert response["safety"]["status"] == "blocked"
    assert "não" in response["answer"].lower()
    assert "diagnóstico definitivo" in response["answer"].lower()
