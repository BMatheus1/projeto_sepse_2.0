from src.tc_fase3.langgraph_flow import run_sepsis_flow


def test_run_sepsis_flow_returns_response():
    response = run_sepsis_flow("PACIENTE_001", "Quais sinais indicam risco de sepse?")
    assert response["patient_id"] == "PACIENTE_001"
    assert response["answer"]
    assert response["sources"]


def test_run_sepsis_flow_contains_main_steps():
    response = run_sepsis_flow("PACIENTE_001", "Quais exames estão pendentes?")
    nodes = response["executed_nodes"]
    assert "validate_input_node" in nodes
    assert "load_patient_node" in nodes
    assert "retrieve_protocols_node" in nodes
    assert "estimate_risk_node" in nodes
    assert "safety_validation_node" in nodes
