import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient

from src.tc_fase2.predict_and_explain import predict_and_explain


def load_api_module():
    path = Path(__file__).resolve().parents[1] / "__main__.py"
    spec = importlib.util.spec_from_file_location("api_main_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_predict_and_explain_works_with_local_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = predict_and_explain(use_llm=False)
    assert "probability" in result
    assert result["explanation_mode"] == "template_fallback"
    assert result["predicted_class"] in [0, 1]


def test_api_predict_and_explain_endpoints_work_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    api_module = load_api_module()
    client = TestClient(api_module.app)
    features = {feature: api_module.medianas.get(feature, 0) for feature in api_module.features_esperadas}
    features.update({"MAP": 58, "Lactate": 3.1, "Resp": 28, "Temp": 38.4, "WBC": 16, "HR": 112})

    predict_response = client.post("/predict", json={"features": features})
    assert predict_response.status_code == 200
    assert "probabilidade_sepse" in predict_response.json()

    explain_response = client.post("/predict/explain", json={"features": features})
    assert explain_response.status_code == 200
    payload = explain_response.json()
    assert "explicacao" in payload
    assert "modo_explicacao" in payload
