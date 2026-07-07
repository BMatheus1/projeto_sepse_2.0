from src.tc_fase2.llm_explainer import build_prompt, generate_explanation, local_template_explanation, save_example


def test_prompt_contains_required_data():
    prompt = build_prompt(
        probability=0.81,
        predicted_class=1,
        clinical_variables={"MAP": 55, "Lactate": 3.5},
        influencing_factors=["MAP baixa", "lactato elevado"],
    )
    assert "0.8100" in prompt
    assert "MAP" in prompt
    assert "Lactate" in prompt
    assert "Não invente" in prompt
    assert "apoio à decisão clínica" in prompt


def test_local_fallback_works_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = generate_explanation(
        probability=0.66,
        predicted_class=1,
        clinical_variables={"Resp": 30},
        influencing_factors=["frequência respiratória aumentada"],
        use_llm=True,
    )
    assert result["mode"] == "template_fallback"
    assert "não substitui avaliação" in result["explanation"]


def test_template_does_not_invent_missing_factors():
    text = local_template_explanation(probability=0.2, predicted_class=0)
    assert "Não foram fornecidos fatores" in text
    assert "não exclui avaliação clínica" in text


def test_save_example_generates_positive_and_negative(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = save_example()
    assert "positive_example" in result["examples"]
    assert "negative_example" in result["examples"]
