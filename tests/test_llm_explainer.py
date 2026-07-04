from src.tc_fase2.llm_explainer import build_prompt, generate_explanation, local_template_explanation


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
    assert "Nao invente" in prompt
    assert "apoio a decisao clinica" in prompt


def test_local_fallback_works_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = generate_explanation(
        probability=0.66,
        predicted_class=1,
        clinical_variables={"Resp": 30},
        influencing_factors=["frequencia respiratoria aumentada"],
        use_llm=True,
    )
    assert result["mode"] == "template_fallback"
    assert "nao substitui avaliacao" in result["explanation"]


def test_template_does_not_invent_missing_factors():
    text = local_template_explanation(probability=0.2, predicted_class=0)
    assert "Nao foram fornecidos fatores" in text
