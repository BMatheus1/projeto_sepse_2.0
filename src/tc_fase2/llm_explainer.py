from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import REPORTS_DIR, ensure_project_dirs


SAFETY_MESSAGE = (
    "Esta explicação é apenas apoio à decisão clínica, não é diagnóstico definitivo "
    "e não substitui avaliação de uma equipe médica."
)


def _class_to_int(predicted_class: int | str) -> int:
    if isinstance(predicted_class, str):
        normalized = predicted_class.strip().lower()
        if normalized in {"1", "sepse", "risco", "alto", "positivo"}:
            return 1
        if normalized in {"0", "sem sepse", "sem risco", "baixo", "negativo"}:
            return 0
    return int(predicted_class)


def _format_dict(data: Optional[Dict[str, Any]]) -> str:
    if not data:
        return "Não informado."
    lines = []
    for key, value in data.items():
        if value is None or value == "":
            continue
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) if lines else "Não informado."


def _format_list(values: Optional[Iterable[str]]) -> str:
    values = [str(value) for value in values or [] if value]
    return "\n".join(f"- {value}" for value in values) if values else "Não informado."


def build_prompt(
    probability: float,
    predicted_class: int | str,
    clinical_variables: Optional[Dict[str, Any]] = None,
    influencing_factors: Optional[List[str]] = None,
    safety_message: str = SAFETY_MESSAGE,
) -> str:
    class_int = _class_to_int(predicted_class)
    risk_label = "risco elevado de sepse" if class_int == 1 else "sem risco elevado de sepse"
    return f"""Você é um assistente que explica a saída de um modelo preditivo de sepse.

Regras obrigatórias:
- Responda em português claro.
- Use apenas os dados fornecidos.
- Não invente informações ausentes.
- Não afirme diagnóstico definitivo.
- Explique que o resultado é apoio à decisão clínica.
- Destaque os fatores clínicos mais relevantes quando eles forem fornecidos.
- Recomende avaliação médica quando houver risco alto.
- Se a classe prevista for 0, explique que risco elevado não foi identificado pelo modelo, mas isso não exclui avaliação clínica.

Dados do modelo:
- Probabilidade prevista de sepse: {probability:.4f}
- Classe prevista: {predicted_class} ({risk_label})

Variáveis clínicas fornecidas:
{_format_dict(clinical_variables)}

Fatores que influenciaram a decisão:
{_format_list(influencing_factors)}

Mensagem de segurança:
{safety_message}

Gere uma explicação curta, objetiva e segura para uma equipe clínica."""


def local_template_explanation(
    probability: float,
    predicted_class: int | str,
    clinical_variables: Optional[Dict[str, Any]] = None,
    influencing_factors: Optional[List[str]] = None,
    safety_message: str = SAFETY_MESSAGE,
) -> str:
    class_int = _class_to_int(predicted_class)
    risk_text = "risco elevado de sepse" if class_int == 1 else "sem risco elevado de sepse"
    if class_int == 1:
        class_sentence = " Recomenda-se avaliação médica imediata conforme protocolo clínico."
    else:
        class_sentence = (
            " O modelo não identificou risco elevado, mas esse resultado não exclui avaliação clínica "
            "quando houver sinais, sintomas ou julgamento profissional."
        )
    factors = ", ".join(influencing_factors or [])
    factor_sentence = (
        f" Os principais fatores informados associados à decisão foram: {factors}."
        if factors
        else " Não foram fornecidos fatores explicativos adicionais."
    )
    variable_names = ", ".join((clinical_variables or {}).keys())
    variable_sentence = (
        f" As variáveis clínicas consideradas na explicação incluem: {variable_names}."
        if variable_names
        else " Não foram fornecidas variáveis clínicas detalhadas para esta explicação."
    )
    return (
        f"O modelo preditivo estimou probabilidade de sepse de {probability:.1%} e classificou o paciente como "
        f"{risk_text}.{class_sentence}{factor_sentence}{variable_sentence} {safety_message}"
    )


def call_llm(prompt: str, model: str = "gpt-4.1-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não configurada.")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("Pacote openai não instalado. Use o fallback local ou instale openai.") from exc

    client = OpenAI(api_key=api_key)
    response = client.responses.create(model=model, input=prompt)
    text = getattr(response, "output_text", None)
    if text:
        return str(text).strip()
    return str(response).strip()


def generate_explanation(
    probability: float,
    predicted_class: int | str,
    clinical_variables: Optional[Dict[str, Any]] = None,
    influencing_factors: Optional[List[str]] = None,
    use_llm: bool = True,
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    prompt = build_prompt(probability, predicted_class, clinical_variables, influencing_factors)
    mode = "llm" if use_llm else "template_fallback"
    try:
        explanation = call_llm(prompt, model=model) if use_llm else local_template_explanation(
            probability, predicted_class, clinical_variables, influencing_factors
        )
    except Exception:
        mode = "template_fallback"
        explanation = local_template_explanation(probability, predicted_class, clinical_variables, influencing_factors)

    return {
        "mode": mode,
        "probability": probability,
        "predicted_class": _class_to_int(predicted_class),
        "clinical_variables": clinical_variables or {},
        "influencing_factors": influencing_factors or [],
        "prompt": prompt,
        "explanation": explanation,
    }


def write_prompt_documentation() -> None:
    ensure_project_dirs()
    example_prompt = build_prompt(
        probability=0.72,
        predicted_class=1,
        clinical_variables={"MAP": 58, "Lactate": 3.1, "Resp": 28},
        influencing_factors=["MAP baixa", "lactato elevado", "frequência respiratória aumentada"],
    )
    content = [
        "# Prompt usado para explicação com LLM",
        "",
        "O prompt abaixo instrui a LLM a explicar a saída do modelo sem assumir papel diagnóstico.",
        "",
        "```text",
        example_prompt,
        "```",
    ]
    (REPORTS_DIR / "llm_prompt_used.md").write_text("\n".join(content), encoding="utf-8")


def save_example() -> Dict[str, Any]:
    ensure_project_dirs()
    examples = {
        "positive_example": generate_explanation(
            probability=0.72,
            predicted_class=1,
            clinical_variables={"MAP": 58, "Lactate": 3.1, "Resp": 28},
            influencing_factors=["MAP baixa", "lactato elevado", "frequência respiratória aumentada"],
            use_llm=bool(os.getenv("OPENAI_API_KEY")),
        ),
        "negative_example": generate_explanation(
            probability=0.07,
            predicted_class=0,
            clinical_variables={"MAP": 82, "Temp": 36.8, "Resp": 17},
            influencing_factors=["sinais vitais dentro do esperado no exemplo"],
            use_llm=bool(os.getenv("OPENAI_API_KEY")),
        ),
    }
    result = {
        "examples": examples,
        "safety_message": SAFETY_MESSAGE,
    }
    write_prompt_documentation()
    (REPORTS_DIR / "llm_explanation_examples.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera exemplo de explicação em linguagem natural.")
    parser.add_argument("--mock", action="store_true", help="Força o fallback local.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mock:
        os.environ.pop("OPENAI_API_KEY", None)
    save_example()


if __name__ == "__main__":
    main()
