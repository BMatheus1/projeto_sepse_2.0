from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import REPORTS_DIR, ensure_project_dirs


SAFETY_MESSAGE = (
    "Esta explicacao e apenas apoio a decisao clinica e nao substitui avaliacao de uma equipe medica."
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
        return "Nao informado."
    lines = []
    for key, value in data.items():
        if value is None or value == "":
            continue
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) if lines else "Nao informado."


def _format_list(values: Optional[Iterable[str]]) -> str:
    values = [str(value) for value in values or [] if value]
    return "\n".join(f"- {value}" for value in values) if values else "Nao informado."


def build_prompt(
    probability: float,
    predicted_class: int | str,
    clinical_variables: Optional[Dict[str, Any]] = None,
    influencing_factors: Optional[List[str]] = None,
    safety_message: str = SAFETY_MESSAGE,
) -> str:
    class_int = _class_to_int(predicted_class)
    risk_label = "risco elevado de sepse" if class_int == 1 else "sem risco elevado de sepse"
    return f"""Voce e um assistente que explica a saida de um modelo preditivo de sepse.

Regras obrigatorias:
- Responda em portugues claro.
- Use apenas os dados fornecidos.
- Nao invente informacoes ausentes.
- Nao afirme diagnostico definitivo.
- Explique que o resultado e apoio a decisao clinica.
- Destaque os fatores clinicos mais relevantes quando eles forem fornecidos.
- Recomende avaliacao medica quando houver risco alto.

Dados do modelo:
- Probabilidade prevista de sepse: {probability:.4f}
- Classe prevista: {predicted_class} ({risk_label})

Variaveis clinicas fornecidas:
{_format_dict(clinical_variables)}

Fatores que influenciaram a decisao:
{_format_list(influencing_factors)}

Mensagem de seguranca:
{safety_message}

Gere uma explicacao curta, objetiva e segura para uma equipe clinica."""


def local_template_explanation(
    probability: float,
    predicted_class: int | str,
    clinical_variables: Optional[Dict[str, Any]] = None,
    influencing_factors: Optional[List[str]] = None,
    safety_message: str = SAFETY_MESSAGE,
) -> str:
    class_int = _class_to_int(predicted_class)
    risk_text = "risco elevado de sepse" if class_int == 1 else "sem risco elevado de sepse"
    factors = ", ".join(influencing_factors or [])
    factor_sentence = (
        f" Os principais fatores informados associados a decisao foram: {factors}."
        if factors
        else " Nao foram fornecidos fatores explicativos adicionais."
    )
    variable_names = ", ".join((clinical_variables or {}).keys())
    variable_sentence = (
        f" As variaveis clinicas consideradas na explicacao incluem: {variable_names}."
        if variable_names
        else " Nao foram fornecidas variaveis clinicas detalhadas para esta explicacao."
    )
    return (
        f"O modelo preditivo estimou probabilidade de sepse de {probability:.1%} e classificou o paciente como "
        f"{risk_text}.{factor_sentence}{variable_sentence} {safety_message}"
    )


def call_llm(prompt: str, model: str = "gpt-4.1-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY nao configurada.")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("Pacote openai nao instalado. Use o fallback local ou instale openai.") from exc

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
    mode = "llm"
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
        influencing_factors=["MAP baixa", "lactato elevado", "frequencia respiratoria aumentada"],
    )
    content = [
        "# Prompt usado para explicacao com LLM",
        "",
        "O prompt abaixo instrui a LLM a explicar a saida do modelo sem assumir papel diagnostico.",
        "",
        "```text",
        example_prompt,
        "```",
    ]
    (REPORTS_DIR / "llm_prompt_used.md").write_text("\n".join(content), encoding="utf-8")


def save_example() -> Dict[str, Any]:
    ensure_project_dirs()
    result = generate_explanation(
        probability=0.72,
        predicted_class=1,
        clinical_variables={"MAP": 58, "Lactate": 3.1, "Resp": 28},
        influencing_factors=["MAP baixa", "lactato elevado", "frequencia respiratoria aumentada"],
        use_llm=bool(os.getenv("OPENAI_API_KEY")),
    )
    write_prompt_documentation()
    (REPORTS_DIR / "llm_explanation_examples.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera exemplo de explicacao em linguagem natural.")
    parser.add_argument("--mock", action="store_true", help="Forca o fallback local.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mock:
        os.environ.pop("OPENAI_API_KEY", None)
    save_example()


if __name__ == "__main__":
    main()
