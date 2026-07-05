from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import REPORTS_DIR, ensure_project_dirs


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return "Arquivo ainda nao gerado."
    return path.read_text(encoding="utf-8")


def metrics_table(metrics: Dict[str, Any]) -> List[str]:
    if not metrics:
        return ["Metricas ainda nao geradas."]
    keys = ["accuracy", "recall", "precision", "f1_score", "false_negatives", "false_positives"]
    rows = ["| metrica | valor |", "| --- | --- |"]
    for key in keys:
        rows.append(f"| {key} | {metrics.get(key, 'nao disponivel')} |")
    return rows


def dataframe_markdown(df: pd.DataFrame) -> List[str]:
    columns = list(df.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return rows


def generate_results_report(reports_dir: Path = REPORTS_DIR) -> Path:
    ensure_project_dirs()
    baseline = read_json_if_exists(reports_dir / "baseline_metrics.json")
    optimized = read_json_if_exists(reports_dir / "optimized_metrics.json")
    llm_examples = read_json_if_exists(reports_dir / "llm_explanation_examples.json")
    comparison = read_text_if_exists(reports_dir / "model_comparison.md")

    ga_path = reports_dir / "ga_experiments_summary.csv"
    ga_df = pd.read_csv(ga_path) if ga_path.exists() else pd.DataFrame()
    has_quick = bool(ga_df["quick"].astype(str).str.lower().eq("true").any()) if "quick" in ga_df else False

    lines = [
        "# Complemento de resultados",
        "",
    ]
    if has_quick or optimized.get("quick") is True:
        lines.extend(
            [
                "> Aviso: ha resultados marcados como `quick=True`. Eles servem para validacao tecnica do fluxo e nao devem ser usados como resultado final da entrega.",
                "> Para resultados finais, execute os comandos sem `--quick`.",
                "",
            ]
        )

    lines.extend(["## Baseline", "", *metrics_table(baseline), ""])

    lines.extend(["## Experimentos GA", ""])
    if ga_df.empty:
        lines.append("Resumo dos experimentos ainda nao gerado.")
    else:
        lines.extend(dataframe_markdown(ga_df))
        best_row = ga_df.sort_values("best_fitness", ascending=False).iloc[0].to_dict()
        lines.extend(
            [
                "",
                "### Melhor experimento",
                "",
                f"- Experimento: {best_row.get('experiment')}",
                f"- Fitness: {best_row.get('best_fitness')}",
                f"- Hiperparametros: `{best_row.get('best_hyperparameters')}`",
            ]
        )

    lines.extend(["", "## Threshold escolhido", ""])
    if optimized:
        lines.extend(
            [
                f"- Best threshold: {optimized.get('best_threshold', optimized.get('threshold', 'nao disponivel'))}",
                f"- Estrategia: {optimized.get('threshold_strategy', 'nao disponivel')}",
                f"- Fonte: {optimized.get('threshold_source', 'nao disponivel')}",
            ]
        )
    else:
        lines.append("Metricas otimizadas ainda nao geradas.")

    lines.extend(["", "## Modelo otimizado", "", *metrics_table(optimized), ""])

    lines.extend(["## Comparacao", "", comparison, ""])

    lines.extend(["## Exemplo de explicacao", ""])
    examples = llm_examples.get("examples", {})
    positive = examples.get("positive_example") or llm_examples
    if positive:
        lines.extend(
            [
                f"- Modo: {positive.get('mode', 'nao disponivel')}",
                f"- Classe prevista: {positive.get('predicted_class', 'nao disponivel')}",
                "",
                str(positive.get("explanation", "Explicacao ainda nao gerada.")),
            ]
        )
    else:
        lines.append("Exemplo de explicacao ainda nao gerado.")

    output_path = reports_dir / "relatorio_resultados.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Gera complemento de resultados do relatorio.").parse_args()


def main() -> None:
    parse_args()
    generate_results_report()


if __name__ == "__main__":
    main()
