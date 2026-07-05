from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import REPORTS_DIR, ensure_project_dirs


COMPARISON_COLUMNS = [
    "model",
    "accuracy",
    "recall",
    "precision",
    "f1_score",
    "false_negatives",
    "false_positives",
]

DELTA_COLUMNS = ["recall", "precision", "f1_score", "false_negatives", "false_positives"]


def read_metrics(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de metricas nao encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def is_quick(metrics: Dict[str, Any]) -> bool:
    value = metrics.get("quick", False)
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def markdown_table(df: pd.DataFrame) -> List[str]:
    columns = list(df.columns)
    table_header = "| " + " | ".join(columns) + " |"
    table_sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    table_rows = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for _, row in df.iterrows()]
    return [table_header, table_sep, *table_rows]


def build_analysis(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> List[str]:
    lines = []
    fn_delta = optimized.get("false_negatives", 0) - baseline.get("false_negatives", 0)
    fp_delta = optimized.get("false_positives", 0) - baseline.get("false_positives", 0)
    precision_delta = optimized.get("precision", 0.0) - baseline.get("precision", 0.0)

    if fn_delta < 0:
        lines.append(f"- Falsos negativos cairam em {abs(fn_delta)}, indicando ganho de sensibilidade.")
    elif fn_delta > 0:
        lines.append(f"- Falsos negativos subiram em {fn_delta}; isso exige revisao porque FN sao criticos em sepse.")
    else:
        lines.append("- Falsos negativos ficaram estaveis.")

    if fp_delta > 0:
        lines.append(f"- Falsos positivos subiram em {fp_delta}, gerando mais alertas falsos e possivel custo operacional.")
    elif fp_delta < 0:
        lines.append(f"- Falsos positivos cairam em {abs(fp_delta)}, melhorando a especificidade operacional.")
    else:
        lines.append("- Falsos positivos ficaram estaveis.")

    if precision_delta < 0:
        lines.append("- Precision caiu; isso evidencia o trade-off de ampliar sensibilidade aceitando mais alertas falsos.")
    elif precision_delta > 0:
        lines.append("- Precision subiu, sugerindo melhor equilibrio entre alertas corretos e falsos positivos.")

    lines.append("- Em sepse, recall e reducao de falsos negativos sao mais criticos do que accuracy isolada.")
    lines.append("- Este modelo e academico e deve ser usado apenas como apoio, nunca como diagnostico definitivo.")
    return lines


def compare_models(
    baseline_path: Path = REPORTS_DIR / "baseline_metrics.json",
    optimized_path: Path = REPORTS_DIR / "optimized_metrics.json",
) -> pd.DataFrame:
    ensure_project_dirs()
    baseline = read_metrics(baseline_path)
    optimized = read_metrics(optimized_path)
    rows = [
        {"model": "baseline_original", **baseline},
        {"model": "modelo_otimizado_ga", **optimized},
    ]
    df = pd.DataFrame(rows)
    comparison = df[COMPARISON_COLUMNS]
    csv_path = REPORTS_DIR / "model_comparison.csv"
    comparison.to_csv(csv_path, index=False)

    delta_rows = []
    for column in DELTA_COLUMNS:
        baseline_value = baseline.get(column)
        optimized_value = optimized.get(column)
        delta_rows.append(
            {
                "metric": column,
                "baseline": baseline_value,
                "optimized": optimized_value,
                "absolute_difference": optimized_value - baseline_value,
            }
        )
    deltas = pd.DataFrame(delta_rows)
    quick_warning = is_quick(baseline) or is_quick(optimized)

    md = [
        "# Comparacao entre modelo original e otimizado",
        "",
        "## Metricas",
        "",
        *markdown_table(comparison),
        "",
        "## Diferencas absolutas",
        "",
        *markdown_table(deltas),
        "",
        "## Analise automatica",
        "",
        *build_analysis(baseline, optimized),
        "",
    ]
    if quick_warning:
        md.extend(
            [
                "## Aviso sobre modo quick",
                "",
                "Pelo menos um arquivo de metricas foi gerado com `quick=True`. Esses numeros servem para validacao tecnica do fluxo e nao devem ser usados como resultado final da entrega.",
                "Para gerar resultados finais, execute os comandos sem `--quick`.",
                "",
            ]
        )
    md.extend([
        "Em contexto medico, accuracy isolada nao e suficiente para avaliar um modelo de triagem de sepse.",
        "A prioridade clinica deste projeto e aumentar recall e reduzir falsos negativos, pois um falso negativo pode classificar um paciente com risco de sepse como sem risco.",
        "O modelo otimizado deve ser interpretado como ferramenta academica de apoio a decisao clinica, nunca como diagnostico definitivo.",
    ])
    (REPORTS_DIR / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara baseline e modelo otimizado.")
    parser.add_argument("--baseline", type=Path, default=REPORTS_DIR / "baseline_metrics.json")
    parser.add_argument("--optimized", type=Path, default=REPORTS_DIR / "optimized_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_models(args.baseline, args.optimized)


if __name__ == "__main__":
    main()
