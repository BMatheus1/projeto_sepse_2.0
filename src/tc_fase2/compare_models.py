from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

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


def read_metrics(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de metricas nao encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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

    table_header = "| " + " | ".join(COMPARISON_COLUMNS) + " |"
    table_sep = "| " + " | ".join(["---"] * len(COMPARISON_COLUMNS)) + " |"
    table_rows = [
        "| " + " | ".join(str(row[col]) for col in COMPARISON_COLUMNS) + " |"
        for _, row in comparison.iterrows()
    ]
    md = [
        "# Comparacao entre modelo original e otimizado",
        "",
        table_header,
        table_sep,
        *table_rows,
        "",
        "Em contexto medico, accuracy isolada nao e suficiente para avaliar um modelo de triagem de sepse.",
        "A prioridade clinica deste projeto e aumentar recall e reduzir falsos negativos, pois um falso negativo pode classificar um paciente com risco de sepse como sem risco.",
        "O modelo otimizado deve ser interpretado como ferramenta academica de apoio a decisao clinica, nunca como diagnostico definitivo.",
    ]
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
