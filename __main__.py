from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn


MODEL_PATH = Path("modelos_salvos/modelo_sepse_sem_tempo_admin.pkl")
FEATURES_PATH = Path("data/processed/features_modelo_sem_tempo_admin.csv")
MEDIANAS_PATH = Path("data/processed/medianas_treino_sem_tempo_admin.csv")
DEFAULT_THRESHOLD = 0.12

app = FastAPI(
    title="API de Detecção de Sepse",
    description="API para inferência de risco de sepse.",
    version="1.3.0",
)


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Dicionário com as features de entrada do paciente.",
    )
    threshold: float = Field(
        DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Threshold de decisão para classificar sepse.",
    )


class PredictResponse(BaseModel):
    probabilidade_sepse: float
    threshold_utilizado: float
    predicao: int
    classe_predita: str


def carregar_artefato():
    if not MODEL_PATH.exists():
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        print(f"Erro ao carregar artefato: {exc}")
        return None


def extrair_modelo(artefato):
    if artefato is None:
        return None

    # Caso o artefato já seja o próprio modelo
    if hasattr(artefato, "predict_proba") or hasattr(artefato, "predict"):
        return artefato

    # Caso seja um dicionário com o modelo dentro
    if isinstance(artefato, dict):
        for chave in [
            "model",
            "modelo",
            "clf",
            "classifier",
            "pipeline",
            "best_model",
            "melhor_modelo",
            "modelo_final",
        ]:
            if chave in artefato:
                candidato = artefato[chave]
                if hasattr(candidato, "predict_proba") or hasattr(candidato, "predict"):
                    return candidato

    return None


def carregar_features_esperadas(modelo) -> List[str]:
    """
    Carrega as features esperadas priorizando exatamente as feature_names
    internas do modelo XGBoost.
    """
    # 1) Tenta pegar do booster do XGBoost
    try:
        if modelo is not None and hasattr(modelo, "get_booster"):
            booster = modelo.get_booster()
            if booster is not None and booster.feature_names:
                return [str(col) for col in booster.feature_names]
    except Exception as exc:
        print(f"Não consegui ler feature_names do booster: {exc}")

    # 2) Tenta feature_names_in_
    try:
        if modelo is not None and hasattr(modelo, "feature_names_in_"):
            return [str(col) for col in modelo.feature_names_in_]
    except Exception as exc:
        print(f"Não consegui ler feature_names_in_: {exc}")

    # 3) Fallback para o CSV
    if FEATURES_PATH.exists():
        try:
            df = pd.read_csv(FEATURES_PATH)

            if "feature" in df.columns:
                return df["feature"].dropna().astype(str).tolist()

            return df.iloc[:, 0].dropna().astype(str).tolist()
        except Exception as exc:
            print(f"Erro ao carregar features do CSV: {exc}")

    return []


def carregar_medianas() -> Dict[str, float]:
    if not MEDIANAS_PATH.exists():
        return {}

    try:
        df = pd.read_csv(MEDIANAS_PATH)

        if {"coluna", "mediana"}.issubset(df.columns):
            return dict(zip(df["coluna"].astype(str), df["mediana"]))

        if {"feature", "mediana"}.issubset(df.columns):
            return dict(zip(df["feature"].astype(str), df["mediana"]))

        if df.shape[1] >= 2:
            return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1]))

    except Exception as exc:
        print(f"Erro ao carregar medianas: {exc}")

    return {}


def preparar_entrada(
    features_recebidas: Dict[str, Any],
    features_esperadas: List[str],
    medianas: Dict[str, float],
) -> pd.DataFrame:
    """
    Monta a entrada final no formato exato esperado pelo modelo.
    """
    entrada = pd.DataFrame([features_recebidas])

    # Garante colunas faltantes
    for col in features_esperadas:
        if col not in entrada.columns:
            entrada[col] = medianas.get(col, 0)

    # Mantém apenas as colunas esperadas, na ordem correta
    entrada = entrada.reindex(columns=features_esperadas)

    # Força valores numéricos e preenche faltantes
    entrada = entrada.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Garante nomes de colunas como string
    entrada.columns = entrada.columns.astype(str)

    return entrada


def prever_probabilidade(modelo, entrada: pd.DataFrame) -> float:
    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(entrada)
        return float(proba[0, 1])

    if hasattr(modelo, "predict"):
        pred = modelo.predict(entrada)
        return float(pred[0])

    raise ValueError("Modelo sem predict_proba() e sem predict().")


artefato = carregar_artefato()
modelo = extrair_modelo(artefato)
features_esperadas = carregar_features_esperadas(modelo)
medianas = carregar_medianas()


@app.get("/")
def root():
    return {
        "mensagem": "API de detecção de sepse ativa.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status_api": "ok",
        "artefato_carregado": artefato is not None,
        "modelo_extraido": modelo is not None,
        "tipo_artefato": str(type(artefato)) if artefato is not None else None,
        "tipo_modelo": str(type(modelo)) if modelo is not None else None,
        "features_carregadas": len(features_esperadas),
        "medianas_carregadas": len(medianas),
        "caminho_modelo": str(MODEL_PATH),
    }


@app.get("/metadata")
def metadata():
    return {
        "nome_projeto": "Detecção de Sepse com Machine Learning",
        "versao_api": "1.3.0",
        "threshold_padrao": DEFAULT_THRESHOLD,
        "modelo_extraido": modelo is not None,
        "features_carregadas": len(features_esperadas),
        "medianas_carregadas": len(medianas),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if modelo is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo não extraído do artefato.",
        )

    if not features_esperadas:
        raise HTTPException(
            status_code=500,
            detail="Lista de features não carregada.",
        )

    try:
        entrada = preparar_entrada(
            payload.features,
            features_esperadas,
            medianas,
        )

        # Prints de diagnóstico para o terminal
        print("Features esperadas:", len(features_esperadas))
        print("Colunas da entrada:", len(entrada.columns))
        print("Primeiras 20 colunas da entrada:", entrada.columns[:20].tolist())

        probabilidade = prever_probabilidade(modelo, entrada)
        predicao = int(probabilidade >= payload.threshold)
        classe = "sepse" if predicao == 1 else "sem sepse"

        return PredictResponse(
            probabilidade_sepse=round(float(probabilidade), 4),
            threshold_utilizado=payload.threshold,
            predicao=predicao,
            classe_predita=classe,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Erro ao processar a predição: {exc}",
        ) from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)