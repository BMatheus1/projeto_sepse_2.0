from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn


MODEL_PATH = Path("modelos_salvos/modelo_sepse_sem_tempo_admin.pkl")
FEATURES_PATH = Path("data/processed/features_modelo_sem_tempo_admin.csv")
DEFAULT_THRESHOLD = 0.12

app = FastAPI(
    title="API de Detecção de Sepse",
    description="API para inferência de risco de sepse a partir de um modelo treinado.",
    version="1.1.0",
)


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Dicionário com as features de entrada do paciente."
    )
    threshold: float = Field(
        DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Threshold de decisão para classificar sepse."
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


def carregar_features() -> Optional[List[str]]:
    if not FEATURES_PATH.exists():
        return None

    try:
        df = pd.read_csv(FEATURES_PATH)
        if "feature" in df.columns:
            return df["feature"].dropna().astype(str).tolist()
        return None
    except Exception as exc:
        print(f"Erro ao carregar features: {exc}")
        return None


def extrair_modelo(artefato):
    """
    Tenta encontrar o modelo real dentro do artefato salvo.
    """
    if artefato is None:
        return None

    # Caso já seja o próprio modelo
    if hasattr(artefato, "predict_proba") or hasattr(artefato, "predict"):
        return artefato

    # Caso seja dicionário
    if isinstance(artefato, dict):
        chaves_possiveis = [
            "model",
            "modelo",
            "estimator",
            "classifier",
            "clf",
            "pipeline",
            "best_model",
            "best_estimator",
            "modelo_final",
            "melhor_modelo",
        ]
        for chave in chaves_possiveis:
            if chave in artefato:
                candidato = artefato[chave]
                if hasattr(candidato, "predict_proba") or hasattr(candidato, "predict"):
                    return candidato

    # Caso seja lista/tupla
    if isinstance(artefato, (list, tuple)):
        for item in artefato:
            if hasattr(item, "predict_proba") or hasattr(item, "predict"):
                return item

    return None


def preparar_entrada(features_recebidas: Dict[str, Any], features_esperadas: Optional[List[str]]) -> pd.DataFrame:
    """
    Monta o DataFrame de entrada e alinha com as features esperadas pelo modelo.
    """
    entrada = pd.DataFrame([features_recebidas])

    if features_esperadas:
        for col in features_esperadas:
            if col not in entrada.columns:
                entrada[col] = 0

        entrada = entrada[features_esperadas]

    return entrada


def prever_probabilidade(modelo, entrada: pd.DataFrame) -> float:
    """
    Tenta obter a probabilidade da classe positiva.
    """
    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(entrada)
        if len(proba.shape) == 2 and proba.shape[1] > 1:
            return float(proba[0, 1])
        return float(proba[0])

    if hasattr(modelo, "predict"):
        pred = modelo.predict(entrada)

        # Sem probabilidade calibrada: retorna 0.0 ou 1.0 com base na predição
        # Isso não é uma probabilidade real, mas evita quebrar a API.
        return float(pred[0])

    raise ValueError("O modelo não possui nem predict_proba() nem predict().")


artefato = carregar_artefato()
modelo = extrair_modelo(artefato)
features_esperadas = carregar_features()


@app.get("/")
def root():
    return {
        "mensagem": "API de detecção de sepse ativa.",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status_api": "ok",
        "artefato_carregado": artefato is not None,
        "modelo_extraido": modelo is not None,
        "tipo_artefato": str(type(artefato)) if artefato is not None else None,
        "tipo_modelo": str(type(modelo)) if modelo is not None else None,
        "caminho_modelo": str(MODEL_PATH),
        "features_carregadas": len(features_esperadas) if features_esperadas else 0
    }


@app.get("/metadata")
def metadata():
    return {
        "nome_projeto": "Detecção de Sepse com Machine Learning",
        "versao_api": "1.1.0",
        "threshold_padrao": DEFAULT_THRESHOLD,
        "modelo_extraido": modelo is not None,
        "features_carregadas": len(features_esperadas) if features_esperadas else 0
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if artefato is None:
        raise HTTPException(
            status_code=500,
            detail=f"Artefato não carregado em '{MODEL_PATH}'."
        )

    if modelo is None:
        tipo = str(type(artefato))
        detalhe_extra = ""

        if isinstance(artefato, dict):
            detalhe_extra = f" Chaves encontradas: {list(artefato.keys())}"

        raise HTTPException(
            status_code=500,
            detail=(
                f"Não foi possível extrair um modelo utilizável do artefato. "
                f"Tipo do artefato: {tipo}.{detalhe_extra}"
            ),
        )

    try:
        entrada = preparar_entrada(payload.features, features_esperadas)
        probabilidade = prever_probabilidade(modelo, entrada)

        predicao = int(probabilidade >= payload.threshold)
        classe = "sepse" if predicao == 1 else "sem sepse"

        return PredictResponse(
            probabilidade_sepse=round(float(probabilidade), 4),
            threshold_utilizado=payload.threshold,
            predicao=predicao,
            classe_predita=classe,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Erro ao processar a predição: {str(exc)}"
        ) from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)