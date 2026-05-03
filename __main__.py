from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn


MODEL_PATH = Path("modelos_salvos/modelo_sepse_sem_tempo_admin.pkl")
DEFAULT_THRESHOLD = 0.12

app = FastAPI(
    title="API de Detecção de Sepse",
    description="API para inferência de risco de sepse a partir de um modelo treinado.",
    version="1.0.0",
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


def carregar_modelo():
    """
    Carrega o modelo salvo em disco.
    """
    if not MODEL_PATH.exists():
        return None

    try:
        modelo = joblib.load(MODEL_PATH)
        return modelo
    except Exception as exc:
        print(f"Erro ao carregar modelo: {exc}")
        return None


modelo = carregar_modelo()


@app.get("/")
def root():
    """
    Endpoint inicial.
    """
    return {
        "mensagem": "API de detecção de sepse ativa.",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    """
    Verifica se a aplicação e o modelo estão disponíveis.
    """
    return {
        "status_api": "ok",
        "modelo_carregado": modelo is not None,
        "caminho_modelo": str(MODEL_PATH)
    }


@app.get("/metadata")
def metadata():
    """
    Retorna metadados básicos da API.
    """
    return {
        "nome_projeto": "Detecção de Sepse com Machine Learning",
        "versao_api": "1.0.0",
        "threshold_padrao": DEFAULT_THRESHOLD,
        "modelo_carregado": modelo is not None
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Realiza inferência no modelo a partir de um dicionário de features.
    """
    if modelo is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Modelo não carregado. Verifique se o arquivo existe em "
                f"'{MODEL_PATH}'."
            ),
        )

    try:
        entrada = pd.DataFrame([payload.features])

        if not hasattr(modelo, "predict_proba"):
            raise HTTPException(
                status_code=500,
                detail="O objeto carregado não possui o método predict_proba()."
            )

        probabilidade = float(modelo.predict_proba(entrada)[0, 1])
        predicao = int(probabilidade >= payload.threshold)
        classe = "sepse" if predicao == 1 else "sem sepse"

        return PredictResponse(
            probabilidade_sepse=round(probabilidade, 4),
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