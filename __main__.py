from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# ============================================================
# CONFIGURAÇÃO BASE
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "modelos_salvos" / "modelo_sepse_sem_tempo_admin.pkl"
FEATURES_PATH = ROOT / "data" / "processed" / "features_modelo_sem_tempo_admin.csv"
MEDIANAS_PATH = ROOT / "data" / "processed" / "medianas_treino_sem_tempo_admin.csv"

DEFAULT_THRESHOLD = 0.12
API_VERSION = "1.4.0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_sepse")

app = FastAPI(
    title="API de Detecção de Sepse",
    description="API para inferência de risco de sepse.",
    version=API_VERSION,
)

# ============================================================
# MODELOS DE ENTRADA E SAÍDA
# ============================================================


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Dicionário com as features de entrada do paciente.",
    )
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Threshold opcional. Se não for enviado, a API usa o threshold salvo no artefato ou o padrão.",
    )


class PredictResponse(BaseModel):
    probabilidade_sepse: float
    threshold_utilizado: float
    predicao: int
    classe_predita: str


# ============================================================
# FUNÇÕES DE CARREGAMENTO
# ============================================================


def carregar_artefato() -> Any:
    if not MODEL_PATH.exists():
        logger.warning("Artefato não encontrado em: %s", MODEL_PATH)
        return None

    try:
        artefato = joblib.load(MODEL_PATH)
        logger.info("Artefato carregado com sucesso: %s", MODEL_PATH)
        return artefato
    except Exception as exc:
        logger.exception("Erro ao carregar artefato: %s", exc)
        return None


def extrair_modelo(artefato: Any) -> Any:
    if artefato is None:
        return None

    if hasattr(artefato, "predict_proba") or hasattr(artefato, "predict"):
        return artefato

    if isinstance(artefato, dict):
        for chave in [
            "modelo",
            "model",
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


def extrair_threshold_padrao(artefato: Any) -> float:
    if isinstance(artefato, dict):
        for chave in ["threshold_validacao_modelo", "threshold_base"]:
            valor = artefato.get(chave)
            if valor is not None:
                try:
                    return float(valor)
                except Exception:
                    pass
    return DEFAULT_THRESHOLD


def extrair_features_do_artefato(artefato: Any) -> List[str]:
    if isinstance(artefato, dict):
        features = artefato.get("features")
        if isinstance(features, list) and features:
            return [str(col) for col in features]
    return []


def carregar_features_esperadas(modelo: Any, artefato: Any) -> List[str]:
    """
    Prioridade:
    1) features salvas no artefato
    2) booster do XGBoost
    3) feature_names_in_
    4) CSV de features
    """
    features_artefato = extrair_features_do_artefato(artefato)
    if features_artefato:
        logger.info("Features carregadas do artefato salvo.")
        return features_artefato

    try:
        if modelo is not None and hasattr(modelo, "get_booster"):
            booster = modelo.get_booster()
            if booster is not None and booster.feature_names:
                logger.info("Features carregadas do booster do modelo.")
                return [str(col) for col in booster.feature_names]
    except Exception as exc:
        logger.warning("Não consegui ler feature_names do booster: %s", exc)

    try:
        if modelo is not None and hasattr(modelo, "feature_names_in_"):
            logger.info("Features carregadas de feature_names_in_.")
            return [str(col) for col in modelo.feature_names_in_]
    except Exception as exc:
        logger.warning("Não consegui ler feature_names_in_: %s", exc)

    if FEATURES_PATH.exists():
        try:
            df = pd.read_csv(FEATURES_PATH)

            if "feature" in df.columns:
                logger.info("Features carregadas do CSV com coluna 'feature'.")
                return df["feature"].dropna().astype(str).tolist()

            logger.info("Features carregadas da primeira coluna do CSV.")
            return df.iloc[:, 0].dropna().astype(str).tolist()
        except Exception as exc:
            logger.warning("Erro ao carregar features do CSV: %s", exc)

    return []


def carregar_medianas(artefato: Any) -> Dict[str, float]:
    if isinstance(artefato, dict):
        medianas_artefato = artefato.get("medianas_treino")
        if isinstance(medianas_artefato, dict) and medianas_artefato:
            try:
                logger.info("Medianas carregadas do artefato salvo.")
                return {str(k): float(v) for k, v in medianas_artefato.items()}
            except Exception as exc:
                logger.warning("Erro ao converter medianas do artefato: %s", exc)

    if not MEDIANAS_PATH.exists():
        return {}

    try:
        df = pd.read_csv(MEDIANAS_PATH)

        if {"coluna", "mediana"}.issubset(df.columns):
            logger.info("Medianas carregadas do CSV com colunas 'coluna' e 'mediana'.")
            return dict(zip(df["coluna"].astype(str), df["mediana"].astype(float)))

        if {"feature", "mediana"}.issubset(df.columns):
            logger.info("Medianas carregadas do CSV com colunas 'feature' e 'mediana'.")
            return dict(zip(df["feature"].astype(str), df["mediana"].astype(float)))

        if df.shape[1] >= 2:
            logger.info("Medianas carregadas das duas primeiras colunas do CSV.")
            return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(float)))

    except Exception as exc:
        logger.warning("Erro ao carregar medianas: %s", exc)

    return {}


# ============================================================
# PREPARAÇÃO DA ENTRADA
# ============================================================


def preparar_entrada(
    features_recebidas: Dict[str, Any],
    features_esperadas: List[str],
    medianas: Dict[str, float],
) -> pd.DataFrame:
    """
    Monta a entrada no formato exato esperado pelo modelo.
    """
    entrada = pd.DataFrame([features_recebidas])

    for col in features_esperadas:
        if col not in entrada.columns:
            entrada[col] = medianas.get(col, 0)

    entrada = entrada.reindex(columns=features_esperadas)
    entrada.columns = entrada.columns.astype(str)

    for col in entrada.columns:
        entrada[col] = pd.to_numeric(entrada[col], errors="coerce")
        entrada[col] = entrada[col].fillna(medianas.get(col, 0))

    entrada = entrada.fillna(0)

    return entrada


def prever_probabilidade(modelo: Any, entrada: pd.DataFrame) -> float:
    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(entrada)
        return float(proba[0, 1])

    if hasattr(modelo, "predict"):
        pred = modelo.predict(entrada)
        return float(pred[0])

    raise ValueError("Modelo sem predict_proba() e sem predict().")


# ============================================================
# ESTADO GLOBAL DA APLICAÇÃO
# ============================================================

artefato = None
modelo = None
features_esperadas: List[str] = []
medianas: Dict[str, float] = {}
threshold_padrao_modelo = DEFAULT_THRESHOLD


def recarregar_recursos() -> None:
    global artefato, modelo, features_esperadas, medianas, threshold_padrao_modelo

    artefato = carregar_artefato()
    modelo = extrair_modelo(artefato)
    features_esperadas = carregar_features_esperadas(modelo, artefato)
    medianas = carregar_medianas(artefato)
    threshold_padrao_modelo = extrair_threshold_padrao(artefato)

    logger.info("Modelo extraído: %s", modelo is not None)
    logger.info("Quantidade de features carregadas: %s", len(features_esperadas))
    logger.info("Quantidade de medianas carregadas: %s", len(medianas))
    logger.info("Threshold padrão carregado: %.4f", threshold_padrao_modelo)


recarregar_recursos()

# ============================================================
# ENDPOINTS
# ============================================================


@app.get("/")
def root():
    return {
        "mensagem": "API de detecção de sepse ativa.",
        "docs": "/docs",
        "health": "/health",
        "metadata": "/metadata",
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
        "threshold_padrao_modelo": threshold_padrao_modelo,
        "caminho_modelo": str(MODEL_PATH),
        "caminho_features_csv": str(FEATURES_PATH),
        "caminho_medianas_csv": str(MEDIANAS_PATH),
    }


@app.get("/metadata")
def metadata():
    return {
        "nome_projeto": "Detecção de Sepse com Machine Learning",
        "versao_api": API_VERSION,
        "threshold_padrao": threshold_padrao_modelo,
        "modelo_extraido": modelo is not None,
        "features_carregadas": len(features_esperadas),
        "medianas_carregadas": len(medianas),
    }


@app.post("/reload")
def reload_model():
    recarregar_recursos()
    return {
        "mensagem": "Recursos recarregados com sucesso.",
        "modelo_extraido": modelo is not None,
        "features_carregadas": len(features_esperadas),
        "medianas_carregadas": len(medianas),
        "threshold_padrao_modelo": threshold_padrao_modelo,
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

        threshold_utilizado = (
            float(payload.threshold)
            if payload.threshold is not None
            else float(threshold_padrao_modelo)
        )

        probabilidade = prever_probabilidade(modelo, entrada)
        predicao = int(probabilidade >= threshold_utilizado)
        classe = "sepse" if predicao == 1 else "sem sepse"

        logger.info("Features esperadas: %s", len(features_esperadas))
        logger.info("Colunas da entrada: %s", len(entrada.columns))
        logger.info("Threshold utilizado: %.4f", threshold_utilizado)
        logger.info("Probabilidade prevista: %.4f", probabilidade)

        return PredictResponse(
            probabilidade_sepse=round(float(probabilidade), 4),
            threshold_utilizado=round(float(threshold_utilizado), 4),
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