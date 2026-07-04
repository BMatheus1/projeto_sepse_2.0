from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"
ORIGINAL_MODELS_DIR = ROOT_DIR / "modelos_salvos"

TRAIN_PATH = DATA_DIR / "train_melhor.parquet"
VALIDATION_PATH = DATA_DIR / "val_melhor.parquet"
TEST_PATH = DATA_DIR / "test_melhor.parquet"
ORIGINAL_MODEL_PATH = ORIGINAL_MODELS_DIR / "modelo_sepse_sem_tempo_admin.pkl"
FEATURES_PATH = DATA_DIR / "features_modelo_sem_tempo_admin.csv"
MEDIANS_PATH = DATA_DIR / "medianas_treino_sem_tempo_admin.csv"

TARGET_COLUMN = "paciente_teve_sepse"
ID_COLUMNS = ["patient_id"]
DEFAULT_THRESHOLD = 0.12
RANDOM_STATE = 42

HYPERPARAMETER_SPACE = {
    "max_depth": (2, 8, "int"),
    "learning_rate": (0.01, 0.20, "float"),
    "n_estimators": (80, 450, "int"),
    "subsample": (0.60, 1.00, "float"),
    "colsample_bytree": (0.60, 1.00, "float"),
    "min_child_weight": (1, 10, "int"),
    "gamma": (0.00, 5.00, "float"),
    "reg_alpha": (0.00, 5.00, "float"),
    "reg_lambda": (0.50, 8.00, "float"),
}


def ensure_project_dirs() -> None:
    for path in [REPORTS_DIR, LOGS_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
