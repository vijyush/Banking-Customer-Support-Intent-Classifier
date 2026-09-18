from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
CATEGORIES_PATH = DATA_DIR / "categories.json"

MODELS_DIR = ROOT / "models"
BASELINE_MODEL_PATH = MODELS_DIR / "tfidf_linear_svc.joblib"
TRANSFORMER_MODEL_PATH = MODELS_DIR / "minilm_logistic_regression.joblib"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CACHE_DIR = ROOT / "cache"

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.15
TARGET_ACCEPTED_ACCURACY = 0.95
