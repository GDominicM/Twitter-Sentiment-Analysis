from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET = BASE_DIR / "data" / "tweets_100k.csv"
MODELS_DIR = BASE_DIR / "models"