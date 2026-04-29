from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "ml-25m"
DEFAULT_RATINGS_PATH = DEFAULT_DATA_DIR / "ratings.csv"
DEFAULT_MOVIES_PATH = DEFAULT_DATA_DIR / "movies.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "recommender_model.pth"

