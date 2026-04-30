from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from recommender.config import (  # noqa: E402,F401
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_MOVIES_PATH,
    DEFAULT_RATINGS_PATH,
    PROJECT_ROOT,
)
