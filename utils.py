from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from recommender.io import load_model, save_model  # noqa: E402,F401
