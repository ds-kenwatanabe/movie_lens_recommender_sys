from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from recommender.model import MatrixFactorization  # noqa: E402,F401
