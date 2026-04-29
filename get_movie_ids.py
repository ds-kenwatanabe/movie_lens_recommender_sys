from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from recommender.recommend import preview_movie_ids_main  # noqa: E402


if __name__ == "__main__":
    preview_movie_ids_main()
