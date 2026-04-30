import importlib.util
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@unittest.skipIf(importlib.util.find_spec("pandas") is None, "pandas is not installed")
class RecommendationTests(unittest.TestCase):
    def test_genre_filter_matches_pipe_separated_genres(self):
        import pandas as pd

        from recommender.recommend import MovieRecommender

        recommender = object.__new__(MovieRecommender)
        recommender.movielens = type("MovieLensStub", (), {"movies": [10, 20]})()
        recommender.movies = pd.DataFrame({
            "movieId": [10, 20],
            "title": ["A", "B"],
            "genres": ["Action|Drama", "Comedy"],
        })

        self.assertTrue(recommender._genre_matches(0, "Drama"))
        self.assertFalse(recommender._genre_matches(1, "Drama"))


if __name__ == "__main__":
    unittest.main()

