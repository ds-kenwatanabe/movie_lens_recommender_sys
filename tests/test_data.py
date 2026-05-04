import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@unittest.skipIf(importlib.util.find_spec("pandas") is None, "pandas is not installed")
class MovieLensDataTests(unittest.TestCase):
    def test_id_maps_use_sorted_unique_id_order(self):
        from recommender.data import MovieLens

        ratings = "\n".join(
            [
                "userId,movieId,rating,timestamp",
                "20,300,4.0,1",
                "10,100,5.0,2",
                "20,100,3.0,3",
                "30,200,2.0,4",
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            ratings_path = Path(directory) / "ratings.csv"
            ratings_path.write_text(ratings, encoding="utf-8")

            dataset = MovieLens(ratings_path)

        self.assertEqual(dataset.user_map, {10: 0, 20: 1, 30: 2})
        self.assertEqual(dataset.movie_map, {100: 0, 200: 1, 300: 2})
        self.assertEqual(dataset.size, (3, 3))


if __name__ == "__main__":
    unittest.main()
