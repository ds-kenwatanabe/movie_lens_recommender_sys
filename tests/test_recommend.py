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
        recommender.movies = pd.DataFrame(
            {
                "movieId": [10, 20],
                "title": ["A", "B"],
                "genres": ["Action|Drama", "Comedy"],
            }
        )

        self.assertTrue(recommender._genre_matches(0, "Drama"))
        self.assertFalse(recommender._genre_matches(1, "Drama"))

    @unittest.skipIf(
        importlib.util.find_spec("torch") is None, "torch is not installed"
    )
    def test_recommend_for_user_excludes_interacted_movies(self):
        import pandas as pd

        from recommender.recommend import MovieRecommender

        class ModelStub:
            def __call__(self, user_ids, movie_ids):
                return movie_ids.float()

        recommender = object.__new__(MovieRecommender)
        recommender.model = ModelStub()
        recommender.movielens = type(
            "MovieLensStub",
            (),
            {
                "user_map": {123: 0},
                "movies": [10, 20, 30],
                "data": pd.DataFrame(
                    {
                        "normalized_user_id": [0],
                        "normalized_movie_id": [1],
                    }
                ),
            },
        )()
        recommender.movies = pd.DataFrame(
            {
                "movieId": [10, 20, 30],
                "title": ["A", "B", "C"],
                "genres": ["Action", "Action", "Action"],
            }
        )

        recommendations = recommender.recommend_for_user(123, top_k=3)

        self.assertEqual([movie_index for movie_index, _ in recommendations], [2, 0])


if __name__ == "__main__":
    unittest.main()
