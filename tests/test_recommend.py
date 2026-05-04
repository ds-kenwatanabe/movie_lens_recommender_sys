import importlib.util
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@unittest.skipIf(importlib.util.find_spec("pandas") is None, "pandas is not installed")
class RecommendationTests(unittest.TestCase):
    def test_ordered_ids_from_map_reconstructs_normalized_order(self):
        from recommender.recommend import MovieRecommender

        self.assertEqual(
            MovieRecommender._ordered_ids_from_map({100: 1, 50: 0, 200: 2}),
            [50, 100, 200],
        )

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

        recommendations = recommender.recommend_for_user(123, top_k=2)

        self.assertEqual([movie_id for movie_id, _ in recommendations], [30, 10])

    def test_unknown_user_gets_cold_start_recommendations(self):
        import pandas as pd

        from recommender.recommend import MovieRecommender

        recommender = object.__new__(MovieRecommender)
        recommender.movielens = type(
            "MovieLensStub", (), {"user_map": {}, "movies": [10, 20]}
        )()
        recommender.movies = pd.DataFrame(
            {
                "movieId": [10, 20],
                "title": ["A", "B"],
                "genres": ["Action", "Drama"],
            }
        )
        recommender.catalog_movie_ids = [10, 20]
        recommender.catalog_popularity_scores = {10: 1.0, 20: 3.0}

        recommendations = recommender.recommend_for_user(999, top_k=1)

        self.assertEqual(recommendations, [(20, 3.0)])

    def test_cold_start_can_recommend_catalog_movies_absent_from_training(self):
        import pandas as pd

        from recommender.recommend import MovieRecommender

        recommender = object.__new__(MovieRecommender)
        recommender.movielens = type(
            "MovieLensStub", (), {"user_map": {}, "movies": [10]}
        )()
        recommender.movies = pd.DataFrame(
            {
                "movieId": [10, 20],
                "title": ["A", "B"],
                "genres": ["Action", "Action"],
            }
        )
        recommender.catalog_movie_ids = [10, 20]
        recommender.catalog_popularity_scores = {10: 1.0, 20: 3.0}

        recommendations = recommender.recommend_for_user(999, top_k=2)

        self.assertEqual(recommendations, [(20, 3.0), (10, 1.0)])

    def test_unknown_similar_movie_uses_catalog_genre_popularity(self):
        import pandas as pd

        from recommender.recommend import MovieRecommender

        recommender = object.__new__(MovieRecommender)
        recommender.movielens = type(
            "MovieLensStub", (), {"movie_map": {}, "movies": [10]}
        )()
        recommender.movies = pd.DataFrame(
            {
                "movieId": [10, 20, 30],
                "title": ["A", "B", "C"],
                "genres": ["Action", "Action|Drama", "Comedy"],
            }
        )
        recommender.catalog_movie_ids = [10, 20, 30]
        recommender.catalog_popularity_scores = {10: 1.0, 20: 3.0, 30: 5.0}

        recommendations = recommender.get_similar(10, top_n=1)

        self.assertEqual(recommendations, [(20, 3.0)])


if __name__ == "__main__":
    unittest.main()
