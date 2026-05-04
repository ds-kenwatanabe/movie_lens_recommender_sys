import importlib.util
import inspect
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class BaselineImplementationTests(unittest.TestCase):
    def test_svd_baseline_uses_sparse_truncated_svd(self):
        from recommender.baselines import SVDBaseline

        source = inspect.getsource(SVDBaseline.fit)

        self.assertIn("csr_matrix", source)
        self.assertIn("TruncatedSVD", source)
        self.assertNotIn("np.full", source)
        self.assertNotIn("np.linalg.svd", source)
        self.assertNotIn("self.predictions", inspect.getsource(SVDBaseline))


@unittest.skipIf(
    importlib.util.find_spec("pandas") is None
    or importlib.util.find_spec("numpy") is None
    or importlib.util.find_spec("scipy") is None
    or importlib.util.find_spec("sklearn") is None,
    "pandas, numpy, scipy, and scikit-learn are required for baseline tests",
)
class BaselineTests(unittest.TestCase):
    def test_compare_baselines_returns_requested_models(self):
        import pandas as pd

        from recommender.baselines import compare_baselines

        data = pd.DataFrame(
            {
                "userId": [10, 10, 20, 20, 30, 30, 10, 20, 30, 40],
                "movieId": [100, 200, 100, 300, 200, 400, 300, 400, 100, 500],
                "rating": [5.0, 3.0, 4.5, 2.0, 5.0, 3.5, 4.0, 4.5, 2.0, 5.0],
                "timestamp": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }
        )
        data["normalized_user_id"] = data["userId"].astype("category").cat.codes
        data["normalized_movie_id"] = data["movieId"].astype("category").cat.codes

        results = compare_baselines(
            data,
            val_ratio=0.3,
            k=2,
            relevance_threshold=4.0,
            negatives_per_positive=1,
            seed=42,
            svd_factors=2,
        )

        self.assertEqual(
            set(results),
            {
                "global_mean",
                "user_mean",
                "movie_mean",
                "popularity",
                "item_item_cosine",
                "svd",
            },
        )
        for metrics in results.values():
            self.assertIn("precision@2", metrics)
            self.assertIn("recall@2", metrics)
            self.assertIn("coverage", metrics)


if __name__ == "__main__":
    unittest.main()
