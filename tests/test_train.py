from pathlib import Path
import importlib.util
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recommender.train import (  # noqa: E402
    build_bpr_samples,
    build_implicit_feedback_samples,
    temporal_split_indices,
)


class TemporalSplitTests(unittest.TestCase):
    def test_temporal_split_uses_earlier_ratings_for_training(self):
        timestamps = [30, 10, 50, 20, 40]

        train_indices, val_indices = temporal_split_indices(timestamps, val_ratio=0.4)

        self.assertEqual(train_indices, [1, 3, 0])
        self.assertEqual(val_indices, [4, 2])

    def test_temporal_split_rejects_invalid_ratio(self):
        with self.assertRaises(ValueError):
            temporal_split_indices([10, 20], val_ratio=0.0)


@unittest.skipIf(importlib.util.find_spec("pandas") is None, "pandas is not installed")
class NegativeSamplingTests(unittest.TestCase):
    def test_negative_sampling_uses_high_ratings_as_positives_and_unseen_movies_as_negatives(
        self,
    ):
        import pandas as pd

        data = pd.DataFrame(
            {
                "normalized_user_id": [0, 0, 1],
                "normalized_movie_id": [0, 1, 2],
                "rating": [5.0, 3.0, 4.5],
            }
        )

        samples = build_implicit_feedback_samples(
            data,
            positive_indices=[0, 1, 2],
            num_movies=4,
            relevance_threshold=4.0,
            negatives_per_positive=1,
            seed=42,
        )

        positives = {
            (user_id, movie_id) for user_id, movie_id, label in samples if label == 1.0
        }
        negatives = {
            (user_id, movie_id) for user_id, movie_id, label in samples if label == 0.0
        }

        self.assertEqual(positives, {(0, 0), (1, 2)})
        self.assertEqual(len(negatives), 2)
        self.assertTrue(
            all(
                movie_id not in {0, 1}
                for user_id, movie_id in negatives
                if user_id == 0
            )
        )
        self.assertTrue(
            all(movie_id != 2 for user_id, movie_id in negatives if user_id == 1)
        )

    def test_bpr_samples_pair_positives_with_sampled_negatives(self):
        import pandas as pd

        data = pd.DataFrame(
            {
                "normalized_user_id": [0, 0, 1],
                "normalized_movie_id": [0, 1, 2],
                "rating": [5.0, 3.0, 4.5],
            }
        )

        samples = build_bpr_samples(
            data,
            positive_indices=[0, 1, 2],
            num_movies=4,
            relevance_threshold=4.0,
            negatives_per_positive=1,
            seed=42,
        )

        self.assertEqual(len(samples), 2)
        self.assertTrue(all(len(sample) == 3 for sample in samples))
        self.assertEqual(
            {(user_id, positive) for user_id, positive, _ in samples}, {(0, 0), (1, 2)}
        )


if __name__ == "__main__":
    unittest.main()
