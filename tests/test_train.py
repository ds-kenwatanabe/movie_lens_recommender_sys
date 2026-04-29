from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recommender.train import temporal_split_indices  # noqa: E402


class TemporalSplitTests(unittest.TestCase):
    def test_temporal_split_uses_earlier_ratings_for_training(self):
        timestamps = [30, 10, 50, 20, 40]

        train_indices, val_indices = temporal_split_indices(timestamps, val_ratio=0.4)

        self.assertEqual(train_indices, [1, 3, 0])
        self.assertEqual(val_indices, [4, 2])

    def test_temporal_split_rejects_invalid_ratio(self):
        with self.assertRaises(ValueError):
            temporal_split_indices([10, 20], val_ratio=0.0)


if __name__ == "__main__":
    unittest.main()

