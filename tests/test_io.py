import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_restores_training_state(self):
        import torch

        from recommender.io import load_checkpoint, save_checkpoint
        from recommender.model import MatrixFactorization

        model = MatrixFactorization(num_users=2, num_movies=3, embedding_size=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
        metrics = {"mae": 0.2, "precision@10": 0.5}
        user_map = {10: 0, 20: 1}
        movie_map = {100: 0, 200: 1, 300: 2}
        config = {"epochs": 3, "learning_rate": 0.01}

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.pth"
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=2,
                validation_metrics=metrics,
                user_map=user_map,
                movie_map=movie_map,
                config=config,
            )

            restored_model = MatrixFactorization(
                num_users=2, num_movies=3, embedding_size=4
            )
            restored_optimizer = torch.optim.Adam(
                restored_model.parameters(), lr=0.01, weight_decay=0.001
            )
            checkpoint = load_checkpoint(
                checkpoint_path, restored_model, restored_optimizer
            )

        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["validation_metrics"], metrics)
        self.assertEqual(checkpoint["user_map"], user_map)
        self.assertEqual(checkpoint["movie_map"], movie_map)
        self.assertEqual(checkpoint["config"], config)


if __name__ == "__main__":
    unittest.main()
