import importlib.util
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
class MatrixFactorizationTests(unittest.TestCase):
    def test_forward_includes_global_mean_and_biases(self):
        import torch

        from recommender.model import MatrixFactorization

        model = MatrixFactorization(
            num_users=1, num_movies=1, embedding_size=2, global_mean=3.5
        )
        with torch.no_grad():
            model.user_embedding.weight.fill_(2.0)
            model.movie_embedding.weight.fill_(4.0)
            model.user_bias.weight.fill_(0.25)
            model.movie_bias.weight.fill_(-0.75)

        prediction = model(torch.tensor([0]), torch.tensor([0]))

        self.assertTrue(torch.allclose(prediction, torch.tensor([19.0])))


if __name__ == "__main__":
    unittest.main()
