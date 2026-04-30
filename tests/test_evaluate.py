import importlib.util
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
class EvaluationMetricTests(unittest.TestCase):
    def test_evaluate_model_returns_error_and_top_k_metrics(self):
        import torch

        from recommender.evaluate import evaluate_model

        class DummyEmbedding:
            num_embeddings = 3

        class DummyModel:
            movie_embedding = DummyEmbedding()

            def __init__(self):
                self.scores = torch.tensor(
                    [
                        [0.1, 0.9, 0.8],
                        [0.7, 0.2, 0.1],
                    ]
                )

            def eval(self):
                return self

            def __call__(self, user_ids, movie_ids):
                return self.scores[user_ids, movie_ids]

        dataloader = [
            (
                torch.tensor([[0], [0], [1]]),
                torch.tensor([[1], [2], [0]]),
                torch.tensor([[5.0], [3.0], [4.0]]),
            )
        ]

        metrics = evaluate_model(
            DummyModel(), dataloader, device="cpu", k=2, relevance_threshold=4.0
        )

        self.assertEqual(
            set(metrics),
            {
                "mae",
                "rmse",
                "precision@2",
                "recall@2",
                "ndcg@2",
                "hitrate@2",
                "coverage",
            },
        )
        self.assertAlmostEqual(metrics["precision@2"], 0.5)
        self.assertAlmostEqual(metrics["recall@2"], 1.0)
        self.assertAlmostEqual(metrics["ndcg@2"], 1.0)
        self.assertAlmostEqual(metrics["hitrate@2"], 1.0)
        self.assertAlmostEqual(metrics["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
