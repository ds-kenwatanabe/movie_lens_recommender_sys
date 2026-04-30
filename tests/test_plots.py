from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class PlotTests(unittest.TestCase):
    def test_embedding_method_validation_is_exposed_by_cli(self):
        from recommender.plots import parse_args
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ["plots", "--embedding-method", "tsne"]
            args = parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual(args.embedding_method, "tsne")


if __name__ == "__main__":
    unittest.main()

