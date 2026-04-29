from pathlib import Path
import importlib.util
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class ImportTests(unittest.TestCase):
    def test_package_modules_are_discoverable(self):
        modules = [
            "recommender",
            "recommender.config",
            "recommender.data",
            "recommender.evaluate",
            "recommender.model",
            "recommender.recommend",
            "recommender.train",
        ]

        for module in modules:
            with self.subTest(module=module):
                self.assertIsNotNone(importlib.util.find_spec(module))


if __name__ == "__main__":
    unittest.main()
