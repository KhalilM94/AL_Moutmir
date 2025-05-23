import unittest
from main import SoilModelTraining

class DummyConfig:
    MODEL_REGISTRY = {
        "DummyModel": {
            "enabled": True,
            "import_path": "sklearn.linear_model.LinearRegression",
            "init_args": {},
            "params": {}
        }
    }

    def __init__(self):
        self.MODEL_REGISTRY = DummyConfig.MODEL_REGISTRY

class TestSoilModelTraining(unittest.TestCase):
    def test_get_model_configurations_basic(self):
        trainer = SoilModelTraining.__new__(SoilModelTraining)
        trainer.config = DummyConfig()
        trainer._dynamic_import = staticmethod(
            lambda path: __import__("sklearn.linear_model", fromlist=["LinearRegression"]).LinearRegression
        )
        configs = trainer._get_model_configurations(num_features=5)
        self.assertIn("DummyModel", configs)
        self.assertTrue(hasattr(configs["DummyModel"]["model"], "fit"))
        self.assertTrue(hasattr(configs["DummyModel"]["model"], "predict"))

if __name__ == "__main__":
    unittest.main()