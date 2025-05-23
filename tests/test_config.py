import unittest
import os
import tempfile
import yaml
from config import Config

class TestConfig(unittest.TestCase):
    def setUp(self):
        # Backup environment variables and set test values
        self.env_backup = dict(os.environ)
        os.environ['DATA_FOLDER'] = 'test_data_folder'
        os.environ['OUTPUT_FOLDER'] = 'test_outputs'
        os.environ['RANDOM_SEED'] = '123'
        os.environ['TEST_SIZE'] = '0.3'
        os.environ['USE_GROUP_SPLIT'] = 'True'
        os.environ['SPLIT_STRATEGY'] = 'groupkfold'
        os.environ['ENABLE_TUNING'] = 'True'
        os.environ['USE_BAYES_OPT'] = 'False'
        os.environ['ENABLE_RFE'] = 'True'
        os.environ['COLUMNS_TO_TRANSFORM'] = 'a,b,c'
        os.environ['TARGET_COLUMNS'] = 'target1,target2'
        os.environ['ELIMINATED_FEATURES'] = 'f1,f2'

        # Create a temporary model registry YAML file
        self.temp_yaml = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.yml')
        self.model_registry_content = {
            "PLSRegression": {
                "enabled": True,
                "import_path": "sklearn.cross_decomposition.PLSRegression",
                "params": {
                    "model__n_components": [2, 4, 6, 8, 10, 12, 14, 16, 18]
                }
            },
            "ElasticNet": {
                "enabled": False,
                "import_path": "sklearn.linear_model.ElasticNet",
                "init_args": {"max_iter": 10000},
                "params": {
                    "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 3.16],
                    "model__l1_ratio": [0.1, 0.325, 0.55, 0.775, 1.0]
                }
            },
            "KerasRegressor": {
                "enabled": False,
                "import_path": "tensorflow.keras.wrappers.scikit_learn.KerasRegressor",
                "custom_model_builder": "helpers.models.build_keras_model",
                "params": {
                    "model__epochs": [50, 100],
                    "model__batch_size": [16, 32],
                    "model__dropout_rate": [0.1, 0.2, 0.3]
                }
            }
        }
        yaml.dump(self.model_registry_content, self.temp_yaml)
        self.temp_yaml.close()
        os.environ['MODEL_REGISTRY_PATH'] = self.temp_yaml.name

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env_backup)
        if hasattr(self, 'temp_yaml'):
            os.unlink(self.temp_yaml.name)

    def test_config_env_override(self):
        config = Config()
        self.assertEqual(config.DATA_FOLDER, 'test_data_folder')
        self.assertEqual(config.OUTPUT_FOLDER, 'test_outputs')
        self.assertEqual(config.RANDOM_SEED, 123)
        self.assertEqual(config.TEST_SIZE, 0.3)
        self.assertTrue(config.USE_GROUP_SPLIT)
        self.assertEqual(config.SPLIT_STRATEGY, 'groupkfold')
        self.assertTrue(config.ENABLE_TUNING)
        self.assertFalse(config.USE_BAYES_OPT)
        self.assertTrue(config.ENABLE_RFE)
        self.assertEqual(config.COLUMNS_TO_TRANSFORM, ['a', 'b', 'c'])
        self.assertEqual(config.TARGET_COLUMNS, ['target1', 'target2'])
        self.assertEqual(config.ELIMINATED_FEATURES, ['f1', 'f2'])

    def test_load_model_registry(self):
        config = Config()
        registry = config.MODEL_REGISTRY
        self.assertIsInstance(registry, dict)
        # The loaded registry should be equal to the YAML content we wrote
        self.assertEqual(registry, self.model_registry_content)

class TestConfigWithRepoYAML(unittest.TestCase):
    def setUp(self):
        # Backup environment variables and set test values
        self.env_backup = dict(os.environ)
        # Set environment variables to use the actual YAML files in the repo
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.environ['CONFIG_PATH'] = os.path.join(repo_dir, 'config.yml')
        os.environ['MODEL_REGISTRY_PATH'] = os.path.join(repo_dir, 'model_registry.yml')

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env_backup)

    def test_load_model_registry_from_repo_file(self):
        # Load the expected content directly from the YAML file
        with open(os.environ['MODEL_REGISTRY_PATH'], 'r') as f:
            expected_registry = yaml.safe_load(f)
        config = Config()
        registry = config.MODEL_REGISTRY
        self.assertIsInstance(registry, dict)
        self.assertEqual(registry, expected_registry)

    def test_config_values_from_repo_file(self):
        # Load the expected config values directly from the YAML file
        with open(os.environ['CONFIG_PATH'], 'r') as f:
            expected_config = yaml.safe_load(f)
        config = Config()
        # Check a few representative values
        self.assertEqual(config.DATA_FOLDER, expected_config['DATA_FOLDER'])
        self.assertEqual(config.OUTPUT_FOLDER, expected_config['OUTPUT_FOLDER'])
        self.assertEqual(config.TEST_SIZE, expected_config['TEST_SIZE'])
        self.assertEqual(config.SPLIT_STRATEGY, expected_config['SPLIT_STRATEGY'])
        self.assertEqual(config.ENABLE_TUNING, expected_config['ENABLE_TUNING'])
        self.assertEqual(config.USE_BAYES_OPT, expected_config['USE_BAYES_OPT'])
        self.assertEqual(config.ENABLE_RFE, expected_config['ENABLE_RFE'])
        self.assertEqual(config.RANDOM_SEED, expected_config['RANDOM_SEED'])
        self.assertEqual(config.COLUMNS_TO_TRANSFORM, expected_config['COLUMNS_TO_TRANSFORM'])
        self.assertEqual(config.TARGET_COLUMNS, expected_config['TARGET_COLUMNS'])
        self.assertEqual(config.ELIMINATED_FEATURES, expected_config['ELIMINATED_FEATURES'])

if __name__ == "__main__":
    unittest.main()