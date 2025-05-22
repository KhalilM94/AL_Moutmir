from helpers.model_trainer import ModelTrainer
from helpers.training_logger import TrainingLogger
from helpers.utils import SpatialClusterSplitter
from helpers.data_manager import DataManager
import pandas as pd
import os
import datetime
import joblib
from typing import Dict
import importlib
from config import Config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class SoilModelTraining:
    def __init__(self):

        
        # Initialize components
        self.config = Config()
        # Setup directories
        self.output_dir = self._setup_directories(self.config.OUTPUT_FOLDER)
        self.logger = TrainingLogger(name='AlMoutmir Soil Models Training', 
                                     log_dir= os.path.join(self.output_dir, "logs")).get_logger()
        # Log configuration options
        self.logger.info("Configuration Options:")
        self.logger.info(f"  COLUMNS_TO_TRANSFORM: {self.config.COLUMNS_TO_TRANSFORM}")
        self.logger.info(f"  SPLIT_STRATEGY: {self.config.SPLIT_STRATEGY}")
        self.logger.info(f"  ENABLE_TUNING: {self.config.ENABLE_TUNING}")
        self.logger.info(f"  USE_BAYES_OPT: {self.config.USE_BAYES_OPT}")
        self.logger.info(f"  ENABLE_RFE: {self.config.ENABLE_RFE}")
        self.logger.info(f"  RANDOM_SEED: {self.config.RANDOM_SEED}")
        self.logger.info(f"  TARGET_COLUMNS: {self.config.TARGET_COLUMNS}")


        self.data_manager = DataManager(self.config, self.logger)
        # Spatial clustering splitter
        self.cluster_splitter = SpatialClusterSplitter(random_state= self.config.RANDOM_SEED)

    @staticmethod    
    def _setup_directories(output_dir: str = "output") -> str:
        """Create a unique output directory inside the given parent directory, with subdirectories for final models and metrics."""
        # Ensure the parent output directory exists
        parent_dir = os.path.abspath(output_dir)
        os.makedirs(parent_dir, exist_ok=True)

        # Create a unique directory name using timestamp inside the parent directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_output_dir = os.path.join(parent_dir, f"run_{timestamp}")

        # Create subdirectories
        final_models_path = os.path.join(unique_output_dir, "final_models")
        metrics_path = os.path.join(unique_output_dir, "metrics")

        os.makedirs(final_models_path, exist_ok=True)
        os.makedirs(metrics_path, exist_ok=True)

        return unique_output_dir
        
    @staticmethod
    def _dynamic_import(import_path):
        module_path, class_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _get_model_configurations(self, num_features: int):
        """Build dynamic model configurations from self.config.MODEL_REGISTRY."""
        model_configs = {}

        for name, spec in self.config.MODEL_REGISTRY.items():
            if not spec.get("enabled", False):
                continue
            try:
                ModelClass = self._dynamic_import(spec["import_path"])
            except Exception as e:
                print(f"[Warning] Failed to import {name}: {e}")
                continue

            init_args = spec.get("init_args", {})
            custom_model_builder = spec.get("custom_model_builder", None)

            # Handle Keras or other wrappers with custom model builders
            if custom_model_builder:
                builder_func = self._dynamic_import(custom_model_builder)
                model_instance = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", ModelClass(build_fn=lambda: builder_func(num_features)))
                ])
            else:
                model_instance = ModelClass(**init_args)

            model_configs[name] = {
                "model": model_instance,
                "params": spec.get("params", {})
            }

        return model_configs
        
    def train_models(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
                    X_test: pd.DataFrame, y_test: pd.DataFrame, 
                    groups_train: pd.Series) -> pd.DataFrame:
        """Train models for all targets and return results."""
        self.logger.info("Starting full training process for all targets...")
        trainer = ModelTrainer(
            model_pipelines = self._get_model_configurations(num_features=X_train.shape[1]),
            columns_to_transform=self.config.COLUMNS_TO_TRANSFORM,
            split_strategy=self.config.SPLIT_STRATEGY,
            enable_hyperparameter_tuning=self.config.ENABLE_TUNING,
            use_bayes_opt=self.config.USE_BAYES_OPT,
            enable_rfe=self.config.ENABLE_RFE,
            seed=self.config.RANDOM_SEED,
            output_dir=self.output_dir,
            logger=self.logger
        )
        
        all_results = []
        for target in self.config.TARGET_COLUMNS:
            results = trainer.train(
                target=target,
                X_train=X_train,
                y_train=y_train[target],
                X_test=X_test,
                y_test=y_test[target],
                groups_train=groups_train
            )
            if results:
                all_results.extend(results)
                
        metrics_df = pd.DataFrame(all_results)
        return metrics_df

    def save_results(self, metrics_df: pd.DataFrame, test_data: Dict):
        """Save training results and test sets."""
        metrics_dir = os.path.join(self.output_dir, "metrics")
        models_dir = os.path.join(self.output_dir, "final_models")

        if metrics_df.empty or "target" not in metrics_df.columns:
            self.logger.error("No valid metrics to save. Skipping result saving.")
            print("No valid metrics to save. Check logs for errors.")
            return

        # Save metrics
        idx = metrics_df.groupby("target")["Test_R2"].idxmax()
        top_models = metrics_df.loc[idx].reset_index(drop=True)

        top_models.to_csv(os.path.join(metrics_dir, "best_models_summary.csv"), index=False)
        metrics_df.to_csv(os.path.join(metrics_dir, "all_models_metrics.csv"), index=False)
        self.logger.info(f"Metrics saved to {metrics_dir}")

        # Save test sets
        joblib.dump({
            "X_test": test_data['X_test'],
            "y_test": test_data['y_test']
        }, os.path.join(models_dir, "test_sets.pkl"))
        self.logger.info(f"Test sets saved to {models_dir}/test_sets.pkl")

        # Display results
        print("\nFinal Metrics DataFrame:")
        print(metrics_df)

def main():
    # Initialize and run the training pipeline
    trainer = SoilModelTraining()
    
    try:
        # Load and preprocess data
        raw_data = trainer.data_manager.load_data()
        processed_data = trainer.data_manager.preprocess_data(raw_data)
        
        # Split data
        split_data = trainer.data_manager.split_data(
            processed_data['X'],
            processed_data['y'],
            processed_data['groups']
        )
        
        # Train models
        metrics = trainer.train_models(
            split_data['X_train'],
            split_data['y_train'],
            split_data['X_test'],
            split_data['y_test'],
            split_data['groups_train']
        )
        
        # Save results
        trainer.save_results(metrics, {
            'X_test': split_data['X_test'],
            'y_test': split_data['y_test']
        })
        
    except Exception as e:
        trainer.logger.error(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    main()