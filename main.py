from yg_eo_soilnet import DataManager, LogTransformer
from yg_eo_soilnet.logger import ParentRunLogger, TrainingLogger
from yg_eo_soilnet.models import ModelConfigFactory
from yg_eo_soilnet.trainers import ModelTrainer
from config import Config
import mlflow
import datetime
from typing import Dict
import argparse

class SoilModelTraining:
    def __init__(
        self,
        run_name: str = "Soil_Model_Training",
        config_path: str = "config.yml",
        registry_path: str = "model_registry.yml",
    ):
        self.run_name = run_name
        # Initialize components
        self.config = Config(config_path=config_path, registry_path=registry_path)
        # Setup directories
        self.log_transformer = LogTransformer()
        self.logger_wrapper = TrainingLogger(name='AlMoutmir Soil Models Training',
                                     log_filename=self.run_name)
        self.logger = self.logger_wrapper.get_logger()

        self.data_manager = DataManager(self.config, self.logger)
        # Spatial clustering splitter
        self.model_configs = ModelConfigFactory(self.config.MODEL_REGISTRY)
        
    def train_models(self, data: Dict):
        """Train models for all targets and return results and fold_preds if CV_ONLY_MODE."""
        self.logger.info("Starting full training process for all targets...")
        X_key = "X_train" if "X_train" in data else "X"
        trainer = ModelTrainer(
            config=self.config,
            columns_to_transform=self.config.COLUMNS_TO_TRANSFORM,
            enable_clustering =self.config.ENABLE_CLUSTERING,
            split_strategy=self.config.SPLIT_STRATEGY,
            seed=self.config.RANDOM_SEED,
            logger=self.logger
        )
        for target in self.config.TARGET_COLUMNS:
            trainer.train(
                target=target,
                data=data,
                model_pipelines = self.model_configs.build_model_configs(num_features= data[X_key].shape[1] ),)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train soil models")
    parser.add_argument(
        "--config-path",
        default="config.yml",
        help="Path to the main YAML config file",
    )
    parser.add_argument(
        "--registry-path",
        default="model_registry.yml",
        help="Path to the model registry YAML file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mlflow.enable_system_metrics_logging()
    EXPERIMENT_NAME = "Soil_Model_Training_Experiment"
    # Set up MLflow tracking
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
    except mlflow.exceptions.MlflowException: # type: ignore
    # Restore or create a new experiment if previously deleted
        mlflow.create_experiment(EXPERIMENT_NAME)
        mlflow.set_experiment(EXPERIMENT_NAME)
    
    if mlflow.active_run():
        mlflow.end_run()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Run_{timestamp}"
    mlflow_logger = ParentRunLogger()
    with mlflow.start_run(run_name=run_name) as main_run:
        # Initialize and run the training pipeline
        trainer = SoilModelTraining(
            run_name=run_name,
            config_path=args.config_path,
            registry_path=args.registry_path,
        )
        mlflow.log_param("CONFIG_PATH", trainer.config.config_path)
        mlflow.log_param("REGISTRY_PATH", trainer.config.registry_path)
        try:
            # Load and preprocess data
            raw_data = trainer.data_manager.load_data()
            processed_data = trainer.data_manager.preprocess_data(raw_data)
            trainer.logger.info("Running in full training mode...")

            # Split data
            split_data = trainer.data_manager.split_data(processed_data)
            # Train models
            trainer.train_models(split_data)

            mlflow_logger.log_parent_summary(main_run.info.run_id, trainer)

        except Exception as e:
            trainer.logger.error(f"An error occurred during training: {e}")
            raise
        if trainer.logger_wrapper.log_file:
            mlflow.log_artifact(trainer.logger_wrapper.log_file)

if __name__ == "__main__":
    main()