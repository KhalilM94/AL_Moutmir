from helpers.model_trainer import ModelTrainer
from helpers.training_logger import TrainingLogger
from helpers.model_config_factory import ModelConfigFactory
from helpers.data_manager import DataManager
from helpers.misc_utils import LogTransformer
from helpers.mlflow_loggers import ParentRunLogger
from config import Config
import mlflow
import datetime
from typing import Dict

class SoilModelTraining:
    def __init__(self, run_name: str = "Soil_Model_Training"):
        self.run_name = run_name
        # Initialize components
        self.config = Config()
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

def main():
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
        trainer = SoilModelTraining(run_name=run_name)
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