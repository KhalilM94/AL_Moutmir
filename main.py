from helpers.model_trainer import ModelTrainer
from helpers.training_logger import TrainingLogger
from helpers.model_config_factory import ModelConfigFactory
from helpers.data_manager import DataManager
from helpers.io_utils import setup_directories
from helpers.plot_utils import plot_observed_vs_predicted
from helpers.misc_utils import LogTransformer
import pandas as pd
import os
import joblib
from typing import Dict
from config import Config

class SoilModelTraining:
    def __init__(self):

        # Initialize components
        self.config = Config()
        # Setup directories
        self.output_dir = setup_directories(self.config.OUTPUT_FOLDER)
        self.log_transformer = LogTransformer()
        self.logger = TrainingLogger(name='AlMoutmir Soil Models Training', 
                                     log_dir= os.path.join(self.output_dir, "logs")).get_logger()
        # Log configuration options
        self.logger.info("Configuration Options:")
        self.logger.info(f"  COLUMNS_TO_TRANSFORM: {self.config.COLUMNS_TO_TRANSFORM}")
        self.logger.info(f"  ENABLE_CLUSTERING: {self.config.ENABLE_CLUSTERING}")
        self.logger.info(f"  CLUSTERING_STRATEGY: {self.config.CLUSTERING_STRATEGY.get('class_path').rsplit(".", 1)[1] if self.config.ENABLE_CLUSTERING else None}")
        self.logger.info(f"  SPLIT_STRATEGY: {self.config.SPLIT_STRATEGY}")
        self.logger.info(f"  ENABLE_TUNING: {self.config.ENABLE_TUNING}")
        self.logger.info(f"  USE_BAYES_OPT: {self.config.USE_BAYES_OPT}")
        self.logger.info(f"  ENABLE_RFE: {self.config.ENABLE_RFE}")
        self.logger.info(f"  RANDOM_SEED: {self.config.RANDOM_SEED}")
        self.logger.info(f"  TARGET_COLUMNS: {self.config.TARGET_COLUMNS}")


        self.data_manager = DataManager(self.config, self.logger)
        # Spatial clustering splitter
        self.model_configs = ModelConfigFactory(self.config.MODEL_REGISTRY)
        
    def train_models(self, data: Dict) -> pd.DataFrame:
        
        """Train models for all targets and return results."""
        self.logger.info("Starting full training process for all targets...")
        trainer = ModelTrainer(
            model_pipelines = self.model_configs.build_model_configs(num_features=data['X_train'].shape[1]),
            columns_to_transform=self.config.COLUMNS_TO_TRANSFORM,
            enable_clustering =self.config.ENABLE_CLUSTERING,
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
                data=data)
            if results:
                all_results.extend(results)
                
        metrics_df = pd.DataFrame(all_results)
        return metrics_df

    def save_results(self, metrics_df: pd.DataFrame, test_data: Dict):
        """Save training results and test sets."""
        metrics_dir = os.path.join(self.output_dir, "metrics")
        models_dir = os.path.join(self.output_dir, "final_models")
        plots_dir = os.path.join(self.output_dir, "plots")

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
            "X_test": test_data['X_test'], "y_test": test_data['y_test']
        }, os.path.join(models_dir, "test_sets.pkl"))
        self.logger.info(f"Test sets saved to {models_dir}/test_sets.pkl")

         # --- Add plot_observed_vs_predicted and save the plot ---
        fig = plot_observed_vs_predicted(
            test_data['X_test'],
            test_data['y_test'],
            self.model_configs.build_model_configs(num_features=test_data['X_test'].shape[1]),
            self.config.TARGET_COLUMNS,
            self.config.COLUMNS_TO_TRANSFORM,
            model_dir=models_dir,
            sup_title="Test set Observed vs Predicted",
            log_transformer=self.log_transformer
        )
        os.makedirs(plots_dir, exist_ok=True)
        fig.savefig(os.path.join(plots_dir, "observed_vs_predicted.png"))   
        self.logger.info(f"Obs_vs_Pred Plot saved to {plots_dir}/observed_vs_predicted.png")

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
        split_data = trainer.data_manager.split_data(processed_data)
        
        # Train models
        metrics = trainer.train_models(split_data)
        
        # Save results
        trainer.save_results(metrics, {'X_test': split_data['X_test'],
                                       'y_test': split_data['y_test']})
        
    except Exception as e:
        trainer.logger.error(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    main()