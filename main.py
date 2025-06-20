from helpers.model_trainer import ModelTrainer
from helpers.training_logger import TrainingLogger
from helpers.model_config_factory import ModelConfigFactory
from helpers.data_manager import DataManager
from helpers.io_utils import setup_directories
from helpers.plot_utils import plot_observed_vs_predicted, plot_cv_folds_observed_vs_predicted
from helpers.preprocessors import load_dict_from_file
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
        self.logger.info(f"  CLUSTERING_STRATEGY: {self.config.CLUSTERING_STRATEGY.get('class_path').rsplit('.', 1)[1] if self.config.ENABLE_CLUSTERING else None}")
        self.logger.info(f"  SPLIT_STRATEGY: {self.config.SPLIT_STRATEGY}")
        self.logger.info(f"  CV_ONLY_MODE: {self.config.CV_ONLY_MODE}")
        self.logger.info(f"  ENABLE_TUNING: {self.config.ENABLE_TUNING}")
        self.logger.info(f"  USE_BAYES_OPT: {self.config.USE_BAYES_OPT}")
        self.logger.info(f"  ENABLE_RFE: {self.config.ENABLE_RFE}")
        self.logger.info(f"  RANDOM_SEED: {self.config.RANDOM_SEED}")
        self.logger.info(f"  TARGET_COLUMNS: {self.config.TARGET_COLUMNS}")


        self.data_manager = DataManager(self.config, self.logger)
        # Spatial clustering splitter
        self.model_configs = ModelConfigFactory(self.config.MODEL_REGISTRY)
        
    def train_models(self, data: Dict):
        """Train models for all targets and return results and fold_preds if CV_ONLY_MODE."""
        self.logger.info("Starting full training process for all targets...")
        X_key = "X_train" if "X_train" in data else "X"
        trainer = ModelTrainer(
            model_pipelines = self.model_configs.build_model_configs(num_features= data[X_key].shape[1] ),
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
        all_fold_preds = {}
        for target in self.config.TARGET_COLUMNS:
            results = trainer.train(
                target=target,
                data=data)
            if results:
                all_results.extend(results)
            # If CV_ONLY_MODE, collect fold_preds
            if hasattr(trainer, 'fold_preds') and trainer.fold_preds is not None:
                all_fold_preds[target] = trainer.fold_preds
        metrics_df = pd.DataFrame(all_results)
        return metrics_df, all_fold_preds

    def save_results(self, metrics_df: pd.DataFrame, test_data: Dict):
        """Save training results and test sets."""
        metrics_dir = os.path.join(self.output_dir, "metrics")
        models_dir = os.path.join(self.output_dir, "final_models")
        plots_dir = os.path.join(self.output_dir, "plots")
        group_label_map = load_dict_from_file(self.config.SOIL_GROUPS_FILE_PATH)
        group_prefix = self.config.CATEGORICAL_FEATURES[0] + "_"

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

        # --- Add plot_observed_vs_predicted and save the plot ---
        fig = plot_observed_vs_predicted(
            test_data['X_test'],
            test_data['y_test'],
            self.model_configs.build_model_configs(num_features=test_data['X_test'].shape[1]),
            self.config.TARGET_COLUMNS,
            self.config.COLUMNS_TO_TRANSFORM,
            model_dir=models_dir,
            sup_title="Test set Observed vs Predicted",
            log_transformer=self.log_transformer,
            group_label_map=group_label_map,
            group_prefix=group_prefix
        )
        os.makedirs(plots_dir, exist_ok=True)
        fig.savefig(os.path.join(plots_dir, "observed_vs_predicted.png"))   
        self.logger.info(f"Obs_vs_Pred Plot saved to {plots_dir}/observed_vs_predicted.png")

        # Display results
        print("\nFinal Metrics DataFrame:")
        print(metrics_df)

    def save_cv_results_and_plots(self, metrics_df, all_fold_preds):
        """Save CV metrics and plots for all targets and models."""
        metrics_dir = os.path.join(self.output_dir, "metrics")
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        metrics_df.to_csv(os.path.join(metrics_dir, "cv_folds_metrics.csv"), index=False)
        from helpers.plot_utils import plot_cv_folds_observed_vs_predicted
        group_label_map = load_dict_from_file(self.config.SOIL_GROUPS_FILE_PATH)
        group_prefix = self.config.CATEGORICAL_FEATURES[0] + "_"
        for target, fold_preds in all_fold_preds.items():
            if not fold_preds:
                continue
            fig = plot_cv_folds_observed_vs_predicted(
                fold_preds,
                target,
                sup_title="CV Folds Observed vs Predicted",
                group_prefix=group_prefix,
                group_label_map=group_label_map,
                columns_to_transform=self.config.COLUMNS_TO_TRANSFORM
            )
            fig.savefig(os.path.join(plots_dir, f"{target}_cv_folds.png"))
            self.logger.info(f"CV folds plot saved to {plots_dir}/{target}_cv_folds.png")
        print("\nCV Folds Metrics DataFrame:")
        print(metrics_df)
    

def main():
    # Initialize and run the training pipeline
    trainer = SoilModelTraining()
    
    try:
        # Load and preprocess data
        raw_data = trainer.data_manager.load_data()
        processed_data = trainer.data_manager.preprocess_data(raw_data)
        if trainer.config.CV_ONLY_MODE and trainer.config.CV_ONLY_MODE.get("enabled", False):
            trainer.logger.info("Running in CV_ONLY_MODE...")
            split_data = processed_data
        else:
            trainer.logger.info("Running in full training mode...")

            # Split data
            split_data = trainer.data_manager.split_data(processed_data)
        # Train models
        metrics, all_fold_preds = trainer.train_models(split_data)
        
        # Save results
        if not (trainer.config.CV_ONLY_MODE and trainer.config.CV_ONLY_MODE.get("enabled", False)):
            trainer.save_results(metrics, {'X_test': split_data['X_test'],
                                           'y_test': split_data['y_test']})
        else:
            trainer.save_cv_results_and_plots(metrics, all_fold_preds)
    

    except Exception as e:
        trainer.logger.error(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    main()