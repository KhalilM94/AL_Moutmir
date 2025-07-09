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
            cv_only_mode=self.config.CV_ONLY_MODE.get("enabled", False),
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
            self.logger.error("Metrics DataFrame is empty or missing 'target' column. Skipping result saving.")
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

        # --- Feature Importance Plots for ALL models in final_models ---
        from helpers.plot_utils import plot_feature_importances, plot_plsr_vip_and_biplot
        bands_csv_path = self.config.BANDS_CSV_PATH
        worldclim_csv_path = self.config.WORLDCLIM_CSV_PATH
        for model_file in os.listdir(models_dir):
            if not model_file.endswith('.pkl'):
                continue
            model_path = os.path.join(models_dir, model_file)
            # Extract target and model name from filename
            base = os.path.splitext(model_file)[0]
            if '_' in base:
                target, model_name = base.rsplit('_', 1)
            else:
                target, model_name = base, ''
            print(f"[DEBUG] Attempting plot for model: {model_name} at {model_path}")
            if target not in test_data['X_test'].columns and target not in test_data['y_test']:
                print(f"[DEBUG] Skipping {model_file}: target not in test data.")
                continue
            try:
                if "plsregression" in model_name.lower():
                    print(f"[DEBUG] Detected PLSRegression model. Calling plot_plsr_vip_and_biplot...")
                    fig = plot_plsr_vip_and_biplot(
                        model_path,
                        test_data['X_test'],
                        bands_csv_path=bands_csv_path,
                        worldclim_csv_path=worldclim_csv_path,
                        top_n=20,
                        title=f"PLSR VIP & Biplot: {target} - {model_name}"
                    )
                    fig.savefig(os.path.join(plots_dir, f"{target}_{model_name}_plsr_vip_biplot.png"))
                    print(f"[DEBUG] Saved PLSR VIP & biplot for {model_name}")
                    self.logger.info(f"PLSR VIP & biplot saved to {plots_dir}/{target}_{model_name}_plsr_vip_biplot.png")
                else:
                    print(f"[DEBUG] Detected non-PLSR model. Calling plot_feature_importances...")
                    fig = plot_feature_importances(
                        model_path,
                        test_data['X_test'],
                        bands_csv_path=bands_csv_path,
                        worldclim_csv_path=worldclim_csv_path,
                        top_n=20,
                        title=f"Top 20 Features: {target} - {model_name}"
                    )
                    fig.savefig(os.path.join(plots_dir, f"{target}_{model_name}_feature_importance.png"))
                    print(f"[DEBUG] Saved feature importance plot for {model_name}")
                    self.logger.info(f"Feature importance plot saved to {plots_dir}/{target}_{model_name}_feature_importance.png")
            except Exception as e:
                self.logger.warning(f"Could not plot feature importances for {target} - {model_name}: {e}")

        # --- Plot train/test histograms for all targets ---
        from helpers.plot_utils import plot_train_test_histograms
        if 'y_train' in test_data and 'y_test' in test_data:
            plot_train_test_histograms(
                test_data['y_train'],
                test_data['y_test'],
                self.config.TARGET_COLUMNS,
                plots_dir,
                filename="train_test_histograms_overlayed.png",
                orientation="vertical"
            )

        # Display results
        self.logger.info("Training results:")
        print(metrics_df)

    def save_cv_results_and_plots(self, metrics_df, all_fold_preds):
        """Save CV metrics and plots for all targets and models."""
        metrics_dir = os.path.join(self.output_dir, "metrics")
        plots_dir = os.path.join(self.output_dir, "plots")
        group_label_map = load_dict_from_file(self.config.SOIL_GROUPS_FILE_PATH)
        group_prefix = self.config.CATEGORICAL_FEATURES[0] + "_"
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        metrics_df.to_csv(os.path.join(metrics_dir, "cv_folds_metrics.csv"), index=False)
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
        
        # --- Feature Importance Plots for ALL models in final_models (CV mode) ---
        from helpers.plot_utils import plot_feature_importances, plot_plsr_vip_and_biplot
        bands_csv_path = self.config.BANDS_CSV_PATH
        worldclim_csv_path = self.config.WORLDCLIM_CSV_PATH
        models_dir = os.path.join(self.output_dir, "final_models")
        plots_dir = os.path.join(self.output_dir, "plots")
        X = None
        # Try to get a representative X (features) from any fold_preds
        for fold_preds in all_fold_preds.values():
            for fold_pred in fold_preds:
                if 'X_val' in fold_pred and fold_pred['X_val'] is not None:
                    X = fold_pred['X_val'] if not isinstance(fold_pred['X_val'], list) else pd.DataFrame(fold_pred['X_val'])
                    break
            if X is not None:
                break
        if X is not None:
            for model_file in os.listdir(models_dir):
                if not model_file.endswith('.pkl'):
                    continue
                model_path = os.path.join(models_dir, model_file)
                base = os.path.splitext(model_file)[0]
                # Robustly parse: {target}_{model}.pkl or {target}_{model}_{fold}.pkl
                parts = base.split('_')
                if len(parts) >= 3 and parts[-1].isdigit():
                    fold = parts[-1]
                    model_name = parts[-2]
                    target = '_'.join(parts[:-2])
                elif len(parts) >= 2:
                    fold = None
                    model_name = parts[-1]
                    target = '_'.join(parts[:-1])
                else:
                    fold = None
                    model_name = base
                    target = base
                print(f"[DEBUG] (CV) Attempting plot for model: {model_name} at {model_path} (fold: {fold})")
                try:
                    if "plsregression" in model_name.lower():
                        print(f"[DEBUG] (CV) Detected PLSRegression model. Calling plot_plsr_vip_and_biplot...")
                        fig = plot_plsr_vip_and_biplot(
                            model_path,
                            X,
                            bands_csv_path=bands_csv_path,
                            worldclim_csv_path=worldclim_csv_path,
                            top_n=20,
                            title=f"PLSR VIP & Biplot: {target} - {model_name} (CV{f'-Fold {fold}' if fold else ''})"
                        )
                        out_name = f"{target}_{model_name}_plsr_vip_biplot_cv{f'_fold{fold}' if fold else ''}.png"
                        fig.savefig(os.path.join(plots_dir, out_name))
                        print(f"[DEBUG] (CV) Saved PLSR VIP & biplot for {model_name} (fold: {fold})")
                        self.logger.info(f"PLSR VIP & biplot (CV) saved to {plots_dir}/{out_name}")
                    else:
                        print(f"[DEBUG] (CV) Detected non-PLSR model. Calling plot_feature_importances...")
                        fig = plot_feature_importances(
                            model_path,
                            X,
                            bands_csv_path=bands_csv_path,
                            worldclim_csv_path=worldclim_csv_path,
                            top_n=20,
                            title=f"Top 20 Features: {target} - {model_name} (CV{f'-Fold {fold}' if fold else ''})"
                        )
                        out_name = f"{target}_{model_name}_feature_importance_cv{f'_fold{fold}' if fold else ''}.png"
                        fig.savefig(os.path.join(plots_dir, out_name))
                        print(f"[DEBUG] (CV) Saved feature importance plot for {model_name} (fold: {fold})")
                        self.logger.info(f"Feature importance plot (CV) saved to {plots_dir}/{out_name}")
                except Exception as e:
                    self.logger.warning(f"Could not plot feature importances (CV) for {target} - {model_name} (fold: {fold}): {e}")
        self.logger.info("CV results and plots saved successfully.")
        self.logger.info("Metrics DataFrame:")
        # Display the metrics DataFrame
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
            trainer.save_results(metrics, split_data)
        else:
            trainer.save_cv_results_and_plots(metrics, all_fold_preds)
    

    except Exception as e:
        trainer.logger.error(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    main()