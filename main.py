from helpers.model_trainer import ModelTrainer
from helpers.logging import TrainingLogger
from helpers.utils import SpatialClusterSplitter
from sklearn.model_selection import GroupShuffleSplit, train_test_split
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

        # Setup directories
        self.output_dir = self._setup_directories()
        # Initialize components
        self.config = Config()
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

        # Spatial clustering splitter
        self.cluster_splitter = SpatialClusterSplitter(random_state= self.config.RANDOM_SEED)

    @staticmethod    
    def _setup_directories(output_dir: str = "output") -> str:
        """Ensure required directories exist."""
        # Create base output directory
        """Create a unique output directory with subdirectories for final models and metrics."""
        # Create a unique directory name using timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_output_dir = f"{output_dir}_{timestamp}"
        output_path = os.path.abspath(unique_output_dir)

        # Create subdirectories
        final_models_path = os.path.join(output_path, "final_models")
        metrics_path = os.path.join(output_path, "metrics")

        os.makedirs(final_models_path, exist_ok=True)
        os.makedirs(metrics_path, exist_ok=True)

        return output_path
        
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

    def load_data(self) -> pd.DataFrame:
        """Load and return the initial dataset."""
        csv_files = [f for f in os.listdir(self.config.DATA_FOLDER) if f.endswith('.csv')]
        self.logger.info(f"Found data files: {csv_files}")
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.config.DATA_FOLDER}")
            
        return pd.read_csv(os.path.join(self.config.DATA_FOLDER, csv_files[0]))
        
    def preprocess_data(self, data: pd.DataFrame) -> Dict:
        """Preprocess the data and return prepared datasets."""
        self.logger.info("Clustering dataset based on spatial location (KMeans)...")
        clustered_data = self.cluster_splitter.cluster(data)
        self.logger.info(f"Cluster value counts:\n{clustered_data['cluster'].value_counts().to_string()}")
        
        # Get feature columns
        feature_columns = [col for col in clustered_data.columns 
                          if col not in self.config.TARGET_COLUMNS + self.config.ELIMINATED_FEATURES]
        
        # Prepare feature matrix
        valid_feature_columns = clustered_data[feature_columns].dropna(axis=1, how='all').columns.tolist()
        X = clustered_data[valid_feature_columns]
        
        # One-hot encoding if needed
        if 'SU_WRB1_PH' in X.columns:
            self.logger.info("One-hot encoding 'SU_WRB1_PH' column...")
            X = pd.get_dummies(X, columns=['SU_WRB1_PH'])
            
        # Fill remaining NaNs with column means
        X = X.apply(lambda row: row.fillna(row.mean()), axis=1)
        clustered_cleaned = clustered_data.loc[X.index]
        
        return {
            'X': X,
            'y': clustered_cleaned[self.config.TARGET_COLUMNS],
            'groups': clustered_cleaned['cluster']
        }
        
    def split_data(self, X: pd.DataFrame, y: pd.DataFrame, groups: pd.Series) -> Dict:
        """Split data into training and test sets."""
        if self.config.USE_GROUP_SPLIT:
            self.logger.info("Splitting dataset into train and test groups using GroupShuffleSplit...")
            
            gss = GroupShuffleSplit(n_splits=1, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_SEED)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]
            
            self.logger.info(f"Train group distribution:\n{groups_train.value_counts().sort_index().to_string()}")
            self.logger.info(f"Test group distribution:\n{groups_test.value_counts().sort_index().to_string()}")
        else:
            self.logger.info("Splitting dataset using simple train-test split...")
            
            X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
                X, y, groups, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_SEED
            )
            
        self.logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'groups_train': groups_train,
            'groups_test': groups_test
        }
        
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
        # Define output paths
        metrics_dir = os.path.join(self.output_dir, "metrics")
        models_dir = os.path.join(self.output_dir, "final_models")

        # Save metrics
        top_models = metrics_df.groupby("target").apply(
            lambda df: df.sort_values("Test_R2", ascending=False).head(1)
        ).reset_index(drop=True)

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
        raw_data = trainer.load_data()
        processed_data = trainer.preprocess_data(raw_data)
        
        # Split data
        split_data = trainer.split_data(
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