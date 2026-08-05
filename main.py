from yg_eo_soilnet.trainers.sklearn_trainer import ModelTrainer
from yg_eo_soilnet.logger.training_logger import TrainingLogger
from yg_eo_soilnet.models.config_fatories.model_config_factory import ModelConfigFactory
from yg_eo_soilnet.data_manager import DataManager
from yg_eo_soilnet.datamodules.scikit.scikit_datamodule import ScikitDataModule
from yg_eo_soilnet.utils import LogTransformer
from yg_eo_soilnet.logger.mlflow_loggers import ParentRunLogger
from config import Config
from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningConfigFactory
from yg_eo_soilnet.trainers.lightning_trainer import LightningTrainer
import mlflow
import datetime 
import time
import shutil
import re
import random
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict
import argparse

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _describe_rows_cols(value: Any) -> str:
    rows = len(value) if hasattr(value, "__len__") else "n/a"
    columns = len(value.columns) if hasattr(value, "columns") else "n/a"
    return f"rows={rows} | cols={columns}"


def _describe_shapes(value: Any, x_key: str = "X", y_key: str = "y") -> str:
    x_shape = getattr(value.get(x_key), "shape", "n/a") if isinstance(value, dict) else "n/a"
    y_shape = getattr(value.get(y_key), "shape", "n/a") if isinstance(value, dict) else "n/a"
    return f"X_shape={x_shape} | y_shape={y_shape}"


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return sanitized or "experiment"


def _resolve_local_tracking_root(tracking_uri: str) -> Path | None:
    parsed = urlparse(tracking_uri)
    if parsed.scheme not in ("", "file"):
        return None
    if parsed.scheme == "file":
        return Path(parsed.path)
    return Path(tracking_uri)


def _resolve_main_relative_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - hardware dependent
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def _export_mlflow_run_folder(trainer, experiment_id: str, run_id: str, run_name: str) -> Path | None:
    if not getattr(trainer.config, "MLFLOW_EXPERIMENT_EXPORT_ENABLED", False):
        return None

    experiment = mlflow.get_experiment(experiment_id)
    if experiment is None:
        return None

    tracking_root = _resolve_local_tracking_root(mlflow.get_tracking_uri())
    if tracking_root is None:
        return None

    source_dir = tracking_root / experiment_id / run_id
    if not source_dir.exists():
        return None

    export_root = _resolve_main_relative_path(trainer.config.MLFLOW_EXPERIMENT_EXPORT_PATH)
    experiment_folder = _sanitize_path_component(experiment.name)
    run_folder = _sanitize_path_component(run_name)
    destination_dir = export_root / experiment_folder / run_folder
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)
    return destination_dir


class SoilModelTraining:
    def __init__(
        self,
        run_name: str = "Soil_Model_Training",
        config_path: str = "configs/main_config.yml",
    ):
        self.run_name = run_name
        # Initialize components
        self.config = Config(
            config_path=config_path,
        )
        _seed_everything(int(self.config.RANDOM_SEED))
        # Setup directories
        self.log_transformer = LogTransformer()
        self.logger_wrapper = TrainingLogger(
            name='AlMoutmir Soil Models Training',
            log_filename=self.run_name,
            enable_file_logging=self.config.MAIN_FILE_LOGGING_ENABLED,
        )
        self.logger = self.logger_wrapper.get_logger()
        self.sklearn_logger_wrapper = TrainingLogger(
            name='AlMoutmir Soil Models Training - sklearn',
            log_filename=f'{self.run_name}_sklearn',
            enable_file_logging=self.config.SKLEARN_FILE_LOGGING_ENABLED,
        )
        self.sklearn_logger = self.sklearn_logger_wrapper.get_logger()

        self.data_manager = DataManager(self.config, self.logger)
        self.scikit_datamodule = ScikitDataModule(self.config, self.logger, self.data_manager)
        # Spatial clustering splitter
        self.model_configs = ModelConfigFactory(self.config.MODEL_REGISTRY, random_state=self.config.RANDOM_SEED)
        self.lightning_model_configs = LightningConfigFactory(
            self.config.LIGHTNING_MODEL_REGISTRY,
            self.config,
            logger=self.logger,
            data_manager=self.data_manager,
        )
        self.lightning_trainer = LightningTrainer(config=self.config, logger=self.logger)

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
            logger=self.sklearn_logger,
        )
        lightning_input = dict(data)

        # Graph and sequence datamodules both span every point at once, so a multi-target run over
        # either is one combined run rather than one run per target.
        run_lightning_once = (
            len(self.config.TARGET_COLUMNS) > 1
            and self.lightning_model_configs.covers_all_targets_in_one_run()
        )
        combined_lightning_target = "__".join(self.config.TARGET_COLUMNS) if run_lightning_once else None

        for target in self.config.TARGET_COLUMNS:
            model_pipelines = self.model_configs.build_model_configs(
                num_features=data[X_key].shape[1],
                default_seed=self.config.RANDOM_SEED,
            )
            if model_pipelines:
                trainer.train(
                    target=target,
                    data=data,
                    model_pipelines=model_pipelines,
                )

            if not run_lightning_once:
                lightning_model_bundles = self.lightning_model_configs.build_lightning_configs(target=target, data=lightning_input)
                if lightning_model_bundles:
                    self.lightning_trainer.train(
                        target=target,
                        data=lightning_input,
                        model_bundles=lightning_model_bundles,
                    )

        if run_lightning_once and combined_lightning_target is not None:
            lightning_model_bundles = self.lightning_model_configs.build_lightning_configs(
                target=combined_lightning_target,
                data=lightning_input,
            )
            if lightning_model_bundles:
                self.lightning_trainer.train(
                    target=combined_lightning_target,
                    data=lightning_input,
                    model_bundles=lightning_model_bundles,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train soil models")
    parser.add_argument(
        "--config-path",
        default="configs/main_config.yml",
        help="Path to the main YAML config file",
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
        )
        mlflow.log_param("CONFIG_PATH", trainer.config.config_path)
        for param_name, param_value in (
            ("DATA_SPEC_PATH", getattr(trainer.config, "data_spec_path", None)),
            ("SKLEARN_CONFIG_PATH", getattr(trainer.config, "sklearn_config_path", None)),
            ("LIGHTNING_CONFIG_PATH", getattr(trainer.config, "lightning_config_path", None)),
        ):
            if param_value is not None:
                mlflow.log_param(param_name, param_value)
        mlflow.log_param("REGISTRY_PATH", trainer.config.registry_path)
        lightning_registry_path = getattr(trainer.config, "lightning_registry_path", None)
        if lightning_registry_path is not None:
            mlflow.log_param("LIGHTNING_REGISTRY_PATH", lightning_registry_path)
        try:
            # Load and preprocess data
            stage_start = time.perf_counter()
            raw_data = trainer.scikit_datamodule.load_frame()
            trainer.logger.info(
                f"load_dataset completed in {time.perf_counter() - stage_start:.2f}s | {_describe_rows_cols(raw_data)}"
            )

            stage_start = time.perf_counter()
            processed_data = trainer.scikit_datamodule.preprocess(raw_data)
            trainer.logger.info(
                "preprocess_data completed in "
                f"{time.perf_counter() - stage_start:.2f}s | {_describe_shapes(processed_data)}"
            )
            trainer.logger.info("Running in full training mode...")

            # Split data
            stage_start = time.perf_counter()
            split_data = trainer.scikit_datamodule.split(processed_data)
            trainer.logger.info(
                f"split_data completed in {time.perf_counter() - stage_start:.2f}s | X_train={getattr(split_data.get('X_train'), 'shape', 'n/a')} | X_test={getattr(split_data.get('X_test'), 'shape', 'n/a')}"
            )

            stage_start = time.perf_counter()
            # Train models; the Lightning layer builds its own spatiotemporal graph on demand
            trainer.train_models(split_data)
            trainer.logger.info(f"train_models completed in {time.perf_counter() - stage_start:.2f}s")

            mlflow_logger.log_parent_summary(main_run.info.run_id, trainer)

        except Exception as e:
            trainer.logger.error(f"An error occurred during training: {e}")
            raise
        if trainer.logger_wrapper.log_file:
            mlflow.log_artifact(trainer.logger_wrapper.log_file)

    experiment_id = getattr(main_run.info, "experiment_id", None)
    run_id = getattr(main_run.info, "run_id", None)
    if experiment_id is not None and run_id is not None:
        _export_mlflow_run_folder(trainer, experiment_id, run_id, run_name)

if __name__ == "__main__":
    main()