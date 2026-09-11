from yg_eo_soilnet.trainers.sklearn_trainer import ModelTrainer
from yg_eo_soilnet.logger.training_logger import TrainingLogger
from yg_eo_soilnet.models.config_fatories.model_config_factory import ModelConfigFactory
from yg_eo_soilnet.data_manager import DataManager
from yg_eo_soilnet.datamodules.scikit.scikit_datamodule import ScikitDataModule
from yg_eo_soilnet.datamodules.split_plan_provider import SplitPlanProvider
from yg_eo_soilnet.utils import LogTransformer
from yg_eo_soilnet.logger.mlflow_loggers import ParentRunLogger
from config import Config
from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningConfigFactory
from yg_eo_soilnet.seeding import seed_everything
from yg_eo_soilnet.trainers.lightning_trainer import LightningTrainer
from yg_eo_soilnet.tracking import (
    close_stale_runs,
    configure_tracking,
    install_run_signal_handlers,
    log_params_once,
    repair_corrupt_runs,
    resolve_local_tracking_root,
    run_owner_tags,
    tracking_settings,
)
from yg_eo_soilnet.targets import join_target_names, resolve_target_groups
import mlflow
import datetime
import logging
import time
import shutil
import re
from pathlib import Path
from typing import Any, Dict
import argparse


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


def _resolve_main_relative_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def _seed_everything(seed: int) -> None:
    """Kept as a thin alias so existing callers and tests keep working.

    Delegates to the shared helper, which also sets PL_SEED_WORKERS - this local version did not,
    so dataloader workers were seeded differently here than on the hyperparameter-search path.
    """
    seed_everything(seed)


def _export_mlflow_run_folder(trainer, experiment_id: str, run_id: str, run_name: str) -> Path | None:
    if not getattr(trainer.config, "MLFLOW_EXPERIMENT_EXPORT_ENABLED", False):
        return None

    experiment = mlflow.get_experiment(experiment_id)
    if experiment is None:
        return None

    tracking_root = resolve_local_tracking_root(mlflow.get_tracking_uri())
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
        # One split for the whole run, decided before either family touches the data. Held here
        # rather than inside a family so both get the SAME object: sklearn and Lightning used to
        # split independently, and a Lightning test point was usually an sklearn training point.
        self.split_plan_provider = SplitPlanProvider(self.config, self.logger, self.data_manager)
        self.scikit_datamodule = ScikitDataModule(
            self.config, self.logger, self.data_manager, split_plan_provider=self.split_plan_provider
        )
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

        # How several targets become models - one joint model with a wide head, or one model each -
        # is now MULTI_TARGET_MODE, and both families read the same answer. Resolved per family
        # because the sklearn side additionally needs the estimator to declare that it can fit a
        # 2-D y; a Lightning head always can.
        model_pipelines = self.model_configs.build_model_configs(
            num_features=data[X_key].shape[1],
            default_seed=self.config.RANDOM_SEED,
        )
        # Resolved per ENTRY, because joint capability is per estimator: PLS and Ridge take a 2-D y,
        # GradientBoosting and TabICL do not. An entry that cannot falls back to one model per
        # target with a warning rather than failing, so one unsupported estimator does not take the
        # whole run down.
        #
        # Entries that agree on a grouping are then trained TOGETHER, in one call per group. That
        # matters beyond tidiness: FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET is a check across the models
        # tried for a target, so splitting them into one call each would turn "every model failed"
        # into "any model failed".
        sklearn_groups: dict[tuple, dict] = {}
        for model_name, pipeline in (model_pipelines or {}).items():
            groups = resolve_target_groups(
                self.config,
                self.config.MODEL_REGISTRY.get(model_name, {}),
                require_joint_support=True,
                logger=self.logger,
                entry_name=model_name,
            )
            for target_group in groups:
                sklearn_groups.setdefault(tuple(target_group), {})[model_name] = pipeline

        # Same per-entry resolution on the Lightning side, so an entry may opt out of joint fitting
        # without changing the mode for the rest. Entries that agree share one build.
        lightning_groups: dict[tuple, list[str]] = {}
        for entry_name, spec in self.config.LIGHTNING_MODEL_REGISTRY.items():
            if not spec.get("enabled", False):
                continue
            for target_group in resolve_target_groups(self.config, spec):
                lightning_groups.setdefault(tuple(target_group), []).append(entry_name)

        # Logged on the PARENT run, before any child opens. Without it the run records how it split
        # the data but not what it decided to fit, so "was this joint or per-target?" could only be
        # guessed at from the child run names afterwards.
        self._log_target_plan(sklearn_groups, lightning_groups)

        for index, (target_group, pipelines) in enumerate(sklearn_groups.items(), start=1):
            label = join_target_names(list(target_group))
            # Progress, because a slow estimator spends tens of minutes per group with nothing to
            # say. A silent gap is indistinguishable from a hang, which is how an OOM-killed run got
            # mistaken for a bug in the grouping.
            self.logger.info(f"[sklearn group {index}/{len(sklearn_groups)}] {label} - starting")
            started = time.perf_counter()
            trainer.train(
                target=label,
                targets=list(target_group),
                data=data,
                model_pipelines=pipelines,
            )
            self.logger.info(
                f"[sklearn group {index}/{len(sklearn_groups)}] {label} - "
                f"done in {(time.perf_counter() - started) / 60:.1f}min"
            )

        for index, (target_group, entry_names) in enumerate(lightning_groups.items(), start=1):
            # `seed` so each model is constructed from a fixed RNG state rather than from
            # whatever the preceding data work and sklearn training left behind. Without it a
            # tuned config cannot reproduce the hyperparameter trial that produced it, and two
            # production runs do not agree with each other either.
            label = join_target_names(list(target_group))
            self.logger.info(f"[lightning group {index}/{len(lightning_groups)}] {label} - starting")
            started = time.perf_counter()
            def build_bundles(seed: int, _label=label, _entries=entry_names):
                """Bundles for this group at one seed. Called once per ensemble member.

                A Lightning model's weights are constructed by the factory, which seeds immediately
                beforehand - so the only way to get a second, differently-initialized member is to
                ask the factory again at a different seed. The dataset payload is cached inside the
                factory, so this re-seeds and rebuilds the model without re-deriving the data.
                """
                return self.lightning_model_configs.build_lightning_configs(
                    target=_label,
                    data=lightning_input,
                    seed=seed,
                    entries=_entries,
                )

            lightning_model_bundles = build_bundles(int(self.config.RANDOM_SEED))
            if lightning_model_bundles:
                self.lightning_trainer.train(
                    target=label,
                    data=lightning_input,
                    model_bundles=lightning_model_bundles,
                    bundle_builder=build_bundles,
                )
            self.logger.info(
                f"[lightning group {index}/{len(lightning_groups)}] {label} - "
                f"done in {(time.perf_counter() - started) / 60:.1f}min"
            )

    def _log_target_plan(self, sklearn_groups: Dict, lightning_groups: Dict) -> None:
        """Record what this run decided to fit, on the parent run.

        The mode and the groups TOGETHER are what identify a fallback: `MULTI_TARGET_MODE: joint`
        beside per-target sklearn groups means an estimator declined the 2-D fit, which is
        otherwise only visible as a warning in a log nobody kept.
        """
        def describe(groups) -> str:
            return " | ".join(join_target_names(list(group)) for group in groups) or "(none)"

        params = {
            "MULTI_TARGET_MODE": getattr(self.config, "MULTI_TARGET_MODE", "joint"),
            "TARGET_COLUMNS": ",".join(self.config.TARGET_COLUMNS) or "(none)",
            "sklearn_target_groups": describe(sklearn_groups),
            "lightning_target_groups": describe(lightning_groups),
        }
        self.logger.info(
            f"Target plan: mode={params['MULTI_TARGET_MODE']} | "
            f"sklearn={params['sklearn_target_groups']} | "
            f"lightning={params['lightning_target_groups']}"
        )
        try:
            # This is the sole writer of TARGET_COLUMNS on the parent run, and it runs at the START
            # of training so a killed run still records what it set out to fit.
            log_params_once(params, logger=self.logger)
        except Exception as exc:  # pragma: no cover - never worth failing a run over
            self.logger.warning(f"Could not log the target plan: {type(exc).__name__}: {exc}")


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
    # Before any run starts: an experiment's artifact_location is fixed when it is created, so this
    # is what keeps a run's metadata and its artifacts in the same directory.
    experiment_name = configure_tracking(tracking_settings(args.config_path))

    if mlflow.active_run():
        mlflow.end_run()

    # Before this run's own runs start, so the sweep cannot see them. Runs abandoned by a dead
    # process sit at RUNNING forever - an OOM kill is a SIGKILL, so nothing in the killed process
    # gets the chance to mark them. Ctrl-C and `kill` ARE catchable, hence the handlers too.
    #
    # Both take a logger, and both used to be called without one - so the sweep did its work in
    # complete silence. That is the whole diagnostic: an abandoned run is the one visible trace an
    # OOM kill leaves behind, and without this message the previous run just looks stuck. The
    # trainer's own logger does not exist yet (it is built inside the run below, and this has to
    # happen first), so a module logger stands in.
    startup_logger = logging.getLogger(__name__)
    install_run_signal_handlers(startup_logger)
    # Before the sweep, which cannot list runs at all while one of them has a truncated meta.yaml -
    # the same process death strands a run AND corrupts it, and the sweep swallows that failure as a
    # warning, so the store stays broken until something less forgiving reads it.
    repair_corrupt_runs(experiment_name, startup_logger)
    close_stale_runs(experiment_name, startup_logger)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Run_{timestamp}"
    mlflow_logger = ParentRunLogger()
    with mlflow.start_run(run_name=run_name) as main_run:
        # Who owns this run, so a later process can tell "finished" from "abandoned".
        mlflow.set_tags(run_owner_tags())
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

            # Decide the split ONCE, for every training family, before either of them touches the
            # data. Both then select their own rows out of it, so `rmse_test` means the same thing
            # on both sides of the leaderboard.
            stage_start = time.perf_counter()
            split_plan = trainer.split_plan_provider.plan()
            mlflow.log_params(split_plan.describe())
            trainer.logger.info(
                f"split plan built in {time.perf_counter() - stage_start:.2f}s | {split_plan.counts()}"
            )

            # Split data
            stage_start = time.perf_counter()
            split_data = trainer.scikit_datamodule.split(processed_data, split_plan)
            trainer.logger.info(
                f"split_data completed in {time.perf_counter() - stage_start:.2f}s | X_train={getattr(split_data.get('X_train'), 'shape', 'n/a')} | X_test={getattr(split_data.get('X_test'), 'shape', 'n/a')}"
            )

            stage_start = time.perf_counter()
            # Train models; the Lightning layer builds its own spatiotemporal graph on demand, but
            # reads the split plan carried in `split_data` rather than splitting for itself.
            trainer.train_models(split_data)
            trainer.logger.info(f"train_models completed in {time.perf_counter() - stage_start:.2f}s")

            # Not fatal, and deliberately narrower than the block below. By this point every model
            # has been trained and logged by its own child run; the summary only reads them back.
            # Letting it re-raise threw all of that away - the log artifact and the run-folder
            # export below never ran, and the run was marked FAILED - over a leaderboard. The tag
            # is what keeps this from being a silent swallow: a run missing its summary says why.
            try:
                mlflow_logger.log_parent_summary(main_run.info.run_id, trainer)
            except Exception as summary_error:
                trainer.logger.error(
                    f"Parent summary failed after training completed: "
                    f"{type(summary_error).__name__}: {summary_error}",
                    exc_info=True,
                )
                mlflow.set_tag(
                    "parent_summary_error",
                    f"{type(summary_error).__name__}: {summary_error}"[:500],
                )

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