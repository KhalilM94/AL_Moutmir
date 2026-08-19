"""Recover a trained model from its checkpoint, without retraining.

A run can train perfectly and still fail to package: the weights are on disk, the metrics are
logged, and yet no model was saved and nothing reached the registry. Retraining to recover a model
that already exists is the wrong trade, and it is unnecessary - a checkpoint written by this project
is self-describing. Besides the weights it carries the hyper-parameters and the fitted
``preprocessing_state``: the scalers, the categorical vocabularies and every feature-name list. So
the model can be rebuilt, packaged and registered with no dataset present at all.

    python relog.py \\
        --checkpoint lightning_logs/version_119/checkpoints/epoch=70-step=3763.ckpt \\
        --run-id 80849677a7d14c41b622726b95a876c7

The model is logged into the ORIGINAL run rather than a fresh one, because champion promotion reads
``rmse_test`` from the run behind a version - the score that justifies a model has to sit with it.

The one thing a checkpoint cannot tell us is its own class: it records ``hparams_name`` but no
import path. That is resolved from the run's ``model_name`` tag against the Lightning registry, with
``--model-class`` as the override.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any

import mlflow

from config import Config
from yg_eo_soilnet.artifacts import ArtifactLayout, log_json
from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger
from yg_eo_soilnet.tracking import CHAMPION_METRIC, configure_tracking_uri, promote_if_better

RELOG_SUMMARY_FILE = "relog_summary.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to the .ckpt to rebuild from")
    parser.add_argument("--run-id", required=True, help="The run the model belongs to")
    parser.add_argument(
        "--model-class",
        default=None,
        help="Dotted path to the LightningModule; inferred from the run's model_name tag otherwise",
    )
    parser.add_argument(
        "--config-path", default="configs/main_config.yml", help="Main config, for the registry lookup"
    )
    parser.add_argument("--no-register", action="store_true", help="Log the model without registering it")
    parser.add_argument("--rows", type=int, default=3, help="Rows in the input example")
    return parser.parse_args(argv)


def resolve_model_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def infer_model_class_path(config, model_name: str) -> str:
    """The import path for a registry entry, so the checkpoint's architecture is not guesswork."""
    entry = (getattr(config, "LIGHTNING_MODEL_REGISTRY", None) or {}).get(model_name)
    import_path = (entry or {}).get("import_path")
    if not import_path:
        raise SystemExit(
            f"Cannot infer the model class: the run's model_name is {model_name!r}, which is not in "
            f"the Lightning registry. Pass --model-class explicitly."
        )
    return str(import_path)


def load_static_frame(run_id: str, rows: int):
    """The run's own test features, when its eval CSV is still reachable.

    Only the static block: that artifact never carried the lab roster or the ragged sequences, so
    those are synthesized regardless. Absent artifact is not an error - the example falls back to
    the checkpoint's stored training means.
    """
    import pandas as pd

    from yg_eo_soilnet.logger.mlflow_loggers import _eval_results_artifact_paths

    for artifact_path in _eval_results_artifact_paths("", ""):
        try:
            local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
            return pd.read_csv(local, nrows=max(1, rows))
        except Exception:
            continue
    return None


def relog(args: argparse.Namespace) -> dict[str, Any]:
    config = Config(config_path=args.config_path)
    # The URI only. Switching to the config's experiment would make start_run(run_id=...) fail
    # whenever the run belongs to a different one - a run is reached by id, not by experiment.
    configure_tracking_uri(config)

    client = mlflow.MlflowClient()
    run = client.get_run(args.run_id)
    # Align the active experiment with the RUN's own, which is what start_run(run_id=...) requires.
    mlflow.set_experiment(experiment_id=run.info.experiment_id)
    tags = run.data.tags
    model_name = tags.get("model_name") or "model"
    target = tags.get("target") or "target"

    class_path = args.model_class or infer_model_class_path(config, model_name)
    module_class = resolve_model_class(class_path)

    # load_from_checkpoint calls on_load_checkpoint, which restores preprocessing_state - the
    # scalers and vocabulary without which the model cannot consume raw data.
    model = module_class.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()

    state = model.get_preprocessing_state()
    if not state:
        raise SystemExit(
            f"{args.checkpoint} carries no preprocessing state, so the model cannot standardize raw "
            "input and is not servable. It predates attach_preprocessing_state; retrain instead."
        )

    from yg_eo_soilnet.serving.lightning_pyfunc import example_from_state

    input_example = example_from_state(
        state,
        static_frame=load_static_frame(args.run_id, args.rows),
        n_rows=args.rows,
        auxiliary_columns=list(getattr(model, "auxiliary_label_columns", None) or []),
    )

    logger = ChildRunLogger()
    register = not args.no_register

    with mlflow.start_run(run_id=args.run_id):
        logged = logger._log_lightning_serialized_model(
            model=model,
            model_name=model_name,
            target=target,
            bundle=None,
            best_model_path=args.checkpoint,
            config=type("_C", (), {"MLFLOW_REGISTER_MODELS": register})(),
            input_example=input_example,
        )
        version = getattr(logger, "_registered_version", None)
        logger._tag_model_logging(target, model_name, logged, None if logged else "relog failed")

        champion = {"promoted": False, "reason": "registration was disabled"}
        if register:
            champion = promote_if_better(
                ArtifactLayout.logged_model_name(target, model_name),
                version,
                run.data.metrics.get(CHAMPION_METRIC),
            )

        summary = {
            "recovered_from_checkpoint": args.checkpoint,
            "model_class": class_path,
            "run_id": args.run_id,
            "target": target,
            "model_name": model_name,
            "logged_model_name": ArtifactLayout.logged_model_name(target, model_name),
            "registered_model_version": version,
            CHAMPION_METRIC: run.data.metrics.get(CHAMPION_METRIC),
            "champion": champion,
        }
        # Beside the original run summary, not over it: that file records the packaging failure,
        # and erasing the evidence of what went wrong would be worse than a second file.
        log_json(summary, RELOG_SUMMARY_FILE, ArtifactLayout.META)

    return summary


def main(argv: list[str] | None = None) -> int:
    summary = relog(parse_args(argv))
    print(json.dumps(summary, indent=2, default=str))
    if summary["registered_model_version"] is None and summary["champion"].get("reason") != (
        "registration was disabled"
    ):
        print("\nThe model was logged but NOT registered.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
