"""Backfill the per-point prediction export onto a run that has already finished.

The export - one file answering "what did every model predict for this point?" - is written during
training, behind ``export_point_predictions.enabled``. Every run trained before that switch existed
has no such file, and retraining to get one is the wrong trade when the fitted models are still
sitting in the run.

    python export_predictions.py --parent-run-id 7d09124142a74a079a11be68a19268a8

What makes this possible is that each child run kept its model: a sklearn child records a loadable
``models:/m-...`` URI in its run summary, and a Lightning child's ``checkpoints/best.ckpt`` is
self-describing - it carries the hyper-parameters and the fitted ``preprocessing_state``, the same
property ``relog.py`` depends on. The features are rebuilt from the configured source data and the
result is checked against what the run itself recorded before a single model is loaded.

A sklearn ensemble needs no special handling: the whole ``EnsembleRegressor`` is one logged object
whose ``predict`` already returns the mean.

A **Lightning ensemble** does. Runs trained before members logged their own weights kept only the
reference member's checkpoint in MLflow; the other members exist solely as local files under
``lightning_logs``. Those are matched back to their member runs by **validation loss** - a property
of the fit, recorded on both sides, and distinct between members of the same target - never by file
time or by version ordering, both of which are accidents of when the run happened. Every member has
to be found: averaging the subset that survived and calling it the ensemble would be neither one
member's prediction nor the ensemble's. Pass ``--no-member-recovery`` to skip them instead.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd

from config import Config
from yg_eo_soilnet.artifacts import ArtifactLayout, log_json
from yg_eo_soilnet.logger import TrainingLogger
from yg_eo_soilnet.logger.mlflow_loggers import (
    MEMBER_RUN_KIND,
    ChildRunLogger,
    ParentRunLogger,
    _scoring_runs,
)
from yg_eo_soilnet.predictions_export import point_id_column
from yg_eo_soilnet.targets import split_target_names
from yg_eo_soilnet.tracking import configure_tracking_uri

BACKFILL_SUMMARY_FILE = "backfill_summary.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--parent-run-id", required=True, help="The finished parent run to backfill")
    parser.add_argument(
        "--config-path",
        default="configs/main_config.yml",
        help="Main config, used to rebuild the features and to resolve model classes",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated registry entries to export; default is every child the run trained",
    )
    parser.add_argument(
        "--skip-models",
        default=None,
        help="Comma-separated entries to leave out; defaults to the config's skip list",
    )
    parser.add_argument(
        "--allow-population-drift",
        action="store_true",
        help=(
            "Continue when the rebuilt population differs from the one the run recorded, instead "
            "of stopping. Use it to pick up points added to the dataset since the run; the "
            "feature-schema check still refuses, because different columns mean the models cannot "
            "predict at all."
        ),
    )
    parser.add_argument(
        "--member-checkpoint-dir",
        default="lightning_logs",
        help=(
            "Where to look for a Lightning ensemble's member checkpoints when the member runs did "
            "not log their own. Matched on validation loss, never on file times."
        ),
    )
    parser.add_argument(
        "--no-member-recovery",
        action="store_true",
        help="Do not scavenge member checkpoints; Lightning ensembles are then skipped.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Rebuild, check for drift and report what would be exported, writing nothing",
    )
    return parser.parse_args(argv)


# --- data ------------------------------------------------------------------


def rebuild_features(config, logger) -> tuple[pd.DataFrame, pd.Series]:
    """The full featurized population and its point ids, rebuilt from the configured source.

    Deliberately NOT ``ScikitDataModule.prepare()``. That builds a split plan and its splitter calls
    ``mlflow.log_artifacts``, which would write ``data_splits/`` into whichever run is active - and
    this command reopens a FINISHED run, so it would overwrite that run's own split artifacts with
    a fresh split. Prediction needs no split at all, so the splitter is skipped entirely.

    ``filter_schema`` is not optional: it is applied inside ``split_data``, not by the preprocessor,
    so it is the step that turns the preprocessor's X into the columns the models were actually
    fitted on. It drops columns only, which is what keeps ``point_ids`` aligned by index.
    """
    from yg_eo_soilnet.data_manager import DataManager
    from yg_eo_soilnet.datamodules.scikit.tabular_preprocessor import TabularPreprocessor

    data_manager = DataManager(config, logger)
    frame = data_manager.load_dataset().tabular
    processed = TabularPreprocessor(config, logger, data_manager).preprocess_data(frame)

    features = data_manager.filter_schema(processed["X"], list(config.TARGET_COLUMNS))
    # The same conversion ModelTrainer applies before fitting, so the estimators see the dtypes they
    # were trained with rather than integer columns a pipeline may treat differently.
    features = features.astype(
        {column: "float64" for column in features.select_dtypes(include=["int64", "int32"]).columns}
    )
    point_ids = pd.Series(
        np.asarray(processed["point_ids"]), index=features.index, name="point_id"
    )
    logger.info(f"Rebuilt {len(features)} points x {features.shape[1]} features from source.")
    return features, point_ids


def _download(run_id: str, artifact_path: str):
    try:
        return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    except Exception:
        return None


def check_for_drift(
    parent_run_id: str,
    features: pd.DataFrame,
    point_ids: pd.Series,
    *,
    allow_population_drift: bool,
    logger,
) -> dict[str, Any]:
    """Compare the rebuilt data against what the run recorded, before any model is loaded.

    Cheap, and it runs first for that reason: discovering a schema change after loading and running
    every model wastes the expensive part of the command.

    The feature-column check never relaxes. Models fitted on a different column set cannot
    legitimately predict on this one - sklearn would eventually raise anyway, from somewhere deep in
    a ColumnTransformer, and the message would not say which column moved.
    """
    report: dict[str, Any] = {"checked": False}

    split_path = _download(parent_run_id, f"{ArtifactLayout.DATA_SPLITS}/split_assignments.parquet")
    test_path = _download(parent_run_id, f"{ArtifactLayout.DATA_SPLITS}/X_test.parquet")
    if split_path is None or test_path is None:
        # Older runs wrote data_splits without a point_id column and with no split_assignments at
        # all, so there is nothing to check the rebuilt data against. Strict means "prove it
        # matches", and an unverifiable run cannot - so it stops here rather than exporting numbers
        # nobody can tie back to what the run actually trained on.
        message = (
            "This run logged no usable data_splits/ artifacts (no split_assignments.parquet or no "
            "X_test.parquet), so the rebuilt data cannot be checked against what it trained on."
        )
        if not allow_population_drift:
            raise SystemExit(
                message + "\nPass --allow-population-drift to export anyway, unverified."
            )
        logger.warning(message + " Continuing unverified because --allow-population-drift was passed.")
        return report

    recorded_columns = [c for c in pd.read_parquet(test_path).columns if c != "point_id"]
    rebuilt_columns = list(features.columns)
    missing = [c for c in recorded_columns if c not in rebuilt_columns]
    extra = [c for c in rebuilt_columns if c not in recorded_columns]

    recorded_points = set(pd.read_parquet(split_path)["point_id"])
    rebuilt_points = set(point_ids)
    added = rebuilt_points - recorded_points
    removed = recorded_points - rebuilt_points

    report = {
        "checked": True,
        "n_recorded_points": len(recorded_points),
        "n_rebuilt_points": len(rebuilt_points),
        "n_points_added": len(added),
        "n_points_removed": len(removed),
        "missing_columns": missing,
        "extra_columns": extra,
    }

    if missing or extra:
        raise SystemExit(
            "The rebuilt features do not match the columns this run was trained on, so its models "
            "cannot predict on them.\n"
            f"  missing: {', '.join(missing[:10]) or '(none)'}\n"
            f"  extra:   {', '.join(extra[:10]) or '(none)'}\n"
            "The config or the source data has changed since the run."
        )

    if added or removed:
        message = (
            f"The rebuilt population differs from the run's: {len(added)} point(s) added, "
            f"{len(removed)} removed."
        )
        if added and not removed:
            # Very often this is not drift at all. A run whose split used population_policy:
            # intersect recorded only the points EVERY active family could use, and the sequence
            # builder drops rows with non-finite statics that the tabular preprocessor keeps - so a
            # Lightning run's recorded population is legitimately narrower than a tabular rebuild of
            # the same data. Points that vanished are the alarming direction; points that appeared
            # usually mean this.
            message += (
                " Nothing was removed, so this is most likely the split's population_policy rather "
                "than changed data: a run whose families disagreed about usable rows records only "
                "the intersection."
            )
        if not allow_population_drift:
            raise SystemExit(
                message + "\nThe exported predictions would describe a different set of points "
                "than the run did. Pass --allow-population-drift to export anyway."
            )
        logger.warning(message + " Continuing because --allow-population-drift was passed.")

    return report


# --- per child -------------------------------------------------------------


def _export_config(config, args: argparse.Namespace):
    """A copy of the config with the export forced on, plus the CLI's model filters.

    Running this command IS the opt-in - the config switch governs training runs, and a user who
    typed the command has already asked for the export. FAIL_ON_ERROR is forced too so a child that
    cannot be exported raises here, where this command can attribute it, rather than returning a
    quiet error dict.
    """
    proxy = copy.copy(config)
    proxy.EXPORT_POINT_PREDICTIONS = True
    proxy.EXPORT_POINT_PREDICTIONS_FAIL_ON_ERROR = True
    if args.models is not None:
        proxy.EXPORT_POINT_PREDICTIONS_MODELS = [n.strip() for n in args.models.split(",") if n.strip()]
    else:
        proxy.EXPORT_POINT_PREDICTIONS_MODELS = []
    if args.skip_models is not None:
        proxy.EXPORT_POINT_PREDICTIONS_SKIP_MODELS = [
            n.strip() for n in args.skip_models.split(",") if n.strip()
        ]
    return proxy


def _run_summary(run_id: str) -> dict:
    path = _download(run_id, f"{ArtifactLayout.META}/{ArtifactLayout.RUN_SUMMARY_FILE}")
    if path is None:
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# --- recovering a Lightning ensemble's members -----------------------------


def _version_candidates(checkpoint_dir: str, target_names: list[str]) -> list[dict]:
    """Version directories whose ``hparams.yaml`` declares exactly this target group.

    The yaml rather than the checkpoint: it carries the same ``target_names`` for a fraction of the
    I/O, and these checkpoints are ~75 MB each. Note ``fitted_target_names`` in the same file lists
    every target the RUN fitted and so discriminates nothing - ``target_names`` is the model's own
    output roster and is the one that identifies it.
    """
    import glob

    import yaml

    wanted = [str(name) for name in target_names]
    candidates = []
    for hparams_path in sorted(glob.glob(os.path.join(checkpoint_dir, "version_*", "hparams.yaml"))):
        version_dir = os.path.dirname(hparams_path)
        checkpoints = glob.glob(os.path.join(version_dir, "checkpoints", "*.ckpt"))
        if not checkpoints:
            continue
        try:
            with open(hparams_path, encoding="utf-8") as handle:
                hparams = yaml.safe_load(handle) or {}
        except Exception:
            continue
        if [str(name) for name in (hparams.get("target_names") or [])] != wanted:
            continue
        candidates.append(
            {
                "version_dir": version_dir,
                "checkpoint": sorted(checkpoints)[0],
                "val_loss": _min_val_loss_from_csv(os.path.join(version_dir, "metrics.csv")),
            }
        )
    return candidates


def _min_val_loss_from_csv(metrics_path: str) -> Optional[float]:
    if not os.path.isfile(metrics_path):
        return None
    try:
        frame = pd.read_csv(metrics_path)
    except Exception:
        return None
    if "val_loss" not in frame.columns:
        return None
    values = frame["val_loss"].dropna()
    return float(values.min()) if len(values) else None


def _min_val_loss_from_run(client, run_id: str) -> Optional[float]:
    try:
        history = client.get_metric_history(run_id, "val_loss")
    except Exception:
        return None
    return min((point.value for point in history), default=None)


def match_member_checkpoints(client, child_run, checkpoint_dir: str, target_names: list[str]) -> list[dict]:
    """Match each of a Lightning ensemble's member runs to the checkpoint it actually wrote.

    Matched on ``min(val_loss)``, NOT on file times or on version ordering. Both of those are
    accidents here: a run's end_time is rewritten every time it is reopened, and the version numbers
    are only contiguous because nothing else happened to train in between. The validation loss is a
    property of the fit itself, it is recorded on both sides, and within a target the members'
    values are distinct - so it identifies a checkpoint by what it is.

    Returns one record per member, in member order. Raises SystemExit only on an ambiguous match;
    a member that simply cannot be found is returned with ``checkpoint: None`` so the caller can
    report how many of how many were recovered.
    """
    members = [
        run
        for run in client.search_runs(
            experiment_ids=[child_run.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{child_run.info.run_id}'",
        )
        if run.data.tags.get("run_kind") == MEMBER_RUN_KIND
    ]
    members.sort(key=lambda run: int(run.data.params.get("ensemble_member", 0)))

    candidates = _version_candidates(checkpoint_dir, target_names)
    matched: list[dict] = []
    used: set[str] = set()

    for member in members:
        # A member that logged its own checkpoint - every run trained after the structural fix -
        # needs none of this. Take it straight from MLflow.
        logged = _download(
            member.info.run_id, f"{ArtifactLayout.CHECKPOINTS}/{ArtifactLayout.CHECKPOINT_FILE}"
        )
        index = int(member.data.params.get("ensemble_member", len(matched)))
        record = {
            "ensemble_member": index,
            "ensemble_seed": member.data.params.get("ensemble_seed"),
            "run_id": member.info.run_id,
        }
        if logged is not None:
            record.update({"checkpoint": logged, "source": "mlflow"})
            matched.append(record)
            continue

        target_loss = _min_val_loss_from_run(client, member.info.run_id)
        hits = [
            candidate
            for candidate in candidates
            if candidate["version_dir"] not in used
            and candidate["val_loss"] is not None
            and target_loss is not None
            and np.isclose(candidate["val_loss"], target_loss, rtol=0.0, atol=1e-9)
        ]
        if len(hits) > 1:
            raise SystemExit(
                f"Member {index} of {child_run.info.run_id} matches {len(hits)} checkpoints on "
                f"val_loss={target_loss}. Refusing to guess which is which."
            )
        if hits:
            used.add(hits[0]["version_dir"])
            record.update(
                {
                    "checkpoint": hits[0]["checkpoint"],
                    "version_dir": hits[0]["version_dir"],
                    "val_loss": hits[0]["val_loss"],
                    "source": "lightning_logs",
                }
            )
        else:
            record.update({"checkpoint": None, "val_loss": target_loss})
        matched.append(record)

    return matched


def sklearn_predictor(run, features: pd.DataFrame):
    """``(predict_callable, n_expected)`` for a sklearn child, from its logged model.

    Two URI forms, in this order. ``runs:/<run_id>/<logged_model_name>`` needs nothing but the run's
    own tags, so it still works when ``meta/run_summary.json`` is missing or predates the field.
    The ``models:/m-...`` id recorded in that summary is the fallback for a run whose logged name no
    longer resolves.
    """
    import mlflow.sklearn

    target = run.data.tags.get("target") or ""
    model_name = run.data.tags.get("model_name") or ""
    candidates = [
        f"runs:/{run.info.run_id}/{ArtifactLayout.logged_model_name(target, model_name)}",
    ]
    recorded = _run_summary(run.info.run_id).get("logged_model_uri")
    if recorded:
        candidates.append(str(recorded))

    errors = []
    for uri in candidates:
        try:
            model = mlflow.sklearn.load_model(uri)
            return (lambda: model.predict(features)), len(features)
        except Exception as exc:
            errors.append(f"{uri}: {type(exc).__name__}: {exc}")

    raise SystemExit(
        f"Could not reload the fitted model for run {run.info.run_id}. Tried:\n  "
        + "\n  ".join(errors)
    )


_BUNDLE_CACHE: dict[str, Any] = {}


def sequence_bundle_for(model_name: str, config, logger):
    """The sequence bundle for one registry entry, built once per invocation.

    Cached because the bundle depends on the config and the data, never on the model: an ensembled
    entry would otherwise rebuild the identical object once per member, which for six children of
    five members is thirty builds of the same thing.
    """
    if model_name not in _BUNDLE_CACHE:
        from yg_eo_soilnet.data_manager import DataManager
        from yg_eo_soilnet.datamodules.sequence.sequence_builder import SoilSequenceBuilder

        entry = (getattr(config, "LIGHTNING_MODEL_REGISTRY", None) or {}).get(model_name, {})
        logger.info(f"Building the sequence bundle for {model_name} (once, then reused).")
        _BUNDLE_CACHE[model_name] = SoilSequenceBuilder(
            config, logger, DataManager(config, logger)
        ).build(sequence_data_args=dict(entry.get("sequence_data_args", {}) or {}))
    return _BUNDLE_CACHE[model_name]


def _restore_lightning_model(checkpoint: str, model_name: str, config):
    from relog import infer_model_class_path, resolve_model_class

    module_class = resolve_model_class(infer_model_class_path(config, model_name))
    model = module_class.load_from_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    return model


def lightning_predictor(run, config, logger):
    """``(predict_callable, point_ids, target_names)`` for a single Lightning child."""
    from yg_eo_soilnet.serving.sequence_predictor import SoilSequencePredictor

    model_name = run.data.tags.get("model_name") or ""
    checkpoint = _download(run.info.run_id, f"{ArtifactLayout.CHECKPOINTS}/{ArtifactLayout.CHECKPOINT_FILE}")
    if checkpoint is None:
        raise SystemExit(
            f"Run {run.info.run_id} logged no {ArtifactLayout.CHECKPOINT_FILE}, so its weights "
            "cannot be restored."
        )

    predictor = SoilSequencePredictor(_restore_lightning_model(checkpoint, model_name, config))
    bundle = sequence_bundle_for(model_name, config, logger)
    # The MODEL's own target names, not the run's tag: the checkpoint knows exactly how many columns
    # it emits and in what order, and that is what the predictions have to be labelled with.
    target_names = list(predictor.preprocessing_state.get("target_names") or [])
    point_ids = list(bundle.point_ids)
    return (lambda: predictor.predict(bundle)), point_ids, target_names


def lightning_ensemble_predictor(run, config, logger, matched: list[dict]):
    """``(predict_callable, point_ids, target_names)`` averaging every recovered member.

    The mean only. This export carries estimates; the members' spread is the run's uncertainty and
    lives in eval_results.csv and the uncertainty/ artifacts.
    """
    from yg_eo_soilnet.serving.sequence_predictor import SoilSequencePredictor
    from yg_eo_soilnet.uncertainty import aggregate

    model_name = run.data.tags.get("model_name") or ""
    bundle = sequence_bundle_for(model_name, config, logger)
    predictors = [
        SoilSequencePredictor(_restore_lightning_model(record["checkpoint"], model_name, config))
        for record in matched
    ]
    target_names = list(predictors[0].preprocessing_state.get("target_names") or [])

    def predict():
        return aggregate([predictor.predict(bundle) for predictor in predictors]).mean

    return predict, list(bundle.point_ids), target_names


def backfill_child(
    run, *, client, config, export_config, features, point_ids, logger, checkpoint_dir
) -> dict[str, Any]:
    """Write one child's ``predictions/point_predictions.csv``. Returns an outcome record."""
    model_name = run.data.tags.get("model_name")
    framework = run.data.tags.get("framework", "sklearn")
    target = run.data.tags.get("target") or ""
    outcome: dict[str, Any] = {
        "run_id": run.info.run_id,
        "model_name": model_name,
        "framework": framework,
        "target": target,
    }

    n_members = run.data.params.get("uncertainty_n_members")
    if framework == "lightning" and n_members:
        # Every member has to be found. Averaging the subset that happens to be recoverable and
        # labelling it the ensemble is the one outcome worth refusing outright - it would be neither
        # a member's prediction nor the ensemble's, under a column claiming to be the model's.
        if not checkpoint_dir:
            outcome["skipped"] = (
                f"trained as an ensemble of {n_members} members and member recovery is disabled"
            )
            return outcome

        matched = match_member_checkpoints(
            client, run, checkpoint_dir, split_target_names(target) or [target]
        )
        found = [record for record in matched if record.get("checkpoint")]
        outcome["members"] = matched
        if len(found) != int(n_members):
            outcome["skipped"] = (
                f"trained as an ensemble of {n_members} members but only {len(found)} checkpoint(s) "
                f"could be recovered from {checkpoint_dir}; averaging a subset would not be the "
                "ensemble the run reported"
            )
            return outcome

        logger.info(
            f"  recovered {len(found)}/{n_members} members for {target} "
            f"({sum(r.get('source') == 'mlflow' for r in found)} from MLflow, "
            f"{sum(r.get('source') == 'lightning_logs' for r in found)} from {checkpoint_dir})"
        )
        predict, child_point_ids, target_names = lightning_ensemble_predictor(
            run, config, logger, found
        )
        n_expected = len(child_point_ids)
    elif framework == "lightning":
        predict, child_point_ids, target_names = lightning_predictor(run, config, logger)
        n_expected = len(child_point_ids)
    else:
        predict, n_expected = sklearn_predictor(run, features)
        child_point_ids = point_ids.to_numpy()
        target_names = split_target_names(target) or [target]

    with mlflow.start_run(run_id=run.info.run_id):
        written = ChildRunLogger()._log_point_predictions(
            config=export_config,
            model_name=model_name,
            target_names=target_names,
            predict=predict,
            point_ids=child_point_ids,
            n_expected=n_expected,
        )
    if not written:
        outcome["skipped"] = "excluded by --models / --skip-models"
    else:
        outcome.update(written)
    return outcome


# --- orchestration ---------------------------------------------------------


def backfill(args: argparse.Namespace) -> dict[str, Any]:
    config = Config(config_path=args.config_path)
    # The URI only, exactly as relog.py does: switching to the config's experiment would make
    # start_run(run_id=...) fail whenever the run belongs to a different one. A run is reached by
    # id, not by experiment.
    configure_tracking_uri(config)

    logger = TrainingLogger(name="prediction-backfill", enable_file_logging=False).get_logger()

    client = mlflow.MlflowClient()
    parent = client.get_run(args.parent_run_id)
    mlflow.set_experiment(experiment_id=parent.info.experiment_id)

    features, point_ids = rebuild_features(config, logger)
    drift = check_for_drift(
        args.parent_run_id,
        features,
        point_ids,
        allow_population_drift=args.allow_population_drift,
        logger=logger,
    )

    children = _scoring_runs(
        client.search_runs(
            experiment_ids=[parent.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{args.parent_run_id}'",
        )
    )
    children = [run for run in children if run.data.tags.get("model_name")]
    logger.info(f"{len(children)} child model run(s) to back-fill.")

    # Where a Lightning ensemble's member checkpoints are scavenged from, for runs that predate
    # members logging their own. None disables recovery and restores the plain skip.
    checkpoint_dir = None if args.no_member_recovery else args.member_checkpoint_dir

    if args.dry_run:
        for run in children:
            logger.info(
                f"  would export {run.data.tags.get('target')} / {run.data.tags.get('model_name')} "
                f"({run.data.tags.get('framework', 'sklearn')})"
            )
        return {"dry_run": True, "drift": drift, "n_children": len(children)}

    export_config = _export_config(config, args)
    outcomes = []
    for run in children:
        label = f"{run.data.tags.get('target')} / {run.data.tags.get('model_name')}"
        try:
            outcome = backfill_child(
                run,
                client=client,
                config=config,
                export_config=export_config,
                features=features,
                point_ids=point_ids,
                logger=logger,
                checkpoint_dir=checkpoint_dir,
            )
        except SystemExit:
            raise
        except Exception as exc:
            outcome = {
                "run_id": run.info.run_id,
                "model_name": run.data.tags.get("model_name"),
                "error": f"{type(exc).__name__}: {exc}",
            }
        # One child failing must not cost the others their export, so the loop records and
        # continues; the summary is where the failures are answered for.
        if outcome.get("error"):
            logger.warning(f"  {label}: {outcome['error']}")
        elif outcome.get("skipped"):
            logger.info(f"  {label}: skipped - {outcome['skipped']}")
        else:
            logger.info(f"  {label}: exported {outcome.get('n_points')} points")
        outcomes.append(outcome)

    exported = [o for o in outcomes if not o.get("skipped") and not o.get("error")]
    summary = {
        "backfilled": True,
        "parent_run_id": args.parent_run_id,
        "config_path": args.config_path,
        "data_file": getattr(config, "DATA_FILE", None),
        "targets_file": getattr(config, "TARGETS_FILE", None),
        "n_points": int(len(features)),
        "id_column": point_id_column(config),
        "drift": drift,
        "member_checkpoint_dir": checkpoint_dir,
        "children": outcomes,
    }

    with mlflow.start_run(run_id=args.parent_run_id):
        if exported:
            ParentRunLogger()._log_point_prediction_export(args.parent_run_id, export_config)
        # Written whether or not anything was exported: "this run was back-filled and produced
        # nothing" is exactly as worth recording as a success, and without this file a back-filled
        # export is indistinguishable from one the run itself produced.
        log_json(summary, BACKFILL_SUMMARY_FILE, ArtifactLayout.PREDICTIONS)
        mlflow.set_tags({"point_predictions_backfilled": "true"})

    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = backfill(args)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
