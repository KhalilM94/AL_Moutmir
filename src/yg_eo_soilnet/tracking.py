"""Where MLflow records runs, under which experiment, and which registered version ships.

Both entry points - ``main.py`` for training and ``tune.py`` for HPO - configure tracking through
this module, because two properties have to hold and neither is MLflow's default.

**An experiment's ``artifact_location`` is an absolute path fixed at creation time.** The original
``Soil_Model_Training_Experiment`` was created in a different checkout, so its metadata has been
written under this repo's ``mlruns/`` while its artifacts went to the old checkout's - a split that
survives any amount of copying, because it lives in the experiment's ``meta.yaml``. Creating the
experiment under the intended tracking root is the only thing that fixes it, and it is why the
experiment name is configurable rather than hardcoded.

**MLflow 3.14 put the filesystem backend in maintenance mode** and raises unless
``MLFLOW_ALLOW_FILE_STORE`` is set. Until now only the ``pixi run mlflow`` task set it, so whether a
process could write to ``mlruns/`` at all depended on how it happened to be launched.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import mlflow
import numpy as np
import yaml

DEFAULT_EXPERIMENT_NAME = "Soil_Model_Training_v2"

_TRACKING_KEYS = ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME")


def tracking_settings(config_path: str | os.PathLike | None) -> SimpleNamespace:
    """Read just the tracking keys from a main config, env first.

    Deliberately NOT a full :class:`Config`. Tracking has to be configured before anything else
    happens, and building the whole config tree first would make "where do runs go" depend on the
    data spec, the registries and every path they reference being valid - so a typo in an unrelated
    file would decide that runs land nowhere.
    """
    values: dict[str, str] = {}

    if config_path:
        try:
            with open(config_path, encoding="utf-8") as handle:
                document = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError):
            document = {}
        common = document.get("common") if isinstance(document, dict) else {}
        for key in _TRACKING_KEYS:
            for source in (common if isinstance(common, dict) else {}, document if isinstance(document, dict) else {}):
                if key in source:
                    values[key] = str(source[key] or "")
                    break

    # Env wins, matching Config._get_config's precedence.
    for key in _TRACKING_KEYS:
        override = os.environ.get(key)
        if override is not None:
            values[key] = override

    return SimpleNamespace(
        MLFLOW_TRACKING_URI=values.get("MLFLOW_TRACKING_URI", ""),
        MLFLOW_EXPERIMENT_NAME=values.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME),
    )


def default_tracking_uri() -> str:
    """``<repo>/mlruns`` as a file URI.

    Anchored to this package's location rather than to the working directory: a run launched from
    elsewhere would otherwise silently start a second, empty ``mlruns/`` beside itself.
    """
    return (Path(__file__).resolve().parents[2] / "mlruns").as_uri()


def resolve_tracking_uri(config=None) -> str:
    configured = str(getattr(config, "MLFLOW_TRACKING_URI", "") or "").strip()
    return configured or default_tracking_uri()


def configure_tracking_uri(config=None) -> str:
    """Point MLflow at the tracking root, without touching the current experiment.

    Split out because resuming an existing run by id must NOT switch experiments: MLflow refuses
    ``start_run(run_id=...)`` when the active experiment is not the one that run belongs to, so a
    caller that only wants to reach an existing run needs the URI without the rest.
    """
    tracking_uri = resolve_tracking_uri(config)

    if urlparse(tracking_uri).scheme in ("", "file"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def configure_tracking(config=None, experiment_name: str | None = None) -> str:
    """Point MLflow at the configured tracking root and experiment; return the experiment name.

    Must run before any run starts - including the implicit one that
    ``SklearnDataSplitter.split_data`` triggers by calling ``mlflow.log_artifacts`` - or the run
    lands in whatever experiment happened to be current.
    """
    configure_tracking_uri(config)

    name = experiment_name or str(
        getattr(config, "MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME) or DEFAULT_EXPERIMENT_NAME
    )
    try:
        mlflow.set_experiment(name)
    except mlflow.exceptions.MlflowException:  # type: ignore[attr-defined]
        # Restore or create a new experiment if previously deleted.
        mlflow.create_experiment(name)
        mlflow.set_experiment(name)
    return name


# --- run ownership and stale-run cleanup ----------------------------------------------------
# An MLflow run that dies without unwinding stays RUNNING forever. `ActiveRun.__exit__` marks a run
# FAILED on a normal exception, but it never runs when the process is SIGKILLed - which is what the
# kernel's OOM killer sends. A run left RUNNING reads as "still working", or worse as a model that
# was successfully made, and it is what made an OOM-killed multi-target run look like a bug in the
# target grouping.
#
# The fix has to be a SWEEP rather than a signal handler, because SIGKILL cannot be caught. The
# handlers below only cover the signals that can be.

HOST_NAME_TAG = "host_name"
HOST_PID_TAG = "host_pid"


def run_owner_tags() -> dict[str, str]:
    """Who is writing this run. Tagged so a later process can tell finished from abandoned."""
    import socket

    return {HOST_NAME_TAG: socket.gethostname(), HOST_PID_TAG: str(os.getpid())}


def _process_is_alive(pid: int) -> bool:
    """Whether a pid on THIS host still exists. Signal 0 checks without delivering anything."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive, owned by somebody else. Not ours to clean up either way.
        return True
    except (OverflowError, ValueError):
        return False
    return True


def close_stale_runs(experiment_name: str | None = None, logger: Any = None) -> list[str]:
    """Mark runs abandoned by a dead process as KILLED. Returns the run ids closed.

    Deliberately narrow. Only a run that is RUNNING, tagged with THIS hostname, and whose recorded
    pid no longer exists is touched. Sweeping on "status is RUNNING" alone would let one training
    process terminate a second one running concurrently, which is a far worse failure than the
    phantom runs this cleans up.
    """
    import socket

    name = experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT_NAME
    try:
        client = mlflow.tracking.MlflowClient()  # type: ignore[attr-defined]
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            return []
        running = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'RUNNING'",
            max_results=1000,
        )
    except Exception as exc:  # pragma: no cover - tracking store unreachable
        # Never fatal: a failed sweep is cosmetic, and refusing to train because old runs could not
        # be tidied would be the worse trade.
        if logger is not None:
            logger.warning(f"Could not sweep stale runs: {type(exc).__name__}: {exc}")
        return []

    hostname = socket.gethostname()
    this_pid = os.getpid()
    closed: list[str] = []
    for run in running:
        tags = run.data.tags
        if tags.get(HOST_NAME_TAG) != hostname:
            continue
        raw_pid = tags.get(HOST_PID_TAG)
        if raw_pid is None:
            # Written before runs carried ownership tags. Left alone rather than guessed at.
            continue
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError):
            continue
        if pid == this_pid or _process_is_alive(pid):
            continue
        try:
            client.set_terminated(run.info.run_id, "KILLED")
            closed.append(run.info.run_id)
        except Exception as exc:  # pragma: no cover
            if logger is not None:
                logger.warning(f"Could not terminate stale run {run.info.run_id}: {exc}")

    if closed and logger is not None:
        logger.info(
            f"Marked {len(closed)} abandoned run(s) as KILLED; their process is gone. "
            "A run left RUNNING is usually one the OOM killer took."
        )
    return closed


def start_child_run(run_name: str, tags: dict | None = None):
    """Open a run for one model, nested under the current one when there is one.

    `nested=True` is an ERROR when nothing is active, so hardcoding it ties the trainers to being
    called from inside main.py's parent run. They are also used directly - by tests, and by anyone
    driving a single model - and there a top-level run is the right thing. Deciding from
    `active_run()` keeps both working.

    Owner tags go on here rather than at each call site, so every run this project opens can be
    told apart from an abandoned one by :func:`close_stale_runs`.
    """
    run = mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None)
    mlflow.set_tags({**run_owner_tags(), **(tags or {})})
    return run


def log_params_once(params: Any, logger: Any = None) -> None:
    """Log params, skipping any key already recorded on this run with a DIFFERENT value.

    MLflow params are immutable: re-logging the same value is fine, changing one raises. That
    exception is worth avoiding rather than propagating, because of WHERE it lands. The parent run's
    params are written partly at the start of training and partly in the summary at the end, so a
    duplicated key does not fail fast - it fails after every model has been fitted, logged and
    registered, and takes the summary, the leaderboard and the run's FINISHED status with it. One
    such clash discarded the tail of an hour-long run.

    Filtering BEFORE the call rather than catching after it: the file store's ``log_batch`` applies
    params one at a time and raises partway through, so a rejected batch can leave some keys written
    and others not.
    """
    params = dict(params)
    if not params:
        return

    run = mlflow.active_run()
    existing: dict = {}
    if run is not None:
        try:
            existing = mlflow.tracking.MlflowClient().get_run(run.info.run_id).data.params  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - unreadable store; let log_params speak for itself
            existing = {}

    writable = {}
    for key, value in params.items():
        # str() is the form MLflow stores, so it is the form to compare against.
        current = existing.get(str(key))
        if current is not None and current != str(value):
            if logger is not None:
                logger.warning(
                    f"Param {key!r} is already logged as {current!r} on this run; keeping that and "
                    f"not overwriting it with {str(value)!r}. Two places are writing the same key."
                )
            continue
        writable[key] = value

    if writable:
        mlflow.log_params(writable)


def install_run_signal_handlers(logger: Any = None) -> None:
    """End the active run stack as KILLED on SIGINT/SIGTERM, then re-raise.

    Covers Ctrl-C and `kill`. It cannot cover SIGKILL - nothing can - which is why
    :func:`close_stale_runs` exists as well.
    """
    import signal

    def handler(signum, frame):
        try:
            while mlflow.active_run() is not None:
                mlflow.end_run("KILLED")
        except Exception:  # pragma: no cover - best effort during teardown
            pass
        if logger is not None:
            logger.warning(f"Received signal {signum}; marked the active run(s) KILLED.")
        # Restore the default and re-raise, so the exit status still says what happened.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, handler)
        except (ValueError, OSError):  # pragma: no cover - not on the main thread
            pass


CHAMPION_ALIAS = "champion"
# The metric the promotion decision reads. It is the one that means the same thing for both
# training families and is in the target's original units; yg_eo_soilnet.metrics is the authority
# on which direction is better, so this module does not restate it.
CHAMPION_METRIC = "rmse_test"


def _version_metric(client, version, metric_name: str) -> float | None:
    """The metric of a registered version, read from the run that produced it."""
    run_id = getattr(version, "run_id", None)
    if not run_id:
        return None
    try:
        value = client.get_run(run_id).data.metrics.get(metric_name)
    except Exception:
        # The run behind an aliased version can be deleted while the version survives.
        return None
    return None if value is None else float(value)


def promote_if_better(
    name: str,
    version: Any,
    metric_value: float | None,
    *,
    client=None,
    alias: str = CHAMPION_ALIAS,
    metric_name: str = CHAMPION_METRIC,
) -> dict:
    """Move ``alias`` onto ``version`` only when it genuinely beats the incumbent.

    Returns the decision - both scores and a reason - so the caller can record it and a promotion is
    auditable rather than a surprise.

    Scope worth being precise about: an MLflow alias belongs to one REGISTERED MODEL NAME. Models
    are registered per target and architecture, so ``champion`` means "the best version of this
    model on this target", not "the best model for this target". Choosing between soil_cnn and
    XGBoost is the leaderboard's job, not this function's.

    The rules, and why each one is not the obvious alternative:

    * **no incumbent** -> promote. The first measurable version should be reachable by alias.
    * **incumbent unmeasurable** (its run or metric is gone) -> promote, and say so. A candidate we
      can score beats one we cannot.
    * **no metric on the new version** -> do NOT promote. A degenerate fit produces no metrics, and
      silently shipping it because it "has no worse score" is the failure this guards against.
    * **equal scores** -> keep the incumbent, so re-running the same config does not churn the alias.
    """
    from yg_eo_soilnet.metrics import METRIC_DIRECTION

    if version is None:
        return {"promoted": False, "reason": "the model was not registered"}

    if metric_value is None or not np.isfinite(metric_value):
        return {
            "promoted": False,
            "reason": f"the new version has no usable {metric_name}",
            "candidate": None,
        }

    if client is None:
        client = mlflow.MlflowClient()

    try:
        incumbent = client.get_model_version_by_alias(name, alias)
    except Exception:
        incumbent = None

    decision: dict[str, Any] = {
        "alias": alias,
        "metric": metric_name,
        "candidate": float(metric_value),
        "candidate_version": str(version),
    }

    if incumbent is None:
        client.set_registered_model_alias(name, alias, version)
        return {**decision, "promoted": True, "reason": f"no version was aliased {alias} yet"}

    decision["incumbent_version"] = str(incumbent.version)
    incumbent_value = _version_metric(client, incumbent, metric_name)
    decision["incumbent"] = incumbent_value

    if incumbent_value is None:
        client.set_registered_model_alias(name, alias, version)
        return {
            **decision,
            "promoted": True,
            "reason": f"the {alias} version has no readable {metric_name}",
        }

    stem = metric_name.split("_")[0]
    higher_is_better = METRIC_DIRECTION.get(stem) == "higher"
    better = metric_value > incumbent_value if higher_is_better else metric_value < incumbent_value

    if not better:
        return {
            **decision,
            "promoted": False,
            "reason": f"{metric_name} {metric_value:.6g} does not beat {incumbent_value:.6g}",
        }

    client.set_registered_model_alias(name, alias, version)
    return {
        **decision,
        "promoted": True,
        "reason": f"{metric_name} {metric_value:.6g} beats {incumbent_value:.6g}",
    }
