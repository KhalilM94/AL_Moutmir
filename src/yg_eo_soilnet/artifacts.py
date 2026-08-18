"""The canonical artifact tree, and the three writers that put things in it.

Every artifact in this project is staged in a temporary directory and uploaded to MLflow - there is
no on-disk results root, and this module does not introduce one. What it does introduce is a single
place that decides WHERE inside a run an artifact lands, because before it every writer hardcoded
its own ``artifact_path`` string and the two training families drifted apart:

* sklearn CV plots were logged with no ``artifact_path`` at all, so they sat at the run root while
  the matching CSVs went to ``cv_results/``;
* Lightning wrote ``eval_plots/`` and ``lightning_metadata/``, neither of which had a sklearn
  counterpart;
* fitted models went to ``models/{model_name}`` on one side and a bare ``{target}_{model_name}`` on
  the other.

The tree below is what both families write now::

    <run>/
      meta/          run_summary_<target>_<model>.json
      eval_results/  eval_results_<target>_<model>.csv, split_summary_<target>.json
      plots/         pred_obs_<target>_<model>.png, cv_val_curve_<target>_<model>.png
      explain/       shap_beeswarm_*.png, shap_bar_*.png, shap_values_*.parquet, shap_summary_*.json
      cv/            cv_results_<target>_<model>.csv        (sklearn only)
      checkpoints/   <best>.ckpt                            (Lightning only)

Fitted models are the one thing that does NOT live in this tree: MLflow 3 stores them as LoggedModels
under ``mlruns/<experiment>/models/``, addressed by name rather than by artifact path. Their shared
naming convention is :meth:`ArtifactLayout.logged_model_name`.

:data:`LEGACY_ARTIFACT_PATHS` maps each new location to the ones it replaced. Readers consult it so
that runs recorded before this change - of which there are many in ``mlruns/`` - still resolve.
Writers never use it.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Any, Iterable

import matplotlib.pyplot as plt
import mlflow
import pandas as pd


class ArtifactLayout:
    """Where each kind of artifact lives inside a run, and how its file is named."""

    META = "meta"
    EVAL_RESULTS = "eval_results"
    PLOTS = "plots"
    EXPLAIN = "explain"
    CV = "cv"
    CHECKPOINTS = "checkpoints"

    # Parent-run only.
    LEADERBOARD_PLOTS = "leaderboard_plots"
    DATA_SPLITS = "data_splits"

    # Stable leaf names. MLflow's compare-runs view matches artifacts by RELATIVE PATH, so a
    # filename carrying the target and model - eval_results_organic_matter_pct_soil_cnn.csv - gives
    # two runs zero paths in common and the compare tab renders "no common artifact to display".
    # The run already identifies its target and model through its name and its target/model_name
    # tags, so repeating them in every leaf bought nothing and cost comparability.
    EVAL_RESULTS_FILE = "eval_results.csv"
    SPLIT_SUMMARY_FILE = "split_summary.json"
    RUN_SUMMARY_FILE = "run_summary.json"
    CV_RESULTS_FILE = "cv_results.csv"
    PRED_OBS_FILE = "pred_obs.png"
    # Lightning names its checkpoints epoch=NN-step=MMM.ckpt, which differs between any two runs
    # even at identical target and model. The original name is preserved in the run summary and as
    # a run tag rather than in the path.
    CHECKPOINT_FILE = "best.ckpt"

    SHAP_BEESWARM_FILE = "shap_beeswarm.png"
    SHAP_BAR_FILE = "shap_bar.png"
    SHAP_BLOCK_BAR_FILE = "shap_block_bar.png"
    SHAP_VALUES_FILE = "shap_values.parquet"
    SHAP_SUMMARY_FILE = "shap_summary.json"

    @classmethod
    def explain_path(cls, target: Any = None) -> str:
        """``explain``, or ``explain/<target>`` when one run emits several targets."""
        return cls._per_target(cls.EXPLAIN, target)

    @classmethod
    def plots_path(cls, target: Any = None) -> str:
        """``plots``, or ``plots/<target>`` when one run emits several targets."""
        return cls._per_target(cls.PLOTS, target)

    @classmethod
    def _per_target(cls, root: str, target: Any = None) -> str:
        """Flat for a single target, nested for several.

        A single-target run keeps the flat path so it compares directly against every other
        single-target run - that comparability is the whole point of the stable names. A run that
        emits SEVERAL targets has to nest, because otherwise its targets write the same leaf and
        silently overwrite one another.
        """
        return f"{root}/{cls.safe(target)}" if target else root

    @staticmethod
    def safe(component: Any) -> str:
        """A path component with everything but ``[A-Za-z0-9_.-]`` collapsed to underscores.

        Target names reach here from user config and can carry spaces, slashes or unicode; an
        unsanitised one would either break the upload or silently create a nested artifact folder.
        """
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(component)).strip("_")
        return cleaned or "unnamed"

    # --- legacy names -------------------------------------------------------
    # Writers no longer call these. They exist so candidate_artifact_paths can still resolve the
    # hundreds of runs already recorded under the old per-run naming; deleting them would make the
    # parent-run leaderboard silently skip every historical run.

    @classmethod
    def filename(cls, kind: str, target: Any, model_name: Any, extension: str) -> str:
        """LEGACY ``<kind>_<target>_<model>.<ext>``. Read-only; use the stable *_FILE names."""
        suffix = extension.lstrip(".")
        return f"{kind}_{cls.safe(target)}_{cls.safe(model_name)}.{suffix}"

    @classmethod
    def eval_results_filename(cls, target: Any, model_name: Any) -> str:
        """LEGACY. Use :attr:`EVAL_RESULTS_FILE` when writing."""
        return cls.filename("eval_results", target, model_name, "csv")

    @classmethod
    def split_summary_filename(cls, target: Any) -> str:
        """LEGACY. Use :attr:`SPLIT_SUMMARY_FILE` when writing."""
        return f"split_summary_{cls.safe(target)}.json"

    @classmethod
    def run_summary_filename(cls, target: Any, model_name: Any) -> str:
        """LEGACY. Use :attr:`RUN_SUMMARY_FILE` when writing."""
        return cls.filename("run_summary", target, model_name, "json")

    @classmethod
    def cv_results_filename(cls, target: Any, model_name: Any) -> str:
        """LEGACY. Use :attr:`CV_RESULTS_FILE` when writing."""
        return cls.filename("cv_results", target, model_name, "csv")

    @classmethod
    def logged_model_name(cls, target: Any, model_name: Any) -> str:
        """The name a fitted model is registered under, identical for both families.

        This is a LoggedModel name, not a path. MLflow 3 deprecated ``artifact_path=`` on
        ``log_model`` in favour of ``name=``, and a named model lands in
        ``mlruns/<experiment>/models/m-<hash>/`` rather than inside the run's artifact tree - so
        unifying the two families here means giving them one naming convention, which the sklearn
        side already had (``{target}_{model}``) and the Lightning side did not (``models/{model}``,
        with no target in it, so two targets overwrote each other's slot).
        """
        return f"{cls.safe(target)}_{cls.safe(model_name)}"


# new artifact_path -> the paths it replaced, oldest last. Readers try the new one first.
LEGACY_ARTIFACT_PATHS: dict[str, tuple[str, ...]] = {
    ArtifactLayout.META: ("lightning_metadata",),
    ArtifactLayout.PLOTS: ("eval_plots", ""),
    ArtifactLayout.CV: ("cv_results",),
    # The eval CSV has always lived under eval_results/, but older runs also wrote it at the run
    # root and the parent-run leaderboard still has to find those.
    ArtifactLayout.EVAL_RESULTS: ("",),
}


def candidate_artifact_paths(artifact_path: str, *filenames: str) -> list[str]:
    """Every location a reader should try for one artifact, current layout first.

    Takes several filenames because the rename to stable leaves left two generations in ``mlruns/``:
    a current run holds ``eval_results/eval_results.csv`` while an older one holds
    ``eval_results/eval_results_<target>_<model>.csv``. Pass the stable name first and the legacy
    one after it, and every directory in :data:`LEGACY_ARTIFACT_PATHS` is tried for each.

    The empty-string legacy entry means "the run root", which is where ``_log_plots`` used to put
    its figures, so it yields the bare filename.
    """
    directories = [artifact_path, *LEGACY_ARTIFACT_PATHS.get(artifact_path, ())]

    candidates: list[str] = []
    for filename in filenames:
        if not filename:
            continue
        for directory in directories:
            candidate = f"{directory}/{filename}" if directory else filename
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def log_table(frame: pd.DataFrame, filename: str, artifact_path: str) -> None:
    """Write a DataFrame to CSV in a temp dir and upload it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, filename)
        frame.to_csv(path, index=False)
        mlflow.log_artifact(path, artifact_path=artifact_path)


def log_parquet(frame: pd.DataFrame, filename: str, artifact_path: str) -> None:
    """Same as :func:`log_table` but columnar, for tables wide enough that CSV is wasteful.

    Used for the full SHAP value matrix, which is (n_samples x n_features) floats and is written
    uncapped even when the plots show only the top N.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, filename)
        frame.to_parquet(path, index=False)
        mlflow.log_artifact(path, artifact_path=artifact_path)


def log_json(payload: dict, filename: str, artifact_path: str) -> None:
    """Write a dict as indented JSON and upload it. ``default=str`` so numpy scalars survive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        mlflow.log_artifact(path, artifact_path=artifact_path)


def log_figure(figure, filename: str, artifact_path: str) -> None:
    """Save a matplotlib Figure, upload it, and close it.

    Closing here rather than at the call site is the point: every previous copy of this logic
    repeated ``savefig`` / ``log_artifact`` / ``plt.close`` and at least one forgot the close, which
    leaks figures across a multi-model run until matplotlib starts warning about open figures.
    """
    if figure is None:
        return
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, filename)
            figure.savefig(path, bbox_inches="tight")
            mlflow.log_artifact(path, artifact_path=artifact_path)
    finally:
        plt.close(figure)


def log_figures(figures: Iterable[tuple], artifact_path: str) -> None:
    """``(figure, filename)`` pairs through :func:`log_figure`, skipping ``None`` figures."""
    for figure, filename in figures:
        log_figure(figure, filename, artifact_path)
