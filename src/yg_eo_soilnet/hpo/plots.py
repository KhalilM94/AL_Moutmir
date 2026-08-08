"""Study figures and the artifacts a finished study leaves behind.

These follow plot_utils.py's conventions - build a Figure, tight_layout, return it, and hand back a
figure carrying a message rather than None when there is nothing to draw - but live in the hpo
package rather than in plot_utils.py itself. plot_utils is imported on the sklearn path by
mlflow_loggers and sklearn_trainer, neither of which has any reason to import optuna.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import optuna

# Explicit: `import optuna` alone does not bind the visualization submodule, so
# `optuna.visualization.matplotlib.plot_*` raises AttributeError without this. The matplotlib
# backend is the only usable one here - the default one needs plotly, which is not installed.
from matplotlib.figure import Figure
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

from yg_eo_soilnet.hpo.tracker import COMPLETE, ObjectiveTracker

ARTIFACT_PATH = "optuna"


def _message_figure(message: str) -> Figure:
    """The repo's empty-data convention: a real Figure carrying text, not None."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _completed(study: optuna.Study) -> int:
    return sum(1 for trial in study.trials if trial.state.name == COMPLETE)


def optimization_history(study: optuna.Study) -> Figure:
    """Objective per trial with the running best."""
    if _completed(study) < 1:
        return _message_figure("No completed trials yet")
    # These return an Axes, not a Figure - every saver in this repo calls fig.savefig, so the
    # .figure hop is what makes them usable at all.
    axes = plot_optimization_history(study)
    axes.grid(True)
    axes.set_axisbelow(True)
    figure = axes.figure
    figure.tight_layout()
    return figure


def param_importances(study: optuna.Study) -> Figure:
    """Which hyperparameters actually moved the objective."""
    # fANOVA needs at least two completed trials and something that varies between them; below that
    # get_param_importances raises rather than returning an empty result.
    if _completed(study) < 2:
        return _message_figure("Need at least two completed trials for importances")
    try:
        axes = plot_param_importances(study)
    except (ValueError, RuntimeError, ZeroDivisionError) as exc:
        return _message_figure(f"Importances unavailable: {exc}")
    axes.grid(True)
    axes.set_axisbelow(True)
    figure = axes.figure
    figure.tight_layout()
    return figure


def write_study_artifacts(
    study: optuna.Study,
    tracker: ObjectiveTracker,
    output_dir: str | Path,
    *,
    log_to_mlflow: bool = True,
    logger: Any = None,
) -> list[Path]:
    """Write trials.csv and the two figures, and attach them to the active MLflow run if any.

    Written to a durable directory first and logged from there, so a --no-mlflow run still leaves
    the same artifacts on disk. That is also why the tempdir dance used in mlflow_loggers is not
    needed here.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    frame = tracker.trials_frame(study)
    csv_path = output_dir / "trials.csv"
    frame.to_csv(csv_path, index=False)
    written.append(csv_path)

    for name, builder in (("optimization_history", optimization_history), ("param_importances", param_importances)):
        try:
            figure = builder(study)
        except Exception as exc:  # pragma: no cover - a plot must never sink a finished study
            if logger is not None:
                logger.warning(f"Could not build the {name} figure: {exc!r}")
            continue
        path = output_dir / f"{name}.png"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        written.append(path)

    if log_to_mlflow and mlflow.active_run() is not None:
        for path in written:
            mlflow.log_artifact(str(path), artifact_path=ARTIFACT_PATH)

    return written
