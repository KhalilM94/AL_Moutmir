"""Two runs must write the same artifact paths, or MLflow cannot compare them.

The defect: MLflow's compare-runs view matches artifacts by RELATIVE PATH, and every filename used
to embed the target and the model - `eval_results/eval_results_organic_matter_pct_soil_cnn.csv` in
one run, `..._organic_matter_g_kg_soil_cnn.csv` in another. Zero paths in common, so the artifact
tab rendered "no common artifact to display". Checkpoints were worse: they kept Lightning's
`epoch=18-step=2014.ckpt`, which differs between any two runs even at identical target and model.

The first test here is the regression test for exactly that, and it is the one that would have
failed before the rename.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import yg_eo_soilnet.logger.mlflow_loggers as mlflow_loggers_module
from yg_eo_soilnet.artifacts import ArtifactLayout, candidate_artifact_paths
from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger


def _record_artifact_paths(monkeypatch) -> list[str]:
    """Collect the run-relative destination of every artifact a logger writes."""
    destinations: list[str] = []

    def capture(path, artifact_path=None):
        import os

        leaf = os.path.basename(path)
        destinations.append(f"{artifact_path}/{leaf}" if artifact_path else leaf)

    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", capture)
    monkeypatch.setattr("yg_eo_soilnet.artifacts.mlflow.log_artifact", capture)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "set_tags", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_metric", MagicMock())
    monkeypatch.setattr("mlflow.pyfunc.log_model", MagicMock())
    return destinations


def _run(logger, monkeypatch, *, target: str, model_name: str, checkpoint) -> set[str]:
    destinations = _record_artifact_paths(monkeypatch)
    logger.log_lightning_child_run(
        config=SimpleNamespace(EXPLAIN_ENABLED=False),
        target=target,
        model_name=model_name,
        evaluation_df=pd.DataFrame(
            {target: [1.0, 2.0, 3.0, 4.0], "prediction": [1.1, 2.2, 2.9, 4.1]}
        ),
        validation_metrics={"val_loss": 0.5},
        test_metrics={"test_loss": 0.4},
        best_model_path=str(checkpoint),
        model=SimpleNamespace(),
    )
    return set(destinations)


def test_two_runs_with_different_targets_and_models_share_every_artifact_path(
    monkeypatch, tmp_path
) -> None:
    """THE regression test for 'no common artifact to display'."""
    logger = ChildRunLogger()

    first_ckpt = tmp_path / "epoch=18-step=2014.ckpt"
    first_ckpt.write_bytes(b"a")
    second_ckpt = tmp_path / "epoch=65-step=3498.ckpt"
    second_ckpt.write_bytes(b"b")

    first = _run(
        logger, monkeypatch, target="organic_matter_pct", model_name="soil_cnn", checkpoint=first_ckpt
    )
    second = _run(
        logger, monkeypatch, target="clay_pct", model_name="soil_sequence", checkpoint=second_ckpt
    )

    assert first, "the first run wrote no artifacts at all"
    assert first == second, (
        "runs wrote different artifact paths, so MLflow's compare view will show nothing in "
        f"common. only in the first: {sorted(first - second)}; only in the second: "
        f"{sorted(second - first)}"
    )


def test_no_artifact_path_contains_a_target_or_model_name(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "epoch=1-step=2.ckpt"
    checkpoint.write_bytes(b"a")

    destinations = _run(
        ChildRunLogger(),
        monkeypatch,
        target="organic_matter_pct",
        model_name="soil_cnn",
        checkpoint=checkpoint,
    )

    for destination in destinations:
        assert "organic_matter_pct" not in destination, destination
        assert "soil_cnn" not in destination, destination


def test_the_checkpoint_lands_on_a_stable_leaf(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "epoch=18-step=2014.ckpt"
    checkpoint.write_bytes(b"a")

    destinations = _run(
        ChildRunLogger(),
        monkeypatch,
        target="om",
        model_name="soil_cnn",
        checkpoint=checkpoint,
    )

    assert "checkpoints/best.ckpt" in destinations
    assert not any("epoch=" in destination for destination in destinations)


def test_a_missing_checkpoint_does_not_lose_the_rest_of_the_run(monkeypatch, tmp_path) -> None:
    """A path Lightning never wrote must not take the eval artifacts and the summary down with it."""
    destinations = _run(
        ChildRunLogger(),
        monkeypatch,
        target="om",
        model_name="soil_cnn",
        checkpoint=tmp_path / "never-written.ckpt",
    )

    assert not any(destination.startswith("checkpoints/") for destination in destinations)
    assert any(destination.startswith("eval_results/") for destination in destinations)
    assert any(destination.startswith("meta/") for destination in destinations)


# --- layout units -----------------------------------------------------------


@pytest.mark.parametrize(
    "leaf",
    [
        ArtifactLayout.EVAL_RESULTS_FILE,
        ArtifactLayout.SPLIT_SUMMARY_FILE,
        ArtifactLayout.RUN_SUMMARY_FILE,
        ArtifactLayout.CV_RESULTS_FILE,
        ArtifactLayout.CHECKPOINT_FILE,
        ArtifactLayout.PRED_OBS_FILE,
        ArtifactLayout.SHAP_BEESWARM_FILE,
        ArtifactLayout.SHAP_VALUES_FILE,
    ],
)
def test_stable_leaves_are_plain_names(leaf: str) -> None:
    assert "/" not in leaf
    assert leaf.islower() or leaf.replace(".", "").replace("_", "").isalnum()


def test_per_target_paths_stay_flat_for_a_single_target() -> None:
    """Flat is what makes single-target runs comparable; nesting is only for multi-output runs."""
    assert ArtifactLayout.explain_path() == "explain"
    assert ArtifactLayout.plots_path() == "plots"
    assert ArtifactLayout.explain_path(None) == "explain"


def test_per_target_paths_nest_when_given_a_target() -> None:
    assert ArtifactLayout.explain_path("clay_pct") == "explain/clay_pct"
    assert ArtifactLayout.plots_path("organic matter/pct") == "plots/organic_matter_pct"


def test_readers_still_resolve_both_generations_of_eval_results() -> None:
    """Hundreds of runs in mlruns/ predate the rename; the leaderboard must still read them."""
    candidates = candidate_artifact_paths(
        ArtifactLayout.EVAL_RESULTS,
        ArtifactLayout.EVAL_RESULTS_FILE,
        ArtifactLayout.eval_results_filename("om", "soil_cnn"),
    )

    assert candidates[0] == "eval_results/eval_results.csv"
    assert "eval_results/eval_results_om_soil_cnn.csv" in candidates
    # Older runs still wrote it at the run root.
    assert "eval_results_om_soil_cnn.csv" in candidates


def test_reader_candidates_are_deduplicated() -> None:
    candidates = candidate_artifact_paths(
        ArtifactLayout.EVAL_RESULTS, ArtifactLayout.EVAL_RESULTS_FILE, ArtifactLayout.EVAL_RESULTS_FILE
    )
    assert len(candidates) == len(set(candidates))
