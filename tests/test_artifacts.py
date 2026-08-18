"""The canonical artifact tree, and the three writers that put things in it."""

from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import pandas as pd

from yg_eo_soilnet import artifacts as artifacts_module
from yg_eo_soilnet.artifacts import (
    ArtifactLayout,
    candidate_artifact_paths,
    log_figure,
    log_json,
    log_table,
)


def test_safe_collapses_characters_that_would_break_an_artifact_path() -> None:
    assert ArtifactLayout.safe("organic matter/pct") == "organic_matter_pct"
    assert ArtifactLayout.safe("clay_pct") == "clay_pct"
    # A name made entirely of separators must still yield a usable component.
    assert ArtifactLayout.safe("///") == "unnamed"


def test_filenames_sanitise_both_identifiers() -> None:
    assert (
        ArtifactLayout.eval_results_filename("organic matter", "soil cnn")
        == "eval_results_organic_matter_soil_cnn.csv"
    )
    assert ArtifactLayout.run_summary_filename("om", "xgb") == "run_summary_om_xgb.json"
    assert ArtifactLayout.cv_results_filename("om", "xgb") == "cv_results_om_xgb.csv"


def test_logged_model_name_carries_the_target() -> None:
    """The Lightning side used to log to models/{model_name} with no target in it, so on a
    multi-target run every target overwrote the same slot."""
    assert ArtifactLayout.logged_model_name("clay_pct", "soil_cnn") == "clay_pct_soil_cnn"
    assert ArtifactLayout.logged_model_name("om_pct", "soil_cnn") != ArtifactLayout.logged_model_name(
        "clay_pct", "soil_cnn"
    )


def test_candidate_paths_try_the_current_layout_first_then_the_ones_it_replaced() -> None:
    candidates = candidate_artifact_paths(ArtifactLayout.PLOTS, "pred_obs_om_cnn.png")

    assert candidates[0] == "plots/pred_obs_om_cnn.png"
    # Lightning wrote eval_plots/, and _log_plots wrote to the run root with no artifact_path.
    assert "eval_plots/pred_obs_om_cnn.png" in candidates
    assert "pred_obs_om_cnn.png" in candidates


def test_eval_results_still_resolves_at_the_run_root_for_older_runs() -> None:
    candidates = candidate_artifact_paths(ArtifactLayout.EVAL_RESULTS, "eval_results_om_cnn.csv")
    assert candidates == ["eval_results/eval_results_om_cnn.csv", "eval_results_om_cnn.csv"]


def test_log_table_uploads_under_the_requested_path(monkeypatch) -> None:
    log_artifact = MagicMock()
    monkeypatch.setattr(artifacts_module.mlflow, "log_artifact", log_artifact)

    log_table(pd.DataFrame({"a": [1, 2]}), "table.csv", ArtifactLayout.EVAL_RESULTS)

    assert log_artifact.call_args.kwargs["artifact_path"] == "eval_results"
    assert log_artifact.call_args.args[0].endswith("table.csv")


def test_log_json_writes_json(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_log_artifact(path, artifact_path=None):
        with open(path, encoding="utf-8") as handle:
            captured["text"] = handle.read()
        captured["artifact_path"] = artifact_path

    monkeypatch.setattr(artifacts_module.mlflow, "log_artifact", fake_log_artifact)

    log_json({"enabled": False}, "summary.json", ArtifactLayout.META)

    assert captured["artifact_path"] == "meta"
    assert '"enabled": false' in captured["text"]


def test_log_figure_closes_the_figure_it_was_given(monkeypatch) -> None:
    """Every previous copy of this logic repeated savefig/log/close, and one forgot the close."""
    monkeypatch.setattr(artifacts_module.mlflow, "log_artifact", MagicMock())

    figure = plt.figure()
    log_figure(figure, "plot.png", ArtifactLayout.PLOTS)

    assert not plt.fignum_exists(figure.number)


def test_log_figure_closes_the_figure_even_when_the_upload_fails(monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise RuntimeError("mlflow is down")

    monkeypatch.setattr(artifacts_module.mlflow, "log_artifact", explode)

    figure = plt.figure()
    try:
        log_figure(figure, "plot.png", ArtifactLayout.PLOTS)
    except RuntimeError:
        pass

    assert not plt.fignum_exists(figure.number)


def test_log_figure_tolerates_a_none_figure(monkeypatch) -> None:
    log_artifact = MagicMock()
    monkeypatch.setattr(artifacts_module.mlflow, "log_artifact", log_artifact)

    log_figure(None, "plot.png", ArtifactLayout.PLOTS)

    log_artifact.assert_not_called()
