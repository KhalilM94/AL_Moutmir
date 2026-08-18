"""Where runs are recorded, and under which experiment.

The defect this addresses: the original experiment was created in a different checkout, so its
`artifact_location` - an absolute path fixed at creation - pointed at that checkout forever after.
Metadata landed in this repo's `mlruns/` while artifacts went to the old one, which is why the
recent model directories here contain only `meta.yaml`, `metrics/`, `params/` and `tags/` and no
`artifacts/` at all. A new experiment created under the intended tracking root gets a correct
`artifact_location` from MLflow by itself.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from yg_eo_soilnet import tracking
from yg_eo_soilnet.tracking import (
    DEFAULT_EXPERIMENT_NAME,
    configure_tracking,
    default_tracking_uri,
    resolve_tracking_uri,
    tracking_settings,
)


@pytest.fixture(autouse=True)
def clean_tracking_env(monkeypatch):
    """Start every test from an unset tracking environment.

    `mlflow.set_tracking_uri()` writes MLFLOW_TRACKING_URI into os.environ, so any earlier test in
    the session that configured tracking for real would otherwise leak its value into the env-first
    precedence being tested here.
    """
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)


def test_default_tracking_uri_is_anchored_to_the_repo_not_the_cwd() -> None:
    """A run launched from elsewhere must not quietly start a second, empty mlruns/ beside itself."""
    uri = default_tracking_uri()

    assert uri.startswith("file://")
    assert uri.endswith("/mlruns")
    assert "yg-eo-soilnet" in uri


def test_configured_uri_wins_over_the_default() -> None:
    config = SimpleNamespace(MLFLOW_TRACKING_URI="sqlite:///mlflow.db")
    assert resolve_tracking_uri(config) == "sqlite:///mlflow.db"


def test_blank_or_missing_uri_falls_back_to_the_repo_default() -> None:
    assert resolve_tracking_uri(SimpleNamespace(MLFLOW_TRACKING_URI="")) == default_tracking_uri()
    assert resolve_tracking_uri(SimpleNamespace(MLFLOW_TRACKING_URI="   ")) == default_tracking_uri()
    assert resolve_tracking_uri(SimpleNamespace()) == default_tracking_uri()
    assert resolve_tracking_uri(None) == default_tracking_uri()


# --- settings reader --------------------------------------------------------


def test_settings_read_the_common_block(tmp_path) -> None:
    path = tmp_path / "main_config.yml"
    path.write_text(
        "common:\n  MLFLOW_EXPERIMENT_NAME: 'From_Yaml'\n  MLFLOW_TRACKING_URI: 'file:///tmp/x'\n",
        encoding="utf-8",
    )

    settings = tracking_settings(path)

    assert settings.MLFLOW_EXPERIMENT_NAME == "From_Yaml"
    assert settings.MLFLOW_TRACKING_URI == "file:///tmp/x"


def test_env_overrides_the_yaml(tmp_path, monkeypatch) -> None:
    path = tmp_path / "main_config.yml"
    path.write_text("common:\n  MLFLOW_EXPERIMENT_NAME: 'From_Yaml'\n", encoding="utf-8")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "From_Env")

    assert tracking_settings(path).MLFLOW_EXPERIMENT_NAME == "From_Env"


def test_settings_do_not_need_a_valid_config_tree(tmp_path) -> None:
    """Tracking is configured first, so it must not depend on the data spec or the registries.

    Building a full Config here would mean a typo in an unrelated file decides that runs land
    nowhere - the failure mode that made this a separate reader.
    """
    path = tmp_path / "main_config.yml"
    path.write_text(
        "common:\n  MLFLOW_EXPERIMENT_NAME: 'Fine'\n  DATA_SPEC_PATH: 'does_not_exist.yml'\n",
        encoding="utf-8",
    )

    assert tracking_settings(path).MLFLOW_EXPERIMENT_NAME == "Fine"


def test_settings_survive_an_unreadable_or_broken_config(tmp_path) -> None:
    broken = tmp_path / "broken.yml"
    broken.write_text("common: [this is not a mapping\n", encoding="utf-8")

    assert tracking_settings(broken).MLFLOW_EXPERIMENT_NAME == DEFAULT_EXPERIMENT_NAME
    assert tracking_settings(tmp_path / "missing.yml").MLFLOW_EXPERIMENT_NAME == DEFAULT_EXPERIMENT_NAME
    assert tracking_settings(None).MLFLOW_EXPERIMENT_NAME == DEFAULT_EXPERIMENT_NAME


# --- configure_tracking -----------------------------------------------------


def test_configure_sets_the_uri_before_the_experiment(monkeypatch) -> None:
    """Order matters: an experiment created against the wrong root bakes in the wrong
    artifact_location, which is the whole defect."""
    calls: list[str] = []

    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", lambda uri: calls.append("uri"))
    monkeypatch.setattr(tracking.mlflow, "set_experiment", lambda name: calls.append("experiment"))

    configure_tracking(SimpleNamespace(MLFLOW_TRACKING_URI="", MLFLOW_EXPERIMENT_NAME="X"))

    assert calls == ["uri", "experiment"]


def test_configure_opts_into_the_file_store(monkeypatch) -> None:
    """MLflow 3.14 put the filesystem backend in maintenance mode and raises without this.

    Only the `pixi run mlflow` task used to set it, so whether `python main.py` could write to
    mlruns/ at all depended on how the process happened to be launched.
    """
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(tracking.mlflow, "set_experiment", MagicMock())

    configure_tracking(SimpleNamespace(MLFLOW_TRACKING_URI=""))

    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"


def test_configure_leaves_a_database_backend_alone(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(tracking.mlflow, "set_experiment", MagicMock())

    configure_tracking(SimpleNamespace(MLFLOW_TRACKING_URI="sqlite:///mlflow.db"))

    assert "MLFLOW_ALLOW_FILE_STORE" not in os.environ


def test_configure_recreates_a_deleted_experiment(monkeypatch) -> None:
    import mlflow

    attempts = {"set": 0}

    def flaky_set_experiment(name):
        attempts["set"] += 1
        if attempts["set"] == 1:
            raise mlflow.exceptions.MlflowException("deleted")

    created = MagicMock()
    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(tracking.mlflow, "set_experiment", flaky_set_experiment)
    monkeypatch.setattr(tracking.mlflow, "create_experiment", created)

    name = configure_tracking(SimpleNamespace(MLFLOW_EXPERIMENT_NAME="Revived"))

    created.assert_called_once_with("Revived")
    assert name == "Revived"


def test_explicit_experiment_name_wins(monkeypatch) -> None:
    """The HPO path passes its own name while reusing the same tracking root."""
    seen: list[str] = []
    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(tracking.mlflow, "set_experiment", lambda name: seen.append(name))

    configure_tracking(SimpleNamespace(MLFLOW_EXPERIMENT_NAME="Training"), experiment_name="Soil_HPO")

    assert seen == ["Soil_HPO"]


def test_the_default_experiment_is_not_the_one_with_the_stale_artifact_location() -> None:
    """A clean experiment is the fix; reusing the old name would inherit its broken path."""
    assert DEFAULT_EXPERIMENT_NAME != "Soil_Model_Training_Experiment"
