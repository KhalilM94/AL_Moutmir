from pathlib import Path

import pytest

from config import Config


def test_config_reads_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    registry_path = tmp_path / "registry.yml"
    config_path.write_text(
        "\n".join(
            [
                "DATA_FOLDER: base_folder",
                "DATA_FILE: base.csv",
                "RANDOM_SEED: 1",
                "TEST_SIZE: 0.3",
                "CLUSTERING_STRATEGY:",
                "  enabled: false",
                "  class_path: yg_eo_soilnet.models.KMeansClusterStrategy",
                "  params: {}",
                "TARGET_COLUMNS: [base_target]",
                "CATEGORICAL_FEATURES: [cat]",
                "EXCLUDE_CATEGORICAL: []",
                "ELIMINATED_FEATURES: []",
                "COLUMNS_TO_TRANSFORM: []",
            ]
        )
    )
    registry_path.write_text("enabled: true\n")

    monkeypatch.setenv("DATA_FOLDER", "override_folder")
    monkeypatch.setenv("RANDOM_SEED", "7")
    monkeypatch.setenv("TEST_SIZE", "0.25")
    monkeypatch.setenv("IGNORE_BANDS", "true")
    monkeypatch.setenv("N_BANDS", "3")
    monkeypatch.setenv(
        "CLUSTERING_STRATEGY",
        '{"enabled": true, "class_path": "yg_eo_soilnet.models.KMeansClusterStrategy", "params": {"n_clusters": 2}}',
    )

    config = Config(config_path=str(config_path), registry_path=str(registry_path))

    assert config.DATA_FOLDER == "override_folder"
    assert config.RANDOM_SEED == 7
    assert config.TEST_SIZE == 0.25
    assert config.IGNORE_BANDS == ["Band_1", "Band_2", "Band_3"]
    assert config.ENABLE_CLUSTERING is True
    assert config.CLUSTERING_STRATEGY["params"]["n_clusters"] == 2


def test_config_raises_for_missing_registry(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "DATA_FOLDER: base_folder",
                "DATA_FILE: base.csv",
                "RANDOM_SEED: 1",
                "TEST_SIZE: 0.3",
                "CLUSTERING_STRATEGY:",
                "  enabled: false",
                "  class_path: yg_eo_soilnet.models.KMeansClusterStrategy",
                "  params: {}",
                "TARGET_COLUMNS: [base_target]",
                "CATEGORICAL_FEATURES: [cat]",
                "EXCLUDE_CATEGORICAL: []",
                "ELIMINATED_FEATURES: []",
                "COLUMNS_TO_TRANSFORM: []",
            ]
        )
    )

    with pytest.raises(FileNotFoundError, match="Model registry YAML not found"):
        Config(config_path=str(config_path), registry_path=str(tmp_path / "missing.yml"))