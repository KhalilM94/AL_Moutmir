from pathlib import Path

import yaml


def test_lightning_registry_contains_generic_template() -> None:
    registry_path = Path(__file__).resolve().parents[1] / "lightning_registry.yml"
    registry = yaml.safe_load(registry_path.read_text())

    assert isinstance(registry, dict)
    assert "ToyRegression" in registry

    spec = registry["ToyRegression"]
    assert spec["modeltype"] == "dl"
    assert spec["enabled"] is False
    assert {"import_path", "datamodule_import_path", "init_args", "datamodule_init_args", "trainer_args"}.issubset(
        spec.keys()
    )