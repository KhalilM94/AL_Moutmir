from types import SimpleNamespace

from yg_eo_soilnet.datamodules.lightning_datamodule import LightningTabularDataModule
from yg_eo_soilnet.models.lightning_config_factory import LightningConfigFactory


class FakeModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_lightning_factory_builds_bundle_with_inferred_input_dim(toy_dataframe) -> None:
    registry = {
        "toy_lightning": {
            "enabled": True,
            "modeltype": "dl",
            "import_path": "fake.module.FakeModel",
            "datamodule_import_path": "fake.module.LightningTabularDataModule",
            "init_args": {"input_dim": None, "output_dim": 1, "hidden_dim": 8},
            "datamodule_init_args": {"batch_size": 2, "val_size": 0.25},
            "trainer_args": {"max_epochs": 3},
            "callbacks": {
                "early_stopping": {"monitor": "val_loss", "mode": "min", "patience": 2},
                "checkpoint": {"monitor": "val_loss", "mode": "min", "save_top_k": 1},
            },
        }
    }
    config = SimpleNamespace(
        LIGHTNING_BATCH_SIZE=2,
        LIGHTNING_VAL_SIZE=0.25,
        LIGHTNING_NUM_WORKERS=0,
        LIGHTNING_PIN_MEMORY=False,
        LIGHTNING_PERSISTENT_WORKERS=False,
        LIGHTNING_SEED=7,
        RANDOM_SEED=42,
        LIGHTNING_MAX_EPOCHS=25,
        LIGHTNING_ACCELERATOR="cpu",
        LIGHTNING_DEVICES=1,
        LIGHTNING_PRECISION="32-true",
        LIGHTNING_ACCUMULATE_GRAD_BATCHES=1,
        LIGHTNING_GRADIENT_CLIP_VAL=0.0,
        LIGHTNING_LOG_EVERY_N_STEPS=1,
        LIGHTNING_EARLY_STOPPING_MONITOR="val_loss",
        LIGHTNING_EARLY_STOPPING_MODE="min",
        LIGHTNING_EARLY_STOPPING_PATIENCE=5,
        LIGHTNING_CHECKPOINT_MONITOR="val_loss",
        LIGHTNING_CHECKPOINT_MODE="min",
        LIGHTNING_SAVE_TOP_K=1,
    )

    factory = LightningConfigFactory(registry=registry, config=config)

    def fake_dynamic_import(path: str):
        if path == "fake.module.FakeModel":
            return FakeModel
        if path == "fake.module.LightningTabularDataModule":
            return LightningTabularDataModule
        raise AssertionError(f"Unexpected import path: {path}")

    factory._dynamic_import = fake_dynamic_import  # type: ignore[method-assign]

    bundles = factory.build_lightning_configs(
        target="target_a",
        data={
            "X_train": toy_dataframe[["cat", "feature"]],
            "X_test": toy_dataframe[["cat", "feature"]].iloc[:2],
            "y_train": toy_dataframe[["target_a", "target_b"]],
            "y_test": toy_dataframe[["target_a", "target_b"]].iloc[:2],
        },
    )

    bundle = bundles["toy_lightning"]
    assert isinstance(bundle.model, FakeModel)
    assert bundle.model.kwargs["input_dim"] == bundle.datamodule.feature_dim
    assert bundle.model.kwargs["output_dim"] == 1
    assert bundle.datamodule.batch_size == 2
    assert bundle.datamodule.target_columns == ["target_a"]
    assert bundle.trainer_kwargs["max_epochs"] == 3
    assert bundle.callback_specs["early_stopping"]["monitor"] == "val_loss"


def test_lightning_factory_rejects_non_dl_entries() -> None:
    factory = LightningConfigFactory(registry={}, config=SimpleNamespace())

    try:
        factory._validate_entry(
            "bad",
            {
                "enabled": True,
                "modeltype": "ml",
                "import_path": "fake.module.FakeModel",
                "datamodule_import_path": "fake.module.LightningTabularDataModule",
            },
        )
    except ValueError as exc:
        assert "must use modeltype 'dl'" in str(exc)
    else:
        raise AssertionError("Expected ValueError")