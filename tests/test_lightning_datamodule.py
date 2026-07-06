from pathlib import Path

import pandas as pd

from yg_eo_soilnet.datamodules.lightning_datamodule import LightningTabularDataModule


def test_lightning_datamodule_builds_train_val_test_loaders(toy_dataframe) -> None:
    X_train = toy_dataframe[["cat", "feature"]]
    y_train = toy_dataframe[["target_a"]]
    X_test = toy_dataframe[["cat", "feature"]].iloc[:2].reset_index(drop=True)
    y_test = toy_dataframe[["target_a"]].iloc[:2].reset_index(drop=True)

    module = LightningTabularDataModule(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        batch_size=2,
        val_size=0.25,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        seed=7,
        target_columns=["target_a"],
    )

    module.setup("fit")

    train_batch = next(iter(module.train_dataloader()))
    val_batch = next(iter(module.val_dataloader()))
    test_batch = next(iter(module.test_dataloader()))

    assert module.feature_dim == 3
    assert module.target_dim == 1
    assert train_batch[0].shape[1] == 3
    assert train_batch[1].shape[1] == 1
    assert val_batch[0].shape[1] == 3
    assert test_batch[0].shape[1] == 3


def test_lightning_datamodule_preserves_target_frame_name(toy_dataframe) -> None:
    module = LightningTabularDataModule(
        X_train=toy_dataframe[["feature"]],
        y_train=toy_dataframe["target_a"],
        X_test=toy_dataframe[["feature"]].iloc[:1],
        y_test=toy_dataframe["target_a"].iloc[:1],
        batch_size=1,
        val_size=0.5,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        seed=11,
    )

    module.setup("fit")

    assert list(module.y_test_frame_.columns) == ["target_a"]