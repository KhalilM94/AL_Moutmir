from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

try:  # pragma: no cover - optional dependency shim
    from lightning.pytorch import LightningDataModule
except ImportError:  # pragma: no cover - keeps imports working without lightning installed
    class LightningDataModule:  # type: ignore[override]
        pass


class _TabularTensorDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        targets_array = np.asarray(targets, dtype=np.float32)
        if targets_array.ndim == 1:
            targets_array = targets_array.reshape(-1, 1)
        self.targets = torch.as_tensor(targets_array, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index]


class LightningTabularDataModule(LightningDataModule):
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series | pd.DataFrame] = None,
        batch_size: int = 32,
        val_size: float = 0.2,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        seed: int = 42,
        shuffle: bool = True,
        drop_last: bool = False,
        feature_columns: Optional[list[str]] = None,
        target_columns: Optional[list[str]] = None,
    ):
        super().__init__()
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_test = X_test.copy() if X_test is not None else None
        self.y_test = y_test.copy() if y_test is not None else None
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.feature_columns = feature_columns
        self.target_columns = target_columns

        self.feature_columns_ = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.X_test_frame_ = None
        self.y_test_frame_ = None
        self.X_train_frame_ = None
        self.y_train_frame_ = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return

        train_features = self._prepare_feature_frame(self.X_train, fit=True)
        train_targets = self._prepare_target_array(self.y_train)

        self.X_train_frame_ = train_features.copy()
        self.y_train_frame_ = self._target_as_frame(self.y_train)

        train_indices = np.arange(len(train_features))
        if 0 < self.val_size < 1 and len(train_indices) > 1:
            train_idx, val_idx = train_test_split(
                train_indices,
                test_size=self.val_size,
                random_state=self.seed,
                shuffle=True,
            )
        else:
            train_idx = train_indices
            val_idx = np.array([], dtype=int)

        self.train_dataset = _TabularTensorDataset(train_features.iloc[train_idx].to_numpy(), train_targets[train_idx])
        self.val_dataset = (
            _TabularTensorDataset(train_features.iloc[val_idx].to_numpy(), train_targets[val_idx])
            if len(val_idx) > 0
            else None
        )

        if self.X_test is not None and self.y_test is not None:
            test_features = self._prepare_feature_frame(self.X_test, fit=False)
            test_targets = self._prepare_target_array(self.y_test)
            self.X_test_frame_ = test_features.copy()
            self.y_test_frame_ = self._target_as_frame(self.y_test)
            self.test_dataset = _TabularTensorDataset(test_features.to_numpy(), test_targets)

    @property
    def feature_dim(self) -> int:
        if self.feature_columns_ is None:
            return int(self._prepare_feature_frame(self.X_train, fit=True).shape[1])
        return int(len(self.feature_columns_))

    @property
    def target_dim(self) -> int:
        if isinstance(self.y_train, pd.DataFrame):
            return int(self.y_train.shape[1])
        return 1

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("fit")
        if self.val_dataset is None:
            raise ValueError("Validation split is empty; increase LIGHTNING_VAL_SIZE or training rows.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        if self.test_dataset is None:
            raise ValueError("Test split is missing; provide X_test and y_test to the datamodule.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def _prepare_feature_frame(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        prepared = frame.copy()
        if self.feature_columns is not None:
            prepared = prepared.reindex(columns=self.feature_columns, fill_value=0)

        prepared = pd.get_dummies(prepared, drop_first=False)

        if fit or self.feature_columns_ is None:
            self.feature_columns_ = list(prepared.columns)
        else:
            prepared = prepared.reindex(columns=self.feature_columns_, fill_value=0)

        return prepared.astype(np.float32)

    def _prepare_target_array(self, target: pd.Series | pd.DataFrame) -> np.ndarray:
        if isinstance(target, pd.DataFrame):
            target_frame = target.copy()
            if self.target_columns is not None:
                target_frame = target_frame.reindex(columns=self.target_columns)
            return target_frame.to_numpy(dtype=np.float32)

        return target.to_numpy(dtype=np.float32)

    def _target_as_frame(self, target: pd.Series | pd.DataFrame) -> pd.DataFrame:
        if isinstance(target, pd.DataFrame):
            return target.copy()

        column_name = self.target_columns[0] if self.target_columns else target.name or "target"
        return target.to_frame(name=column_name)