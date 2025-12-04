import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import traceback

import torch
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from .training_logger import TrainingLogger
from .mlflow_loggers import ChildRunLogger
from .trainer_utils import CVSplitter, TargetNanFilter
from .misc_utils import LogTransformer


class ModelTrainer:
    def __init__(
        self,
        config,
        columns_to_transform: Optional[List[str]] = None,
        enable_clustering: bool = False,
        split_strategy: str = 'kfold',
        seed: int = 42,
        n_splits: int = 5,
        logger=None,
        batch_size: int = 64,
        max_epochs: int = 100,
        lr: float = 1e-3,
        patience: int = 10,
        accelerator: str = "auto",
        devices=None,
        num_workers: int = 4,
        verbose: bool = True,
    ):
        self.config = config
        self.columns_to_transform = columns_to_transform or []
        self.enable_clustering = enable_clustering
        self.split_strategy = split_strategy
        self.seed = seed
        self.n_splits = n_splits
        self.logger = logger or TrainingLogger().get_logger()
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.patience = patience
        self.accelerator = accelerator
        self.devices = devices
        self.num_workers = num_workers
        self.verbose = verbose

        self.log_transformer = LogTransformer()

    def train(
        self,
        target: str,
        data: Dict,
        model_pipelines: Dict[str, Dict],
    ):
        """
        Train models for a specific target using Lightning.
        Only handles DL models; skips scikit-learn ML pipelines.
        """
        X_train = data['X_train']
        y_train = data['y_train'][target]
        X_test = data.get('X_test')
        y_test = data.get('y_test', {}).get(target)
        groups_train = data.get('groups_train') if self.enable_clustering else None

        # Remove NaNs
        X_train, y_train, groups_train = TargetNanFilter().transform(X_train, y_train, groups_train)
        if X_test is not None and y_test is not None:
            X_test, y_test, _ = TargetNanFilter().transform(X_test, y_test)
        else:
            X_test, y_test = None, None

        if self._should_skip_target(y_train, y_test, target):
            return

        mlflow_logger = ChildRunLogger()

        for model_name, config in model_pipelines.items():
            try:
                self.logger.info(f"Training {model_name} for target {target}")

                modeltype = config.get("modeltype", "dl")
                if modeltype != "dl":
                    continue  # skip non-DL models

                model = config["model"]
                init_args = config.get("init_args", {})

                # CV splitting
                cv_splitter = CVSplitter(cv_strategy=self.split_strategy, n_splits=self.n_splits, random_state=self.seed)
                splits = cv_splitter.create_splits(X_train, y_train, groups_train)

                for fold_idx, (train_idx, val_idx) in enumerate(splits):
                    self.logger.info(f"Fold {fold_idx + 1}/{len(splits)}")

                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                    # Convert to tensors
                    X_tr_tensor = torch.tensor(X_tr.values, dtype=torch.float32)
                    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
                    y_tr_tensor = torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1)
                    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

                    train_ds = TensorDataset(X_tr_tensor, y_tr_tensor)
                    val_ds = TensorDataset(X_val_tensor, y_val_tensor)

                    train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
                    val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

                    # Trainer with early stopping + checkpoint
                    checkpoint_cb = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1)
                    early_stop_cb = EarlyStopping(monitor="val/loss", patience=self.patience, mode="min", verbose=self.verbose)

                    trainer = L.Trainer(
                        max_epochs=self.max_epochs,
                        accelerator=self.accelerator,
                        devices=self.devices,
                        callbacks=[checkpoint_cb, early_stop_cb],
                        enable_progress_bar=self.verbose,
                        logger=False
                    )

                    # Fit
                    trainer.fit(model, train_loader, val_loader)

                    # Optionally log metrics
                    mlflow_logger.log_child_run(
                        config=self.config,
                        model_name=model_name,
                        fold=fold_idx,
                        model=model,
                        X_train=X_tr,
                        y_train=y_tr,
                        X_val=X_val,
                        y_val=y_val,
                        target=target
                    )

            except Exception as e:
                self.logger.warning(f"Training failed for {model_name} on {target}: {e}")
                print(f"Exception caught:\n{traceback.format_exc()}")

    def _should_skip_target(self, y_train: pd.Series, y_test: Optional[pd.Series], target: str) -> bool:
        n_train = len(y_train) if y_train is not None else 0
        n_test = len(y_test) if y_test is not None else 0
        self.logger.info(f"Training for target: {target} — {n_train} train samples, {n_test} test samples.")
        if y_train is None or y_train.empty or (y_test is not None and hasattr(y_test, 'empty') and y_test.empty):
            self.logger.warning(f"Skipping {target} — no valid data after filtering NaNs.")
            return True
        return False
