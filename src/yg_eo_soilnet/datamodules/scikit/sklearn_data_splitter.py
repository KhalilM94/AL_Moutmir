from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Callable, Dict, Optional

import mlflow
import numpy as np
import pandas as pd

from yg_eo_soilnet.datamodules.splitting import SplitPlan, TEST, TRAIN, VAL


class SklearnDataSplitter:
    """Selects this family's rows out of the run's shared :class:`SplitPlan`.

    This class used to *decide* the split - a 70/30 ``train_test_split``, or a ``GroupShuffleSplit``
    over spatial clusters. It no longer does. The decision moved to
    :mod:`yg_eo_soilnet.datamodules.splitting`, ahead of the family fork, so the Lightning
    datamodules hold out exactly the same points. Both strategies survive there, and
    ``ENABLE_CLUSTERING``/``CLUSTERING_STRATEGY`` are honoured as the legacy spelling of
    ``split.strategy: spatial_group``.

    One thing to know when reading the returned dict: **``X_train`` is the FIT POOL, train ∪ val.**
    sklearn selects hyperparameters by k-fold *inside* that pool (see ``CVSplitter``), so it has no
    use for a separate validation holdout, while Lightning early-stops on ``val``. Both then score
    on the same ``X_test``. ``X_train_only``/``X_val`` are returned alongside for the audit
    artifacts, and nothing in the training path reads them.
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def split_data(
        self,
        processed_data: Dict[str, Any],
        *,
        sanitize_features: Callable[[pd.DataFrame, list[str]], pd.DataFrame] | None = None,
        model_config_factory: Any = None,
        split_plan: Optional[SplitPlan] = None,
    ) -> Dict[str, Any]:
        X: pd.DataFrame = processed_data["X"]
        y: pd.DataFrame = processed_data["y"]
        lat = processed_data["lat"]
        lon = processed_data["lon"]
        target_columns = list(getattr(self.config, "TARGET_COLUMNS", []))

        if sanitize_features is not None:
            X = sanitize_features(X, target_columns)

        if split_plan is None:
            raise ValueError(
                "SklearnDataSplitter needs the run's split_plan. Build it with "
                "SplitPlanProvider(config, logger, data_manager).plan() before splitting, so this "
                "family and the Lightning families hold out the same points."
            )

        point_ids = self._point_ids(processed_data, X)
        labels = split_plan.labels_for(point_ids).to_numpy()

        train_pos = np.flatnonzero(labels == TRAIN)
        val_pos = np.flatnonzero(labels == VAL)
        test_pos = np.flatnonzero(labels == TEST)
        # train ∪ val, in the frame's own order so the fit pool is not silently reordered.
        fit_pos = np.sort(np.concatenate([train_pos, val_pos])) if len(val_pos) else train_pos
        unassigned = int(len(labels) - (len(train_pos) + len(val_pos) + len(test_pos)))

        self.logger.info(
            f"Applying the shared split plan ({split_plan.strategy}, "
            f"policy={split_plan.population_policy}): fit pool={len(fit_pos)} "
            f"(train={len(train_pos)} + val={len(val_pos)}) | test={len(test_pos)} rows"
        )
        if unassigned:
            self.logger.info(
                f"{unassigned} row(s) are outside the shared split population and are used by no "
                f"split. That is population_policy={split_plan.population_policy} doing its job."
            )
        if len(test_pos) == 0:
            raise ValueError(
                "The shared split plan assigned no test rows to the sklearn family. Check "
                "split.test_size and split.population_policy."
            )

        split_data: Dict[str, Any] = {}

        X_train, y_train = X.iloc[fit_pos], y.iloc[fit_pos]
        X_test, y_test = X.iloc[test_pos], y.iloc[test_pos]
        lat_train, lat_test = lat.iloc[fit_pos], lat.iloc[test_pos]
        lon_train, lon_test = lon.iloc[fit_pos], lon.iloc[test_pos]

        if split_plan.clusters is not None:
            # GroupKFold inside the fit pool needs a group per row; it is the same clustering the
            # holdout was blocked on, so the inner folds respect the same spatial structure.
            groups = split_plan.clusters.reindex(pd.Index(point_ids))
            groups.index = X.index
            split_data["groups_train"] = groups.iloc[fit_pos]
            split_data["groups_test"] = groups.iloc[test_pos]
            self.logger.info(
                f"Fit-pool group distribution:\n"
                f"{split_data['groups_train'].value_counts().sort_index().to_string()}"
            )
            self.logger.info(
                f"Test group distribution:\n"
                f"{split_data['groups_test'].value_counts().sort_index().to_string()}"
            )

        point_id_series = pd.Series(np.asarray(point_ids), index=X.index, name="point_id")
        self._log_split_artifacts(
            split_plan,
            point_ids=point_id_series,
            frames={
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            },
        )

        split_data["X_train"] = X_train
        split_data["X_test"] = X_test
        split_data["y_train"] = y_train
        split_data["y_test"] = y_test
        split_data["lat_train"] = lat_train
        split_data["lat_test"] = lat_test
        split_data["lon_train"] = lon_train
        split_data["lon_test"] = lon_test
        # Audit-only, so the fit pool can be decomposed after the fact. Nothing in the sklearn
        # training path reads these; GridSearchCV's k-fold is the validation mechanism.
        split_data["X_train_only"] = X.iloc[train_pos]
        split_data["y_train_only"] = y.iloc[train_pos]
        split_data["X_val"] = X.iloc[val_pos]
        split_data["y_val"] = y.iloc[val_pos]
        split_data["point_ids"] = point_id_series
        split_data["split_labels"] = pd.Series(labels, index=X.index, name="split")
        split_data["split_plan"] = split_plan

        return split_data

    def _point_ids(self, processed_data: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
        point_ids = processed_data.get("point_ids")
        if point_ids is None:
            raise KeyError(
                "processed_data carries no 'point_ids'. TabularPreprocessor.preprocess_data adds "
                "it; a hand-built dict must too, because the shared split is keyed on point id."
            )
        if isinstance(point_ids, pd.Series):
            # sanitize_features drops columns, not rows, so the labels still line up - but reindex
            # rather than assume it, since a mismatch here would silently mis-key the whole split.
            return point_ids.reindex(X.index).to_numpy()
        point_ids = np.asarray(point_ids)
        if len(point_ids) != len(X):
            raise ValueError(
                f"processed_data['point_ids'] has {len(point_ids)} entries for {len(X)} feature "
                f"rows; the shared split cannot be keyed."
            )
        return point_ids

    def _log_split_artifacts(
        self, split_plan: SplitPlan, *, point_ids: pd.Series, frames: Dict[str, pd.DataFrame]
    ) -> None:
        """Write the split to MLflow, with a join key this time.

        The previous version wrote four parquet files with ``index=False``, and ``filter_schema``
        had already stripped the id column out of X - so the artifacts named no rows and the split
        could not be reconstructed from a finished run. Every frame now carries ``point_id``, and
        the plan itself is written as one assignment table.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            split_start = time.perf_counter()
            for name, frame in frames.items():
                keyed = frame.copy()
                keyed.insert(0, "point_id", point_ids.reindex(frame.index).to_numpy())
                keyed.to_parquet(os.path.join(tmpdir, f"{name}.parquet"), index=False)
            split_plan.to_frame().to_parquet(
                os.path.join(tmpdir, "split_assignments.parquet"), index=False
            )
            self.logger.info(
                f"split_data parquet writes completed in {time.perf_counter() - split_start:.2f}s"
            )

            artifact_start = time.perf_counter()
            mlflow.log_artifacts(tmpdir, artifact_path="data_splits")
            self.logger.info(
                f"split_data mlflow.log_artifacts completed in {time.perf_counter() - artifact_start:.2f}s"
            )
