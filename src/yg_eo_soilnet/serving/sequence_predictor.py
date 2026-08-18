"""Predict with a restored sequence/CNN checkpoint, using the statistics it was trained with."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle
from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule


class SoilSequencePredictor:
    """A trained model plus its training-time preprocessing, ready to score new points.

    Usage after a run::

        model = mlflow.pytorch.load_model(model_uri)          # or Module.load_from_checkpoint(path)
        bundle = SoilSequenceBuilder(...).build(frame)        # raw frames -> a bundle
        predictions = SoilSequencePredictor(model).predict(bundle)

    The predictions come back in ORIGINAL TARGET UNITS: ``predict_step`` applies
    ``inverse_transform_targets``, which undoes the standardization and the ``10 * log1p`` transform.

    The one rule this class exists to enforce: **statistics are never re-fitted on the incoming
    points.** A serving batch is not a training split - it can be a single point - so fitting a
    scaler on it would standardize each request against itself and make a point's prediction depend
    on which other points happened to arrive with it. The stored state is installed instead, via
    :meth:`SoilSequenceDataModule.apply_preprocessing_state`.
    """

    def __init__(self, model, preprocessing_state: Mapping[str, Any] | None = None):
        self.model = model
        state = preprocessing_state
        if state is None and hasattr(model, "get_preprocessing_state"):
            state = model.get_preprocessing_state()
        if not state:
            raise ValueError(
                "This checkpoint carries no preprocessing state, so it cannot standardize raw input. "
                "It predates attach_preprocessing_state; retrain, or pass preprocessing_state "
                "explicitly from the datamodule that trained it."
            )
        self.preprocessing_state = dict(state)

    def _datamodule(self, bundle: SoilSequenceBundle, **datamodule_kwargs) -> SoilSequenceDataModule:
        datamodule = SoilSequenceDataModule(sequence_bundle=bundle, **datamodule_kwargs)
        datamodule.apply_preprocessing_state(self.preprocessing_state)
        return datamodule

    @torch.no_grad()
    def predict(
        self,
        bundle: "SoilSequenceBundle | Mapping[str, Any]",
        *,
        batch_size: int = 64,
        **datamodule_kwargs,
    ) -> np.ndarray:
        """``(n_points, target_dim)`` predictions in the target's original units."""
        bundle = SoilSequenceBundle.from_mapping(bundle)
        if bundle.num_points == 0:
            return np.empty((0, int(getattr(self.model, "target_dim", 1))), dtype=np.float64)

        datamodule = self._datamodule(bundle, **datamodule_kwargs)

        was_training = self.model.training
        self.model.eval()
        try:
            outputs = []
            for start in range(0, bundle.num_points, max(1, int(batch_size))):
                indices = np.arange(start, min(start + batch_size, bundle.num_points))
                batch = datamodule.collate(indices)
                # predict_step, not forward: forward stops in standardized log1p space and only
                # predict_step inverts it. Calling forward here would return predictions that look
                # plausible and are in the wrong units.
                outputs.append(np.asarray(self.model.predict_step(batch, 0).detach().cpu(), dtype=np.float64))
            predictions = np.concatenate(outputs, axis=0)
        finally:
            if was_training:
                self.model.train()

        return predictions.reshape(bundle.num_points, -1)

    def predict_frame(self, bundle, **kwargs):
        """:meth:`predict` as a DataFrame, one column per target name, indexed by point id."""
        import pandas as pd

        bundle = SoilSequenceBundle.from_mapping(bundle)
        predictions = self.predict(bundle, **kwargs)
        names = list(self.preprocessing_state.get("target_names") or []) or [
            f"target_{index}" for index in range(predictions.shape[1])
        ]
        names = names[: predictions.shape[1]]
        return pd.DataFrame(predictions, columns=names, index=list(bundle.point_ids))
