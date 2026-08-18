"""The shape a SHAP explanation takes once both backends have finished with it.

Both the sklearn and the Lightning explainer return a list of these, one per target, so everything
downstream - plots, the parquet sidecar, the run summary - is written once rather than per backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Which numeric space the SHAP values are contributions TO. This is not decoration: for a
# log-transformed target the sklearn pipeline wraps the estimator in a TransformedTargetRegressor
# and the explained estimator predicts 10*log1p(y), so a value of 0.4 means 0.4 in that space and
# not 0.4 percentage points of organic matter.
ORIGINAL_UNITS = "original_units"
LOG1P_X10 = "log1p_x10"
STANDARDIZED_LOG1P = "standardized_log1p"


@dataclass
class ShapResult:
    """Per-sample SHAP values for one target, already folded down to one column per feature."""

    values: np.ndarray  # (n_samples, n_features)
    data: np.ndarray  # (n_samples, n_features), the beeswarm colour values; NaN where meaningless
    feature_names: list[str]
    target_name: str
    output_space: str
    # Which block each feature belongs to: "static", "categorical", "auxiliary", or a modality name.
    # Drives the rolled-up bar plot.
    blocks: list[str] = field(default_factory=list)
    base_value: float = 0.0
    # Which shap explainer produced these values. Recorded because the sklearn path silently falls
    # back from the exact TreeExplainer to a model-agnostic one when shap cannot parse the model -
    # an approximation the reader of a plot deserves to know about.
    explainer: str = "unknown"

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.values.shape != self.data.shape:
            raise ValueError(
                f"values {self.values.shape} and data {self.data.shape} must have the same shape"
            )
        if self.values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"{self.values.shape[1]} value column(s) but {len(self.feature_names)} feature name(s)"
            )
        if not self.blocks:
            self.blocks = ["all"] * len(self.feature_names)
        if len(self.blocks) != len(self.feature_names):
            raise ValueError(
                f"{len(self.blocks)} block label(s) but {len(self.feature_names)} feature name(s)"
            )

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])

    def mean_abs(self) -> np.ndarray:
        """Mean |SHAP| per feature - the height of each bar in the bar plot."""
        return np.abs(self.values).mean(axis=0)

    def ranking(self) -> list[int]:
        """Feature indices ordered by mean |SHAP|, most important first."""
        return list(np.argsort(-self.mean_abs()))

    def block_mean_abs(self) -> dict[str, float]:
        """Mean |summed contribution| per block.

        Summed WITHIN a block per sample before taking the absolute value, not the sum of the
        per-feature mean |SHAP|. The distinction matters: two features in one block that cancel each
        other out on the same sample contribute nothing jointly, and the block view is meant to
        answer "how much does this branch move the prediction", which is the joint quantity.
        """
        totals: dict[str, float] = {}
        for block in dict.fromkeys(self.blocks):
            columns = [index for index, name in enumerate(self.blocks) if name == block]
            totals[block] = float(np.abs(self.values[:, columns].sum(axis=1)).mean())
        return totals

    def to_frame(self) -> pd.DataFrame:
        """The COMPLETE per-sample matrix, uncapped, for the parquet sidecar.

        Long rather than wide because the feature count runs into the hundreds once every band of
        every modality has its own row, and a long frame keeps the block and colour value attached
        to each number instead of needing three parallel wide tables.
        """
        n_samples, n_features = self.values.shape
        return pd.DataFrame(
            {
                "sample": np.repeat(np.arange(n_samples), n_features),
                "feature": np.tile(np.asarray(self.feature_names, dtype=object), n_samples),
                "block": np.tile(np.asarray(self.blocks, dtype=object), n_samples),
                "shap_value": self.values.reshape(-1),
                "feature_value": self.data.reshape(-1),
                "target": self.target_name,
                "output_space": self.output_space,
            }
        )

    def summary(self, top_n: int | None = None) -> dict:
        """Ranking plus provenance, for the JSON sidecar and the run summary."""
        mean_abs = self.mean_abs()
        order = self.ranking()
        if top_n is not None:
            order = order[:top_n]
        return {
            "target": self.target_name,
            "output_space": self.output_space,
            "explainer": self.explainer,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "base_value": float(self.base_value),
            "blocks": self.block_mean_abs(),
            "ranking": [
                {
                    "feature": self.feature_names[index],
                    "block": self.blocks[index],
                    "mean_abs_shap": float(mean_abs[index]),
                }
                for index in order
            ],
        }
