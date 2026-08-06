"""Fitted categorical vocabularies for the PyTorch path.

The sklearn path has had a real categorical capability for a while: a vocabulary fitted inside the
pipeline, with an explicit unknown policy (``scikit_trainer_utils.PipelineBuilder``). This module is
its deep-learning counterpart, and it is deliberately **framework-neutral** - pandas and numpy only,
no torch, no Lightning, no bundle or graph concept - so that any datamodule can fit a vocabulary here
and any model can consume the integer codes it produces. The torch side lives in
``models.lightningmodules.tabular_encoders``.

Three properties distinguish this from ``frame_cleaning.encode_categorical_features``, which it
replaces on the sequence path and which survives only for the disabled graph path:

* **The vocabulary is fitted, not derived.** ``pd.factorize`` re-derives its mapping from whatever
  frame it is handed, so the same category takes a different integer in training and in inference and
  a checkpoint cannot be applied to new data. Here the vocabulary is an object you fit once, on the
  training split alone, and carry with the model.
* **Index 0 is reserved.** Every category the training split never saw, and every missing value,
  maps to it. There is always somewhere for an unknown to go, so inference cannot fail on a category
  that simply did not exist when the model was fitted.
* **The codes are indices, not magnitudes.** Nothing here casts them to float, because the consumer
  is an ``nn.Embedding`` lookup rather than a ``nn.Linear`` that would read ``peak_ridge < valley``
  as a meaningful inequality.
"""

from __future__ import annotations

from typing import Any, Iterable, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd


#: Reserved code for "not in the fitted vocabulary" and for "missing". One shared slot rather than
#: two: with a handful of categories per column there is rarely enough signal to learn a separate
#: embedding row for absence, and a single reserved index keeps the cardinality arithmetic obvious
#: (``cardinality == len(vocabulary) + 1``).
OOV_INDEX = 0

#: Label reported for :data:`OOV_INDEX` in logs and summaries. Never a key in a vocabulary.
OOV_TOKEN = "<OOV>"


class FeatureBlocks(NamedTuple):
    """Which of a frame's feature columns are continuous and which are categorical."""

    continuous_columns: list[str]
    categorical_columns: list[str]


def _normalize_label(value: Any) -> Optional[str]:
    """Map one raw cell to a vocabulary key, or to ``None`` when it is missing.

    Missing means NaN/None/pd.NA **or** a blank string: a CSV that quotes its empty cells produces
    ``""`` rather than NaN, and treating that as a genuine category would give it an embedding row
    trained on whatever "we did not record this" happens to correlate with.
    """
    if value is None:
        return None
    # pd.isna on a scalar is safe; guard anyway so an unexpected array-like cannot raise here.
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):  # pragma: no cover - non-scalar cells are not expected
        pass
    text = str(value).strip()
    return text or None


def _as_object_matrix(values: Any, feature_names: Optional[Sequence[str]]) -> tuple[np.ndarray, list[str]]:
    """Coerce a DataFrame/2-D array-like to an ``(n_rows, n_features)`` object array plus its names."""
    if isinstance(values, pd.DataFrame):
        names = list(values.columns) if feature_names is None else list(feature_names)
        return values.to_numpy(dtype=object), [str(name) for name in names]

    matrix = np.asarray(values, dtype=object)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"Categorical values must be 2-D (rows, features), got {matrix.ndim}-D")
    names = (
        [f"categorical_{index}" for index in range(matrix.shape[1])]
        if feature_names is None
        else [str(name) for name in feature_names]
    )
    return matrix, names


class CategoricalEncoder:
    """Fits a per-column vocabulary and turns raw labels into embedding indices.

    Follows the sklearn ``fit``/``transform`` shape so it slots into the places a datamodule already
    fits statistics - on the sequence path it is fitted next to the standardization statistics, on
    the training split alone, for the same reason.

    Codes are ``1 .. len(vocabulary)``; :data:`OOV_INDEX` is reserved. So ``cardinalities[i]`` is
    ``len(vocabularies[i]) + 1`` and is exactly the ``num_embeddings`` an ``nn.Embedding`` needs.
    """

    def __init__(self) -> None:
        self._feature_names: list[str] = []
        self._vocabularies: list[list[str]] = []
        self._lookups: list[dict[str, int]] = []
        self._is_fitted = False

    # --- construction ------------------------------------------------------

    @classmethod
    def from_vocabularies(
        cls, feature_names: Sequence[str], vocabularies: Sequence[Sequence[str]]
    ) -> "CategoricalEncoder":
        """Rebuild a fitted encoder from vocabularies carried in a checkpoint.

        This is what makes a trained model applicable to a frame it has never seen: the mapping
        travels with the weights instead of being re-derived from the new data.
        """
        feature_names = [str(name) for name in feature_names]
        vocabularies = [[str(category) for category in vocabulary] for vocabulary in vocabularies]
        if len(feature_names) != len(vocabularies):
            raise ValueError(
                f"Got {len(feature_names)} feature name(s) but {len(vocabularies)} vocabulary/ies"
            )

        encoder = cls()
        encoder._feature_names = feature_names
        encoder._vocabularies = vocabularies
        encoder._lookups = [
            {category: index + 1 for index, category in enumerate(vocabulary)} for vocabulary in vocabularies
        ]
        encoder._is_fitted = True
        return encoder

    # --- fitted state ------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def vocabularies(self) -> list[list[str]]:
        """Per column, the training categories in code order. Plain ``str``, so this survives a
        checkpoint round-trip under ``torch.load(weights_only=True)``."""
        return [list(vocabulary) for vocabulary in self._vocabularies]

    @property
    def cardinalities(self) -> list[int]:
        """Per column, ``len(vocabulary) + 1`` - the reserved slot included."""
        return [len(vocabulary) + 1 for vocabulary in self._vocabularies]

    # --- fit / transform ---------------------------------------------------

    def fit(self, values: Any, feature_names: Optional[Sequence[str]] = None) -> "CategoricalEncoder":
        matrix, names = _as_object_matrix(values, feature_names)
        self._feature_names = names
        self._vocabularies = []
        self._lookups = []

        for column_index in range(matrix.shape[1]):
            observed = {
                label
                for label in (_normalize_label(cell) for cell in matrix[:, column_index])
                if label is not None
            }
            # Sorted, so the mapping depends on the set of training categories and not on the row
            # order they happened to arrive in. Two runs over shuffled copies of the same split must
            # produce the same integers, or a checkpoint's vocabulary means nothing.
            vocabulary = sorted(observed)
            self._vocabularies.append(vocabulary)
            self._lookups.append({category: index + 1 for index, category in enumerate(vocabulary)})

        self._is_fitted = True
        return self

    def transform(self, values: Any) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("CategoricalEncoder.transform called before fit")

        matrix, _ = _as_object_matrix(values, self._feature_names)
        if matrix.shape[1] != len(self._lookups):
            raise ValueError(
                f"Expected {len(self._lookups)} categorical column(s) "
                f"({', '.join(self._feature_names) or 'none'}), got {matrix.shape[1]}"
            )

        codes = np.zeros(matrix.shape, dtype=np.int64)
        for column_index, lookup in enumerate(self._lookups):
            for row_index, cell in enumerate(matrix[:, column_index]):
                label = _normalize_label(cell)
                # Unknown and missing share OOV_INDEX, which is what `codes` is already filled with.
                if label is not None:
                    codes[row_index, column_index] = lookup.get(label, OOV_INDEX)
        return codes

    def fit_transform(self, values: Any, feature_names: Optional[Sequence[str]] = None) -> np.ndarray:
        return self.fit(values, feature_names).transform(values)

    # --- reporting ---------------------------------------------------------

    def oov_fraction(self, codes: np.ndarray) -> dict[str, float]:
        """Share of rows landing on the reserved slot, per column.

        Worth logging after transform: a column that is mostly OOV on the validation split is a
        vocabulary the training split could not cover, and its embedding is close to a constant.
        """
        codes = np.asarray(codes)
        if codes.size == 0:
            return {name: 0.0 for name in self._feature_names}
        return {
            name: float((codes[:, index] == OOV_INDEX).mean())
            for index, name in enumerate(self._feature_names)
        }

    def summary(self) -> str:
        return ", ".join(
            f"{name} ({len(vocabulary)} categories + {OOV_TOKEN})"
            for name, vocabulary in zip(self._feature_names, self._vocabularies)
        )


def resolve_categorical_columns(
    config: Any,
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    logger: Any,
) -> FeatureBlocks:
    """Split schema-filtered feature columns into continuous and categorical, from the declaration.

    ``CATEGORICAL_FEATURES`` is the single source of truth, which is the point: dtype sniffing cannot
    express "this integer column is a class id", and it silently disagreed with the list the sklearn
    path reads, so the two paths trained on different predictors while claiming otherwise.

    Both failure modes raise rather than warn. A declared column that is not in the frame is usually
    a config left over from a retired dataset, and dropping it quietly is how the deep path ended up
    ignoring the declaration in the first place. An undeclared non-numeric column would previously be
    factorized into a magnitude; refusing it forces a decision instead of inventing an ordering.
    """
    declared = [str(column) for column in (getattr(config, "CATEGORICAL_FEATURES", None) or [])]
    excluded = {str(column) for column in (getattr(config, "EXCLUDE_CATEGORICAL", None) or [])}
    declared = [column for column in dict.fromkeys(declared) if column not in excluded]

    absent = [column for column in declared if column not in frame.columns]
    if absent:
        raise KeyError(
            f"CATEGORICAL_FEATURES declares column(s) absent from the data: {', '.join(absent)}. "
            "Remove them from the configuration, or add them to EXCLUDE_CATEGORICAL."
        )

    feature_columns = [column for column in feature_columns if column in frame.columns]
    declared_set = set(declared)

    categorical_columns = [column for column in feature_columns if column in declared_set]
    continuous_columns = [column for column in feature_columns if column not in declared_set]

    undeclared_non_numeric = [
        column for column in continuous_columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if undeclared_non_numeric:
        raise ValueError(
            f"Non-numeric feature column(s) not declared in CATEGORICAL_FEATURES: "
            f"{', '.join(undeclared_non_numeric)}. Declare them to give them entity embeddings, or "
            "add them to ELIMINATED_FEATURES to drop them."
        )

    # Declared but filtered out of the schema - dropped as an id, an eliminated feature or a target.
    # Legitimate, unlike the cases above, but worth saying out loud since the model will not see it.
    withheld = [column for column in declared if column not in categorical_columns]
    if withheld:
        logger.info(
            f"Declared categorical feature(s) not in the model's feature set: {', '.join(withheld)}"
        )

    return FeatureBlocks(continuous_columns=continuous_columns, categorical_columns=categorical_columns)


def split_feature_blocks(frame: pd.DataFrame, blocks: FeatureBlocks) -> tuple[np.ndarray, np.ndarray]:
    """Extract the two feature blocks as arrays: continuous float32, categorical **raw labels**.

    The labels are returned unencoded on purpose. Encoding needs a vocabulary, a vocabulary must be
    fitted on the training split alone, and at this point in the pipeline the split does not exist
    yet - fitting one here is exactly the leak this module removes.
    """
    rows = len(frame)
    continuous = (
        frame[list(blocks.continuous_columns)].to_numpy(dtype=np.float32)
        if blocks.continuous_columns
        else np.empty((rows, 0), dtype=np.float32)
    )
    categorical = (
        frame[list(blocks.categorical_columns)].to_numpy(dtype=object)
        if blocks.categorical_columns
        else np.empty((rows, 0), dtype=object)
    )
    return continuous, categorical
