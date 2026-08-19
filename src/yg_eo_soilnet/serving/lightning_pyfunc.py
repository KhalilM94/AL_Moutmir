"""An MLflow pyfunc wrapper for the sequence models, so a trained CNN can actually be served.

Why this exists rather than a plain ``mlflow.pytorch.log_model``: MLflow 3 defaults that call to
``serialization_format="pt2"``, a traced-graph format that runs ``model.forward`` on an example
input. ``SoilCNNLightningModule.forward`` consumes a **dict batch of ragged, date-stamped
sequences**, so there is no tensor example that can trace it - the run failed with
``If serialization_format is set to 'pt2', then input_example is required``, and passing a tensor
instead of the DataFrame would not have helped. Tracing is simply the wrong strategy for this
architecture.

The input contract is one row per point. Static covariates and categoricals are ordinary columns;
the ragged part travels as nested arrays, which MLflow schemas express as ``Array(Double)`` and
``Array(Array(Double))``:

===========================  ==========================  ===================================
column                       type                        meaning
===========================  ==========================  ===================================
``point_id``                 string, optional            echoed onto the output index
one per static feature       double                      covariates in RAW units, unscaled
one per categorical feature  string                      raw labels; unseen ones hit the OOV index
one per label feature        double, optional            only when the model uses lab auxiliaries
``<modality>__time``         ``Array(Double)``           decimal years, one per observation
``<modality>__values``       ``Array(Array(Double))``    per observation, one value per band
===========================  ==========================  ===================================

Bands are ordered by ``modality_column_names[modality]`` from the checkpoint's own preprocessing
state, so the contract is self-describing: the model says which columns it wants and in what order.

Prediction delegates to :class:`~yg_eo_soilnet.serving.sequence_predictor.SoilSequencePredictor`,
which installs the training-time statistics rather than re-fitting them on the request.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

TIME_SUFFIX = "__time"
VALUES_SUFFIX = "__values"
POINT_ID_COLUMN = "point_id"

# The only modules inference touches. Shipping the whole package instead made MLflow infer
# geopandas, pyproj, matplotlib, seaborn and statsmodels as runtime dependencies of a CNN forward
# pass, because importing yg_eo_soilnet pulls all of them through its __init__.
SERVING_MODULES: tuple[str, ...] = (
    "serving",
    "models/__init__.py",
    "models/lightningmodules",
    "datamodules/__init__.py",
    "datamodules/sequence",
    "datamodules/categorical.py",
    "datamodules/frame_cleaning.py",
    "utils.py",
)

# Every __init__.py in the staged copy is replaced with this. Mandatory, not cosmetic: every package
# __init__ in this project re-exports its subpackages, and those re-exports reach modules the
# serving subset deliberately leaves out - yg_eo_soilnet/__init__ pulls clustering_utils and
# plot_utils, datamodules/__init__ pulls the graph path, lightningmodules/__init__ pulls the graph
# module. A staged copy carrying the real ones fails on import.
#
# Blanking ALL of them rather than curating a list is safe because nothing in the serving subset
# imports from a package namespace; every import is a fully-qualified submodule import. The
# subprocess load test is what keeps that true.
_MINIMAL_INIT = '''"""Inference-only subset of yg_eo_soilnet, staged by the model logger.

Deliberately empty. The real package __init__ files re-export the training surface, which would drag
the geospatial and plotting stacks into a serving environment that only runs a forward pass. Import
submodules directly.
"""
'''


def stage_serving_package(destination: str) -> str:
    """Copy the inference modules into ``destination`` and return the path to pass as code_paths."""
    import shutil
    from pathlib import Path

    source_root = Path(__file__).resolve().parents[1]
    staged_root = Path(destination) / source_root.name
    staged_root.mkdir(parents=True, exist_ok=True)

    for relative in SERVING_MODULES:
        source = source_root / relative
        target = staged_root / relative
        if not source.exists():
            raise FileNotFoundError(f"SERVING_MODULES names {relative!r}, which does not exist")
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copyfile(source, target)

    staged_root.joinpath("__init__.py").write_text(_MINIMAL_INIT, encoding="utf-8")
    for init_file in staged_root.rglob("__init__.py"):
        init_file.write_text(_MINIMAL_INIT, encoding="utf-8")

    return str(staged_root)


def serving_requirements() -> list[str]:
    """The runtime dependencies, declared rather than inferred.

    MLflow infers requirements from the modules imported in the LOGGING process, where the whole
    training stack is already loaded, so inference reports geopandas and seaborn no matter how
    little code is shipped. Declaring is the only reliable lever. Versions are read from the
    installed packages so the serving environment matches the one that trained the model.
    """
    from importlib.metadata import PackageNotFoundError, version

    requirements = []
    for package in ("torch", "lightning", "numpy", "pandas"):
        try:
            requirements.append(f"{package}=={version(package)}")
        except PackageNotFoundError:  # pragma: no cover - all four are hard dependencies
            requirements.append(package)
    return requirements


def time_column(modality: str) -> str:
    return f"{modality}{TIME_SUFFIX}"


def values_column(modality: str) -> str:
    return f"{modality}{VALUES_SUFFIX}"


def _as_list_of_arrays(column: pd.Series, *, dtype, label: str) -> list[np.ndarray]:
    """One array per row, tolerating the several shapes a nested column arrives in.

    MLflow's JSON transport, pandas' own parquet round-trip and a hand-built frame all deliver
    nested values slightly differently (lists, numpy object arrays, tuples), so this normalises
    rather than assuming one of them.
    """
    arrays: list[np.ndarray] = []
    for position, value in enumerate(column.tolist()):
        if value is None:
            arrays.append(np.empty((0,), dtype=dtype))
            continue
        try:
            arrays.append(np.asarray(list(value), dtype=dtype))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} row {position} is not a numeric sequence: {value!r}") from exc
    return arrays


def bundle_from_frame(frame: pd.DataFrame, state: Mapping[str, Any]):
    """Rebuild a :class:`SoilSequenceBundle` from the serving contract above.

    Every column list comes from the checkpoint's stored preprocessing state, never from the
    incoming frame, so a request with extra columns is ignored rather than silently reshaping the
    model's inputs, and a request missing one is refused by name.
    """
    from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle

    static_names = list(state.get("static_feature_names") or [])
    categorical_names = list(state.get("categorical_feature_names") or [])
    label_names = list(state.get("label_feature_names") or [])
    modality_columns = {
        str(name): list(columns)
        for name, columns in (state.get("modality_column_names") or {}).items()
    }

    missing = [name for name in static_names + categorical_names if name not in frame.columns]
    if missing:
        raise KeyError(
            f"Input is missing {len(missing)} column(s) this model was trained on: "
            f"{', '.join(missing[:10])}"
        )

    n_rows = len(frame)
    point_ids = (
        frame[POINT_ID_COLUMN].tolist() if POINT_ID_COLUMN in frame.columns else list(range(n_rows))
    )

    static_features = (
        frame[static_names].to_numpy(dtype=np.float32)
        if static_names
        else np.empty((n_rows, 0), dtype=np.float32)
    )
    static_categoricals = (
        frame[categorical_names].astype(object).to_numpy()
        if categorical_names
        else np.empty((n_rows, 0), dtype=object)
    )
    # The lab block is rebuilt at FULL ROSTER WIDTH, with each supplied column at its own roster
    # position and the rest left NaN.
    #
    # Width, because the model index_selects roster positions: it reads only the few columns named
    # in auxiliary_label_columns, but it addresses them by where they sit in the roster it was built
    # against, and it refuses a batch of any other width. Handing it just the supplied columns
    # produced "Batch carries 0 lab column(s) ... against 24" and silently stopped every model being
    # logged, hence registered.
    #
    # Position rather than order of appearance, because a shifted column would feed the model a
    # different measurement under the right name without raising anything.
    #
    # NaN rather than zero, because NaN is the bundle's own convention for absent (see
    # SoilSequenceBundle.label_features) and SoilSequenceDataModule._standardize_labels median-fills
    # it from the training split and flags it in the validity channel - so a column the caller did
    # not supply arrives marked as unmeasured instead of as a real zero.
    if label_names:
        label_features = np.full((n_rows, len(label_names)), np.nan, dtype=np.float32)
        for index, name in enumerate(label_names):
            if name in frame.columns:
                label_features[:, index] = frame[name].to_numpy(dtype=np.float32)
    else:
        label_features = np.empty((n_rows, 0), dtype=np.float32)

    sequences: dict[str, list[np.ndarray]] = {}
    sequence_times: dict[str, list[np.ndarray]] = {}
    for modality, columns in modality_columns.items():
        time_name, values_name = time_column(modality), values_column(modality)
        for name in (time_name, values_name):
            if name not in frame.columns:
                raise KeyError(
                    f"Input is missing '{name}'. Modality '{modality}' needs '{time_name}' "
                    f"(decimal years) and '{values_name}' (one value per band, ordered "
                    f"{columns})."
                )

        times = _as_list_of_arrays(frame[time_name], dtype=np.float64, label=time_name)
        raw_values = _as_list_of_arrays(frame[values_name], dtype=np.float32, label=values_name)

        per_point = []
        for position, values in enumerate(raw_values):
            reshaped = values.reshape(-1, len(columns)) if values.size else np.empty(
                (0, len(columns)), dtype=np.float32
            )
            if reshaped.shape[0] != times[position].shape[0]:
                raise ValueError(
                    f"Row {position}: '{values_name}' has {reshaped.shape[0]} observation(s) but "
                    f"'{time_name}' has {times[position].shape[0]}; they must line up."
                )
            per_point.append(reshaped)

        sequences[modality] = per_point
        sequence_times[modality] = times

    return SoilSequenceBundle(
        point_ids=point_ids,
        static_features=static_features,
        static_feature_names=static_names,
        static_categoricals=static_categoricals,
        categorical_feature_names=categorical_names,
        targets=np.empty((n_rows, 0), dtype=np.float32),
        target_names=list(state.get("target_names") or []),
        label_features=label_features,
        label_feature_names=list(label_names),
        sequences=sequences,
        sequence_times=sequence_times,
        modality_columns=modality_columns,
        temporal_enabled=bool(modality_columns),
    )


def frame_from_bundle(
    bundle,
    state: Mapping[str, Any],
    n_rows: int | None = None,
    auxiliary_columns: Any = None,
) -> pd.DataFrame:
    """The inverse of :func:`bundle_from_frame`: a serving frame built from a training bundle.

    Used to produce the ``input_example`` that ships with the logged model, which is what makes the
    signature real and gives `mlflow models serve` something valid to echo back.

    ``auxiliary_columns`` is the list of lab columns the model actually consumes - normally
    ``model.auxiliary_label_columns``, which is empty for a model that does not use them. It is a
    parameter rather than being read off the state because the bundle carries the WHOLE lab roster
    whenever ``CARRY_LABEL_COLUMNS`` is on, and emitting all of it put ~20 unused columns into the
    signature as required inputs - including the prediction target itself, which made the contract
    ask for the value being predicted.
    """
    from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle

    bundle = SoilSequenceBundle.from_mapping(bundle)
    count = bundle.num_points if n_rows is None else min(int(n_rows), bundle.num_points)

    data: dict[str, Any] = {POINT_ID_COLUMN: [str(value) for value in bundle.point_ids[:count]]}

    for index, name in enumerate(state.get("static_feature_names") or []):
        data[name] = np.asarray(bundle.static_features[:count, index], dtype=np.float64)

    categoricals = np.asarray(bundle.static_categoricals, dtype=object)
    for index, name in enumerate(state.get("categorical_feature_names") or []):
        data[name] = [str(value) for value in categoricals[:count, index]]

    labels = np.asarray(bundle.label_features)
    wanted = {str(name) for name in (auxiliary_columns or [])}
    if labels.size and wanted:
        for index, name in enumerate(state.get("label_feature_names") or []):
            if name in wanted:
                data[name] = np.asarray(labels[:count, index], dtype=np.float64)

    for modality in (state.get("modality_column_names") or {}):
        data[time_column(modality)] = [
            np.asarray(values, dtype=np.float64).tolist()
            for values in bundle.sequence_times.get(modality, [])[:count]
        ]
        data[values_column(modality)] = [
            np.asarray(values, dtype=np.float64).tolist()
            for values in bundle.sequences.get(modality, [])[:count]
        ]

    return pd.DataFrame(data)


class SoilSequencePyfunc:
    """The prediction half of the served model: raw frame in, original-unit predictions out.

    Deliberately NOT a ``mlflow.pyfunc.PythonModel`` subclass here. The mixin is applied in
    ``serving/_pyfunc_entry.py``, the models-from-code script, so that importing this module - which
    the logger does on every Lightning run, and which the training path also pulls in - does not
    depend on mlflow's pyfunc stack. It is also directly usable without mlflow at all, which is what
    lets the tests compare it against SoilSequencePredictor.
    """

    def __init__(self, model=None):
        self.model = model
        self._predictor = None

    def load_context(self, context) -> None:  # pragma: no cover - exercised via mlflow
        self._predictor = None

    def _ensure_predictor(self):
        if self._predictor is None:
            from yg_eo_soilnet.serving.sequence_predictor import SoilSequencePredictor

            self._predictor = SoilSequencePredictor(self.model)
        return self._predictor

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        predictor = self._ensure_predictor()
        frame = pd.DataFrame(model_input)
        bundle = bundle_from_frame(frame, predictor.preprocessing_state)
        predictions = predictor.predict(bundle)

        names = list(predictor.preprocessing_state.get("target_names") or [])
        names = names[: predictions.shape[1]] or [
            f"prediction_{index}" for index in range(predictions.shape[1])
        ]
        return pd.DataFrame(predictions, columns=names)


def example_from_state(
    state: Mapping[str, Any],
    static_frame: pd.DataFrame | None = None,
    n_rows: int = 3,
    sequence_length: int = 4,
    auxiliary_columns: Any = None,
) -> pd.DataFrame:
    """A schema-correct serving frame built from ``preprocessing_state`` alone - no bundle needed.

    The no-dataset sibling of :func:`build_input_example`. A checkpoint carries every column name and
    every fitted statistic, so a valid example can be assembled long after the training data is out
    of reach - which is what lets a model that failed to package be recovered from its checkpoint
    rather than retrained.

    ``static_frame`` supplies real measured values for whichever static columns it happens to carry;
    anything absent falls back to the stored training mean. Means rather than zeros: the data the
    caller sends is in RAW units, and a zero would be an outlier for most covariates, whereas the
    training mean is by construction inside the range the model was fitted on.

    Note what cannot come from a run's ``eval_results.csv``: it holds the static block and the
    predictions, never the lab roster and never the ragged sequences. Those are always synthesized
    here.
    """
    static_names = list(state.get("static_feature_names") or [])
    static_mean = list(state.get("static_mean") or [])
    categorical_names = list(state.get("categorical_feature_names") or [])
    vocabularies = list(state.get("categorical_vocabularies") or [])
    label_names = list(state.get("label_feature_names") or [])
    label_mean = list(state.get("label_mean") or [])
    modality_columns = {
        str(name): list(columns)
        for name, columns in (state.get("modality_column_names") or {}).items()
    }
    sequence_mean = state.get("sequence_mean") or {}

    data: dict[str, Any] = {POINT_ID_COLUMN: [f"point_{index}" for index in range(n_rows)]}

    for index, name in enumerate(static_names):
        if static_frame is not None and name in static_frame.columns:
            column = pd.to_numeric(static_frame[name], errors="coerce").to_numpy(dtype=np.float64)
            values = np.resize(column[np.isfinite(column)], n_rows) if np.isfinite(column).any() else None
            if values is not None:
                data[name] = values
                continue
        fallback = float(static_mean[index]) if index < len(static_mean) else 0.0
        data[name] = np.full(n_rows, fallback, dtype=np.float64)

    for index, name in enumerate(categorical_names):
        # A real vocabulary entry, not a placeholder: an unknown label would land on the reserved
        # out-of-vocabulary index and quietly exercise a different embedding row than production.
        vocabulary = vocabularies[index] if index < len(vocabularies) else []
        data[name] = [str(vocabulary[0]) if vocabulary else "unknown"] * n_rows

    # Only the columns the model actually reads reach the signature; the rest of the roster is
    # reconstructed at its own position by bundle_from_frame. Passed in rather than read from the
    # state because the selection is a MODEL hyper-parameter - the state describes the data.
    for name in list(auxiliary_columns or []):
        index = label_names.index(name) if name in label_names else -1
        fallback = float(label_mean[index]) if 0 <= index < len(label_mean) else 0.0
        data[name] = np.full(n_rows, fallback, dtype=np.float64)

    for modality, columns in modality_columns.items():
        means = list(sequence_mean.get(modality) or [])
        per_band = [float(means[index]) if index < len(means) else 0.0 for index in range(len(columns))]
        data[time_column(modality)] = [
            [2020.0 + step / 4.0 for step in range(sequence_length)] for _ in range(n_rows)
        ]
        data[values_column(modality)] = [
            [list(per_band) for _ in range(sequence_length)] for _ in range(n_rows)
        ]

    return pd.DataFrame(data)


def build_input_example(model, bundle, n_rows: int = 3) -> pd.DataFrame:
    """A small, valid serving frame for the model's own training bundle.

    Asks the model which lab columns it consumes rather than offering the whole roster, so the
    inferred signature describes the inputs the model genuinely needs.
    """
    state = model.get_preprocessing_state() if hasattr(model, "get_preprocessing_state") else {}
    if not state:
        raise ValueError("The model carries no preprocessing state, so no input example can be built.")
    return frame_from_bundle(
        bundle,
        state,
        n_rows=n_rows,
        auxiliary_columns=list(getattr(model, "auxiliary_label_columns", None) or []),
    )
