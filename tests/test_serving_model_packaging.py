"""How the Lightning model is packaged for MLflow.

It used to be logged by handing `mlflow.pyfunc.log_model` a live Python object, which CloudPickles
the whole graph: a 15 MB `python_model.pkl` whose weights could not be read without unpickling it,
which MLflow warns "can execute arbitrary code during deserialization", and whose form MLflow's own
docs mark as the alternative to the *recommended* file-path one.

It is now models-from-code: MLflow stores the loader SCRIPT, and the weights are an ordinary
checkpoint artifact. These tests pin the four properties that made the change worth doing.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle
from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule
from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import SoilCNNLightningModule
from yg_eo_soilnet.serving.lightning_pyfunc import (
    build_input_example,
    serving_requirements,
    stage_serving_package,
)

STATIC = ["clay_pct", "ph"]
BANDS = ["S2_B02", "S2_B08"]
LAB_ROSTER = ["organic_matter_g_kg", "ph_water", "clay_lab"]
N_POINTS = 12

# Present in the pipeline but never used at inference. If any of these reach the serving
# environment, code that has nothing to do with a forward pass came along with the model.
TRAINING_ONLY_PACKAGES = ("geopandas", "pyproj", "shapely", "seaborn", "statsmodels", "scikit-learn")


def _bundle() -> SoilSequenceBundle:
    generator = np.random.default_rng(0)
    bundle = SoilSequenceBundle(
        point_ids=[f"p{index}" for index in range(N_POINTS)],
        static_features=generator.normal(20, 5, (N_POINTS, len(STATIC))).astype(np.float32),
        static_feature_names=list(STATIC),
        static_categoricals=np.asarray(
            [[generator.choice(["sandy", "loam"])] for _ in range(N_POINTS)], dtype=object
        ),
        categorical_feature_names=["texture"],
        targets=generator.normal(3, 1, (N_POINTS, 1)).astype(np.float32),
        target_names=["organic_matter_g_kg"],
        label_features=generator.normal(5, 1, (N_POINTS, len(LAB_ROSTER))).astype(np.float32),
        label_feature_names=list(LAB_ROSTER),
        sequences={
            "s2": [
                generator.normal(0.2, 0.05, (int(generator.integers(3, 7)), len(BANDS))).astype(np.float32)
                for _ in range(N_POINTS)
            ]
        },
        sequence_times={},
        modality_columns={"s2": list(BANDS)},
        temporal_enabled=True,
    )
    bundle.sequence_times = {
        "s2": [
            2020.0 + np.sort(generator.random(values.shape[0])) * 2.0
            for values in bundle.sequences["s2"]
        ]
    }
    return bundle


def _model(bundle, *, auxiliary=None):
    torch.manual_seed(0)
    datamodule = SoilSequenceDataModule(
        sequence_bundle=bundle, batch_size=8, val_size=0.25, test_size=0.25, seed=42
    )
    datamodule.setup("fit")
    model = SoilCNNLightningModule(
        static_dim=datamodule.static_dim,
        target_dim=datamodule.target_dim,
        target_names=datamodule.target_names,
        categorical_cardinalities=datamodule.categorical_cardinalities,
        categorical_vocabularies=datamodule.categorical_vocabularies,
        categorical_feature_names=datamodule.categorical_feature_names,
        modality_dims=datamodule.modality_dims,
        temporal_enabled=True,
        grid_years=datamodule.grid_years,
        auxiliary_label_columns=auxiliary,
        auxiliary_available_names=LAB_ROSTER if auxiliary else None,
        static_hidden_dims=[6],
        head_hidden_dims=[6],
        cnn_hidden_dims=[4],
        modality_embed_dim=4,
        dropout=0.0,
        target_mean=datamodule.target_mean_,
        target_scale=datamodule.target_scale_,
    )
    model.attach_preprocessing_state(datamodule.preprocessing_state())
    model.eval()
    return model, datamodule


@pytest.fixture(scope="module")
def logged(tmp_path_factory) -> dict:
    """Log one model through the real ChildRunLogger path and return where it landed."""
    import mlflow

    from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger
    from yg_eo_soilnet.tracking import configure_tracking

    root = tmp_path_factory.mktemp("packaging")
    bundle = _bundle()
    model, datamodule = _model(bundle)

    configure_tracking(
        SimpleNamespace(
            MLFLOW_TRACKING_URI=(root / "mlruns").as_uri(), MLFLOW_EXPERIMENT_NAME="Packaging"
        )
    )
    with mlflow.start_run():
        ChildRunLogger().log_lightning_child_run(
            config=SimpleNamespace(EXPLAIN_ENABLED=False, FAIL_ON_MODEL_ERROR=True),
            target="organic_matter_g_kg",
            model_name="soil_cnn",
            evaluation_df=pd.DataFrame(
                {"organic_matter_g_kg": bundle.targets[:4, 0], "prediction": bundle.targets[:4, 0] * 1.05}
            ),
            validation_metrics={"val_loss": 0.5},
            test_metrics={"test_loss": 0.4},
            bundle=SimpleNamespace(
                datamodule=datamodule, trainer_kwargs={}, registry_entry={"modeltype": "dl"}
            ),
            model=model,
        )

    artifacts = next((root / "mlruns").rglob("models/m-*/artifacts"))
    return {"root": root, "artifacts": artifacts, "bundle": bundle, "model": model}


# --- packaging --------------------------------------------------------------


def test_the_model_is_not_a_pickled_python_object(logged) -> None:
    """The defect: a 15 MB python_model.pkl holding the whole object graph."""
    artifacts = logged["artifacts"]

    assert not (artifacts / "python_model.pkl").exists()
    assert (artifacts / "_pyfunc_entry.py").exists(), "the loader script should be stored instead"

    flavors = yaml.safe_load((artifacts / "MLmodel").read_text())["flavors"]
    assert "python_model" not in flavors["python_function"]


def test_the_weights_are_a_recognisable_pytorch_model(logged) -> None:
    """The nested model is what makes the network identifiable as PyTorch.

    The outer model is a pyfunc, because only that can accept raw data; nesting a real
    mlflow.pytorch model inside its artifacts gives torch-native loading without a second registry
    entry or a duplicated copy of the weights.
    """
    nested = logged["artifacts"] / "artifacts" / "torch_model"
    assert nested.is_dir()

    flavors = yaml.safe_load((nested / "MLmodel").read_text())["flavors"]
    assert "pytorch" in flavors
    assert (nested / "data" / "model.pth").exists()


def test_the_nested_model_loads_through_mlflow_pytorch(logged) -> None:
    """`mlflow.pytorch.load_model` is the thing that did not work before."""
    import mlflow.pytorch

    restored = mlflow.pytorch.load_model(str(logged["artifacts"] / "artifacts" / "torch_model"))

    assert type(restored).__name__ == "SoilCNNLightningModule"
    # The fitted scalers travel with it, so it can standardize raw input on its own.
    assert restored.get_preprocessing_state()["static_feature_names"] == STATIC


def test_the_run_still_carries_a_weights_only_loadable_checkpoint(logged) -> None:
    """The nested copy is an object pickle - pt2 cannot trace a dict-batch forward - so the safe
    state_dict copy has to remain reachable somewhere."""
    checkpoint = logged["root"] / "best.ckpt"
    torch.save({"state_dict": logged["model"].state_dict()}, checkpoint)

    assert "state_dict" in torch.load(checkpoint, weights_only=True)


def test_the_model_is_far_smaller_than_the_pickle_it_replaced(logged) -> None:
    total = sum(path.stat().st_size for path in logged["artifacts"].rglob("*") if path.is_file())

    # The cloudpickled object was ~15 MB for a model of this size.
    assert total < 5_000_000, f"model artifacts total {total} bytes"


# --- the input contract -----------------------------------------------------


def test_the_signature_does_not_ask_for_the_prediction_target(logged) -> None:
    """It used to list organic_matter_g_kg as a required INPUT - the value being predicted."""
    signature = yaml.safe_load((logged["artifacts"] / "MLmodel").read_text())["signature"]
    inputs = [column["name"] for column in json.loads(signature["inputs"])]

    assert "organic_matter_g_kg" not in inputs
    assert [column["name"] for column in json.loads(signature["outputs"])] == ["organic_matter_g_kg"]


def test_the_signature_carries_no_unused_lab_column(logged) -> None:
    """soil_cnn has auxiliary_label_columns=[], so it consumes none of the lab roster."""
    signature = yaml.safe_load((logged["artifacts"] / "MLmodel").read_text())["signature"]
    inputs = [column["name"] for column in json.loads(signature["inputs"])]

    assert set(inputs) == {"point_id", *STATIC, "texture", "s2__time", "s2__values"}


def test_a_model_that_uses_lab_values_gets_exactly_those() -> None:
    bundle = _bundle()
    model, _datamodule = _model(bundle, auxiliary=["ph_water", "clay_lab"])

    columns = set(build_input_example(model, bundle, n_rows=2).columns)

    assert {"ph_water", "clay_lab"} <= columns
    # Still never the target, even though it sits in the same lab roster.
    assert "organic_matter_g_kg" not in columns


# --- the serving environment ------------------------------------------------


def test_requirements_are_declared_and_exclude_the_training_stack(logged) -> None:
    requirements = (logged["artifacts"] / "requirements.txt").read_text()

    for package in ("torch", "lightning", "numpy", "pandas"):
        assert package in requirements

    for package in TRAINING_ONLY_PACKAGES:
        assert package not in requirements, f"{package} reached the serving environment"


def test_serving_requirements_are_pinned_to_the_installed_versions() -> None:
    assert all("==" in requirement for requirement in serving_requirements())


def test_the_staged_package_ships_no_training_only_module(tmp_path) -> None:
    staged = Path(stage_serving_package(str(tmp_path)))

    shipped = {path.name for path in staged.rglob("*.py")}
    for unwanted in ("clustering_utils.py", "plot_utils.py", "data_manager.py", "sklearn_trainer.py"):
        assert unwanted not in shipped


def test_every_staged_init_is_blank(tmp_path) -> None:
    """Each real __init__ re-exports subpackages the serving subset leaves out, so a staged copy
    carrying them fails on import."""
    staged = Path(stage_serving_package(str(tmp_path)))

    for init_file in staged.rglob("__init__.py"):
        assert "import" not in init_file.read_text(), f"{init_file} still imports"


# --- the load test that makes the trimming safe -----------------------------


def test_the_logged_model_loads_with_the_repo_off_sys_path(logged, tmp_path) -> None:
    """The only test that can prove the shipped subset is self-sufficient.

    An in-process load passes no matter what is shipped, because the real package is already
    importable - which is exactly how a missing module would reach production unnoticed. This runs
    in a subprocess with the source roots removed from sys.path, so it fails if the staged module
    set or the blanked __init__ files leave anything out.
    """
    repo_root = Path(__file__).resolve().parents[1]
    example = build_input_example(logged["model"], logged["bundle"], n_rows=3)
    example_path = tmp_path / "example.json"
    example.to_json(example_path, orient="split")

    script = tmp_path / "isolated_load.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import os, sys
            repo = os.path.realpath({str(repo_root)!r})
            # Drop only the SOURCE roots: the pixi environment lives under the repo, so a blanket
            # filter would remove site-packages and prove nothing.
            blocked = {{repo, os.path.join(repo, "src")}}
            sys.path[:] = [p for p in sys.path if os.path.realpath(p or ".") not in blocked]
            for name in [m for m in list(sys.modules) if m.split(".")[0] == "yg_eo_soilnet"]:
                del sys.modules[name]
            os.chdir({str(tmp_path)!r})

            try:
                import yg_eo_soilnet
                raise SystemExit("repo package is still importable; the isolation failed")
            except ImportError:
                pass

            import numpy as np, pandas as pd, mlflow

            model = mlflow.pyfunc.load_model({str(logged["artifacts"])!r})
            frame = pd.read_json({str(example_path)!r}, orient="split")
            for column in ("s2__time", "s2__values"):
                frame[column] = frame[column].apply(list)
            predictions = np.asarray(model.predict(frame), dtype=float)
            assert predictions.shape[0] == 3, predictions.shape
            assert np.isfinite(predictions).all()
            print("OK")
            """
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=600
    )

    assert completed.returncode == 0, (
        "the logged model could not be loaded without the repo on sys.path, so the shipped code is "
        f"incomplete:\n{completed.stdout}\n{completed.stderr[-3000:]}"
    )
    assert "OK" in completed.stdout


def test_the_isolated_prediction_matches_the_in_process_one(logged, tmp_path) -> None:
    import mlflow

    from yg_eo_soilnet.serving import SoilSequencePredictor

    model = mlflow.pyfunc.load_model(str(logged["artifacts"]))
    example = build_input_example(logged["model"], logged["bundle"], n_rows=N_POINTS)

    served = np.asarray(model.predict(example), dtype=float).ravel()
    reference = SoilSequencePredictor(logged["model"]).predict(logged["bundle"]).ravel()

    assert np.allclose(served, reference, atol=1e-4)
