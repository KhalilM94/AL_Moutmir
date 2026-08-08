from copy import deepcopy
from types import SimpleNamespace

import optuna
import pytest
import yaml

from yg_eo_soilnet.hpo.export import OVERRIDES_ATTR, best_overrides, build_tuned_spec, export_best_config
from yg_eo_soilnet.hpo.objective import ObjectiveContext, TrialObjective
from yg_eo_soilnet.hpo.search_space import Objective, SearchSpace
from yg_eo_soilnet.hpo.study import (
    FINGERPRINT_ATTR,
    create_or_load_study,
    default_study_name,
    ensure_storage_directory,
    reset_study,
    study_state,
    summarize,
)


class FakeDataModule:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.static_dim = 3
        self.target_dim = 1
        self.temporal_enabled = False

    def setup(self, stage=None):
        return None


class FakeModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


REGISTRY_ENTRY = {
    "enabled": False,
    "modeltype": "dl",
    "input_kind": "sequence",
    "import_path": f"{__name__}.FakeModel",
    "datamodule_import_path": f"{__name__}.FakeDataModule",
    "init_args": {"static_dim": "auto", "target_dim": "auto", "learning_rate": 0.001, "dropout": 0.1},
    "datamodule_init_args": {
        "batch_size": 32,
        "val_size": 0.3,
        "test_size": 0.15,
        "seed": 42,
        # Production throughput settings, deliberately different from what a trial would use.
        "num_workers": 11,
        "pin_memory": False,
        "persistent_workers": False,
    },
    "trainer_args": {"max_epochs": 500, "enable_checkpointing": True, "deterministic": True},
    "callbacks": {
        "early_stopping": {"monitor": "val_loss", "mode": "min", "patience": 30},
        "checkpoint": {"monitor": "val_loss", "mode": "min", "save_top_k": 1},
    },
    "tags": {"framework": "lightning"},
}

OBJECTIVE = Objective(metric="val_r2", direction="maximize")


def _space(**overrides):
    mapping = {
        "objective": {"metric": "val_r2", "direction": "maximize"},
        "fixed": {"trainer.max_epochs": 150},
        "params": {"model.dropout": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.05}},
    }
    mapping.update(overrides)
    return SearchSpace.from_mapping("fake_entry", mapping)


def _study_with_a_best_trial(values=(0.3, 0.8), storage=None, name="fake_study"):
    space = _space()
    study = (
        create_or_load_study(space, name, storage)
        if storage
        else optuna.create_study(direction="maximize", study_name=name)
    )

    context = ObjectiveContext.from_config(
        "fake_entry",
        SimpleNamespace(
            LIGHTNING_MODEL_REGISTRY={"fake_entry": deepcopy(REGISTRY_ENTRY)},
            TARGET_COLUMNS=["organic_matter_pct"],
            RANDOM_SEED=42,
        ),
        data={"sequence_bundle": object()},
    )
    objective = TrialObjective(context, space)

    call_count = {"n": 0}

    def fake_run(bundle, trial, *, report=True):
        value = values[call_count["n"] % len(values)]
        call_count["n"] += 1
        return SimpleNamespace(value=value, best_epoch=2, epochs_run=7)

    objective.runner.run = fake_run
    study.optimize(objective, n_trials=len(values))
    return study, context


# --- the tuned spec ----------------------------------------------------------


def test_the_tuned_spec_carries_the_winning_overrides():
    spec = build_tuned_spec(REGISTRY_ENTRY, {"model.dropout": 0.35, "trainer.max_epochs": 150}, OBJECTIVE)

    assert spec["init_args"]["dropout"] == 0.35
    assert spec["trainer_args"]["max_epochs"] == 150
    # Untouched registry values, including the `auto` sentinels, survive intact.
    assert spec["init_args"]["static_dim"] == "auto"
    assert spec["init_args"]["learning_rate"] == 0.001
    assert spec["tags"] == {"framework": "lightning"}


def test_the_tuned_spec_is_enabled_and_checkpoints():
    spec = build_tuned_spec(REGISTRY_ENTRY, {}, OBJECTIVE)

    assert spec["enabled"] is True
    assert spec["trainer_args"]["enable_checkpointing"] is True


def test_the_monitors_follow_the_study_objective():
    """Otherwise the production run selects a different epoch than the study scored."""
    spec = build_tuned_spec(REGISTRY_ENTRY, {}, OBJECTIVE)

    for group in ("early_stopping", "checkpoint"):
        assert spec["callbacks"][group]["monitor"] == "val_r2"
        assert spec["callbacks"][group]["mode"] == "max"
    assert spec["callbacks"]["early_stopping"]["patience"] == 30  # untouched


def test_infrastructure_settings_come_back_from_the_registry():
    """num_workers and friends are throughput knobs, not results. Production keeps its own."""
    overrides = {
        "datamodule.num_workers": 4,
        "datamodule.persistent_workers": True,
        "datamodule.pin_memory": True,
        "datamodule.batch_size": 128,
    }
    spec = build_tuned_spec(REGISTRY_ENTRY, overrides, OBJECTIVE)

    assert spec["datamodule_init_args"]["num_workers"] == 11  # the registry's, not the trial's
    assert spec["datamodule_init_args"]["persistent_workers"] is False
    assert spec["datamodule_init_args"]["pin_memory"] is False
    # A genuine tuned value on the same block is untouched.
    assert spec["datamodule_init_args"]["batch_size"] == 128


def test_an_infrastructure_key_the_registry_omits_is_dropped():
    """Removing it restores the factory's config.LIGHTNING_* default, as the registry would."""
    entry = deepcopy(REGISTRY_ENTRY)
    entry["datamodule_init_args"].pop("num_workers")
    spec = build_tuned_spec(entry, {"datamodule.num_workers": 4}, OBJECTIVE)

    assert "num_workers" not in spec["datamodule_init_args"]


def test_tuned_model_values_survive_the_infrastructure_restore():
    spec = build_tuned_spec(
        REGISTRY_ENTRY, {"model.dropout": 0.35, "datamodule.num_workers": 4}, OBJECTIVE
    )

    assert spec["init_args"]["dropout"] == 0.35
    assert spec["datamodule_init_args"]["num_workers"] == 11


def test_the_source_registry_entry_is_not_mutated():
    build_tuned_spec(REGISTRY_ENTRY, {"model.dropout": 0.9}, OBJECTIVE)

    assert REGISTRY_ENTRY["init_args"]["dropout"] == 0.1
    assert REGISTRY_ENTRY["enabled"] is False


# --- reading the best trial --------------------------------------------------


def test_best_overrides_come_from_the_trial_not_a_replay():
    study, _ = _study_with_a_best_trial()

    assert "model.dropout" in best_overrides(study)
    assert best_overrides(study)["trainer.max_epochs"] == 150


def test_a_trial_without_stored_overrides_fails_loudly():
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=1)

    with pytest.raises(KeyError, match=OVERRIDES_ATTR):
        best_overrides(study)


# --- the written file --------------------------------------------------------


def test_the_exported_file_reloads_and_builds(tmp_path):
    """The end-to-end contract: the export is a registry file that trains as-is."""
    study, context = _study_with_a_best_trial()
    path = export_best_config(study, "fake_entry", context.registry_entry, OBJECTIVE, tmp_path / "tuned.yml")

    document = yaml.safe_load(path.read_text())
    assert set(document) == {"fake_entry"}

    # Reloaded through the factory, exactly as main.py would via LIGHTNING_MODEL_REGISTRY_PATH.
    reloaded = ObjectiveContext.from_config(
        "fake_entry",
        SimpleNamespace(LIGHTNING_MODEL_REGISTRY=document, TARGET_COLUMNS=["organic_matter_pct"], RANDOM_SEED=42),
        data={"sequence_bundle": object()},
    )
    bundle = TrialObjective(reloaded, _space()).build_bundle({})

    assert bundle.model.kwargs["dropout"] == study.best_trial.params["model.dropout"]
    assert bundle.trainer_kwargs["max_epochs"] == 150
    assert bundle.model.kwargs["static_dim"] == 3  # `auto` still resolves from the datamodule


def test_the_export_carries_its_provenance(tmp_path):
    study, context = _study_with_a_best_trial()
    path = export_best_config(
        study, "fake_entry", context.registry_entry, OBJECTIVE, tmp_path / "tuned.yml", registry_path="configs/x.yml"
    )

    header = path.read_text().split("\n")
    assert any("fake_study" in line for line in header)
    assert any("val_r2" in line for line in header)
    assert any("configs/x.yml" in line for line in header)
    assert any("LIGHTNING_MODEL_REGISTRY_PATH" in line for line in header)


def test_the_export_creates_missing_directories(tmp_path):
    study, context = _study_with_a_best_trial()
    path = export_best_config(
        study, "fake_entry", context.registry_entry, OBJECTIVE, tmp_path / "a" / "b" / "tuned.yml"
    )

    assert path.exists()


# --- study persistence -------------------------------------------------------


def test_a_study_resumes_from_sqlite(tmp_path):
    """Re-running the same command must continue the study, not start a new one."""
    storage = f"sqlite:///{tmp_path}/soilnet.db"
    first, _ = _study_with_a_best_trial(values=(0.3, 0.8), storage=storage, name="resumable")
    assert len(first.trials) == 2

    second, _ = _study_with_a_best_trial(values=(0.9,), storage=storage, name="resumable")
    assert len(second.trials) == 3  # continued, not restarted
    assert second.best_value == pytest.approx(0.9)
    # Both handles are views on the one stored study, not independent copies.
    assert len(first.trials) == 3


def test_the_storage_directory_is_created(tmp_path):
    storage = f"sqlite:///{tmp_path}/nested/deeper/soilnet.db"
    ensure_storage_directory(storage)

    assert (tmp_path / "nested" / "deeper").is_dir()


def test_summarize_counts_trial_states():
    study, _ = _study_with_a_best_trial(values=(0.3, 0.8))
    summary = summarize(study, _space())

    assert summary["trials"] == 2
    assert summary["complete"] == 2
    assert summary["best_val_r2"] == pytest.approx(0.8)
    assert summary["best_trial"] == 1


def test_summarize_reports_no_best_when_nothing_completed():
    study = optuna.create_study(direction="maximize")
    summary = summarize(study, _space())

    assert summary["best_trial"] is None
    assert summary["trials"] == 0


# --- study identity ----------------------------------------------------------


def test_the_default_study_name_carries_the_space_fingerprint():
    space = _space()

    assert default_study_name("soil_cnn", space) == f"soil_cnn-{space.fingerprint()}"


def test_a_new_study_records_the_fingerprint_it_was_drawn_from(tmp_path):
    space = _space()
    study = create_or_load_study(space, "fresh", f"sqlite:///{tmp_path}/soilnet.db")

    assert study.user_attrs[FINGERPRINT_ATTR] == space.fingerprint()


def test_resuming_a_study_under_an_edited_space_is_refused(tmp_path):
    """The guard the naming scheme cannot provide: --study-name pinned across an edit.

    Direction alone let this through - both spaces here maximize, they just no longer measure the
    same thing, so `best_trial` would rank trials from two spaces against each other.
    """
    storage = f"sqlite:///{tmp_path}/soilnet.db"
    first, _ = _study_with_a_best_trial(storage=storage, name="pinned")
    edited = _space(objective={"metric": "val_pred_std_ratio", "direction": "maximize"})

    with pytest.raises(ValueError, match="drawn from search space"):
        create_or_load_study(edited, "pinned", storage)
    assert first.user_attrs[FINGERPRINT_ATTR] == _space().fingerprint()  # left untouched


def test_resuming_an_unedited_space_is_allowed(tmp_path):
    storage = f"sqlite:///{tmp_path}/soilnet.db"
    _study_with_a_best_trial(values=(0.3, 0.8), storage=storage, name="pinned")

    resumed = create_or_load_study(_space(), "pinned", storage)

    assert len(resumed.trials) == 2


def test_a_flipped_direction_is_still_refused(tmp_path):
    """Optuna silently keeps the STORED direction on load, i.e. optimizes the wrong way."""
    storage = f"sqlite:///{tmp_path}/soilnet.db"
    create_or_load_study(_space(), "pinned", storage)
    flipped = _space(objective={"metric": "val_loss", "direction": "minimize"})

    with pytest.raises(ValueError, match="optimize the wrong way"):
        create_or_load_study(flipped, "pinned", storage)


def test_changing_only_the_sampler_still_resumes(tmp_path):
    """Sampler and pruner are outside the fingerprint: they change how, not what."""
    storage = f"sqlite:///{tmp_path}/soilnet.db"
    _study_with_a_best_trial(values=(0.3, 0.8), storage=storage, name="pinned")

    resumed = create_or_load_study(_space(sampler={"name": "random", "seed": 7}), "pinned", storage)

    assert len(resumed.trials) == 2


def test_study_state_distinguishes_a_new_study_from_a_resumed_one(tmp_path):
    storage = f"sqlite:///{tmp_path}/soilnet.db"
    assert study_state(create_or_load_study(_space(), "counted", storage)) == "(new)"

    _study_with_a_best_trial(values=(0.3, 0.8), storage=storage, name="counted")

    assert study_state(create_or_load_study(_space(), "counted", storage)) == (
        "(resuming, 2 trials on record)"
    )


def test_reset_deletes_the_study_so_the_next_run_starts_clean(tmp_path):
    storage = f"sqlite:///{tmp_path}/soilnet.db"
    _study_with_a_best_trial(values=(0.3, 0.8), storage=storage, name="doomed")

    reset_study("doomed", storage)

    assert len(create_or_load_study(_space(), "doomed", storage).trials) == 0


def test_reset_on_a_study_that_does_not_exist_is_not_an_error(tmp_path):
    reset_study("never_ran", f"sqlite:///{tmp_path}/soilnet.db")


def test_reset_clears_the_way_for_an_edited_space(tmp_path):
    """The remedy the refusal points at has to actually work."""
    storage = f"sqlite:///{tmp_path}/soilnet.db"
    _study_with_a_best_trial(storage=storage, name="pinned")
    edited = _space(objective={"metric": "val_pred_std_ratio", "direction": "maximize"})

    reset_study("pinned", storage)
    study = create_or_load_study(edited, "pinned", storage)

    assert study.user_attrs[FINGERPRINT_ATTR] == edited.fingerprint()
