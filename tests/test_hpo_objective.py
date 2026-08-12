from types import SimpleNamespace

import optuna
import pytest

from yg_eo_soilnet.hpo.objective import OVERRIDES_ATTR, ObjectiveContext, TrialObjective
from yg_eo_soilnet.hpo.search_space import SearchSpace
from yg_eo_soilnet.hpo.trial_runner import UnrecoverableAcceleratorError


class FakeDataModule:
    """Mirrors the shape contract SoilSequenceDataModule exposes to the factory."""

    instances = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.static_dim = 3
        self.target_dim = 1
        self.temporal_enabled = False
        self.target_mean_ = [1.0]
        self.target_scale_ = [2.0]
        self.target_transform = "log1p"
        FakeDataModule.instances += 1

    def setup(self, stage=None):
        return None


class FakeModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


REGISTRY = {
    "fake_entry": {
        "enabled": False,  # the study must not care what the file has switched on
        "modeltype": "dl",
        "input_kind": "sequence",
        "import_path": f"{__name__}.FakeModel",
        "datamodule_import_path": f"{__name__}.FakeDataModule",
        "init_args": {"static_dim": "auto", "target_dim": "auto", "learning_rate": 0.001, "dropout": 0.1},
        "datamodule_init_args": {"batch_size": 32, "val_size": 0.3, "test_size": 0.15, "seed": 42},
        "trainer_args": {"max_epochs": 500, "deterministic": True},
        "callbacks": {"early_stopping": {"monitor": "val_loss", "mode": "min", "patience": 30}},
    }
}


def _config(**overrides):
    base = dict(LIGHTNING_MODEL_REGISTRY=REGISTRY, TARGET_COLUMNS=["organic_matter_pct"], RANDOM_SEED=42)
    base.update(overrides)
    return SimpleNamespace(**base)


def _space(**overrides):
    mapping = {
        "objective": {"metric": "val_r2", "direction": "maximize"},
        "fixed": {"trainer.max_epochs": 150},
        "params": {"model.dropout": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.05}},
    }
    mapping.update(overrides)
    return SearchSpace.from_mapping("fake_entry", mapping)


def _objective(context=None, space=None, **kwargs):
    context = context or ObjectiveContext.from_config("fake_entry", _config(), data={"sequence_bundle": object()})
    return TrialObjective(context, space or _space(), **kwargs)


# --- context -----------------------------------------------------------------


def test_context_combines_multiple_targets_like_main_does():
    context = ObjectiveContext.from_config("fake_entry", _config(TARGET_COLUMNS=["a", "b"]), data={})

    assert context.target == "a__b"


def test_context_uses_the_single_target_verbatim():
    assert ObjectiveContext.from_config("fake_entry", _config(), data={}).target == "organic_matter_pct"


def test_a_missing_registry_entry_names_what_is_available():
    with pytest.raises(KeyError, match="fake_entry"):
        ObjectiveContext.from_config("no_such_entry", _config(), data={})


def test_context_snapshots_the_registry_entry():
    """Trials mutate copies; the pristine entry must survive the study."""
    context = ObjectiveContext.from_config("fake_entry", _config(), data={})
    context.registry_entry["init_args"]["dropout"] = 0.99

    assert REGISTRY["fake_entry"]["init_args"]["dropout"] == 0.1


# --- bundle construction -----------------------------------------------------


def test_overrides_reach_the_model_and_the_disabled_entry_is_still_built():
    bundle = _objective().build_bundle({"model.dropout": 0.35, "trainer.max_epochs": 150})

    assert bundle.model.kwargs["dropout"] == 0.35
    assert bundle.trainer_kwargs["max_epochs"] == 150
    # Untouched registry values survive, and `auto` is still resolved from the datamodule.
    assert bundle.model.kwargs["learning_rate"] == 0.001
    assert bundle.model.kwargs["static_dim"] == 3
    assert bundle.model.kwargs["target_dim"] == 1


def test_datamodule_overrides_reach_the_datamodule():
    bundle = _objective().build_bundle({"datamodule.batch_size": 64})

    assert bundle.datamodule.kwargs["batch_size"] == 64


def test_the_supplied_sequence_bundle_is_reused_rather_than_rebuilt():
    """Rebuilding from the raw CSVs per trial would dominate the cost of a study."""
    payload = object()
    context = ObjectiveContext.from_config("fake_entry", _config(), data={"sequence_bundle": payload})
    bundle = TrialObjective(context, _space()).build_bundle({})

    assert bundle.datamodule.kwargs["sequence_bundle"] is payload


def test_the_datamodule_cache_reuses_one_instance_across_trials():
    context = ObjectiveContext.from_config(
        "fake_entry", _config(), data={"sequence_bundle": object()}, datamodule_cache={}
    )
    objective = TrialObjective(context, _space())

    FakeDataModule.instances = 0
    first = objective.build_bundle({"model.dropout": 0.1})
    second = objective.build_bundle({"model.dropout": 0.4})
    third = objective.build_bundle({"datamodule.batch_size": 64})

    assert first.datamodule is second.datamodule  # only model args changed
    assert third.datamodule is not first.datamodule  # a datamodule arg changed
    assert FakeDataModule.instances == 2


def test_without_a_cache_every_trial_gets_a_fresh_datamodule():
    objective = _objective()

    FakeDataModule.instances = 0
    first = objective.build_bundle({})
    second = objective.build_bundle({})

    assert first.datamodule is not second.datamodule
    assert FakeDataModule.instances == 2


# --- the objective call ------------------------------------------------------


def _run_one_trial(objective, values):
    """Drive one trial with a stubbed runner that returns `values` in order."""
    calls = []

    def fake_run(bundle, trial, *, report=True):
        calls.append((bundle, report))
        return SimpleNamespace(value=values[len(calls) - 1], best_epoch=3, epochs_run=9)

    objective.runner.run = fake_run
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)
    return study.trials[0], calls


def test_the_resolved_overrides_are_stored_on_the_trial():
    """Export reads this back; a conditional draw cannot be replayed outside a live trial."""
    trial, _ = _run_one_trial(_objective(), [0.6])

    overrides = trial.user_attrs[OVERRIDES_ATTR]
    assert overrides["trainer.max_epochs"] == 150
    assert "model.dropout" in overrides
    assert trial.value == 0.6
    assert trial.user_attrs["best_epoch"] == 3


def test_seed_repeats_average_and_report_only_once():
    trial, calls = _run_one_trial(_objective(seed_repeats=3), [0.2, 0.4, 0.6])

    assert trial.value == pytest.approx(0.4)
    assert trial.user_attrs["seed_values"] == [0.2, 0.4, 0.6]
    assert [report for _, report in calls] == [True, False, False]


def test_workers_are_released_once_per_seed_repeat(monkeypatch):
    """Without this the Trainer cycle survives and the next trial's fork inherits its iterators."""
    releases = []
    monkeypatch.setattr(
        "yg_eo_soilnet.hpo.objective.release_dataloader_workers", lambda: releases.append(1)
    )

    _run_one_trial(_objective(seed_repeats=3), [0.1, 0.2, 0.3])

    assert len(releases) == 3


def test_workers_are_released_even_when_the_trial_is_pruned(monkeypatch):
    """A pruned trial raises out of run(); the traceback keeps the frame, so the finally matters."""
    releases = []
    monkeypatch.setattr(
        "yg_eo_soilnet.hpo.objective.release_dataloader_workers", lambda: releases.append(1)
    )

    objective = _objective()

    def pruning_run(bundle, trial, *, report=True):
        raise optuna.TrialPruned("pruned in the test")

    objective.runner.run = pruning_run
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    assert study.trials[0].state.name == "PRUNED"
    assert len(releases) == 1


def test_release_dataloader_workers_finalizes_a_reference_cycle():
    """The mechanism itself: Lightning's Trainer graph is cyclic, so only a collection frees it.

    Until it is freed its DataLoader iterators stay alive to be inherited by the next fork.
    """
    import weakref

    from yg_eo_soilnet.hpo.trial_runner import release_dataloader_workers

    finalized = []

    class Cyclic:
        def __init__(self):
            self.self_reference = self  # the cycle refcounting cannot break

        def __del__(self):
            finalized.append(1)

    reference = weakref.ref(Cyclic())
    assert reference() is not None  # refcounting alone never frees this

    release_dataloader_workers()

    assert reference() is None
    assert finalized == [1]


def test_each_trial_is_seeded_before_the_model_is_built(monkeypatch):
    """LightningTrainer seeds after construction, so weight init is unseeded on that path."""
    seeded: list[int] = []
    monkeypatch.setattr("yg_eo_soilnet.hpo.objective.seed_everything", lambda seed: seeded.append(seed))

    objective = _objective(seed=7, seed_repeats=2)
    _run_one_trial(objective, [0.1, 0.2])

    assert seeded == [7, 8]


# --- a lost accelerator must stop the study, not kill the process -------------


def test_a_dead_accelerator_during_seeding_aborts_the_study(monkeypatch):
    """The gap that ended a 266-trial study: seed_everything sat outside every guard.

    It reaches torch.cuda.manual_seed_all, so once the device is gone it raises before the runner
    is ever entered - and the raw error escaped study.optimize and killed tune.py.
    """
    objective = _objective()
    monkeypatch.setattr(
        "yg_eo_soilnet.hpo.objective.seed_everything",
        lambda seed: (_ for _ in ()).throw(RuntimeError("CUDA error: unknown error")),
    )
    monkeypatch.setattr("yg_eo_soilnet.hpo.trial_runner.cuda_context_is_dead", lambda: True)

    with pytest.raises(UnrecoverableAcceleratorError, match="died during trial"):
        objective(optuna.create_study(direction="maximize").ask())


def test_a_live_accelerator_lets_a_seeding_failure_propagate_as_itself(monkeypatch):
    """Only a dead device is special. Anything else keeps its own type and traceback."""
    objective = _objective()
    monkeypatch.setattr(
        "yg_eo_soilnet.hpo.objective.seed_everything", lambda seed: (_ for _ in ()).throw(ValueError("bad seed"))
    )
    monkeypatch.setattr("yg_eo_soilnet.hpo.trial_runner.cuda_context_is_dead", lambda: False)

    with pytest.raises(ValueError, match="bad seed"):
        objective(optuna.create_study(direction="maximize").ask())


def test_workers_are_still_released_when_seeding_fails(monkeypatch):
    """The finally must survive the new except, or a dead trial leaks its DataLoader workers."""
    released = []
    objective = _objective()
    monkeypatch.setattr(
        "yg_eo_soilnet.hpo.objective.seed_everything", lambda seed: (_ for _ in ()).throw(ValueError("bad seed"))
    )
    monkeypatch.setattr("yg_eo_soilnet.hpo.trial_runner.cuda_context_is_dead", lambda: False)
    monkeypatch.setattr(
        "yg_eo_soilnet.hpo.objective.release_dataloader_workers", lambda: released.append(True)
    )

    with pytest.raises(ValueError):
        objective(optuna.create_study(direction="maximize").ask())
    assert released == [True]


def test_a_pruned_trial_is_not_mistaken_for_a_device_failure(monkeypatch):
    """TrialPruned must reach Optuna untouched, whatever the probe would have said."""
    objective = _objective()
    probed = []
    monkeypatch.setattr(
        "yg_eo_soilnet.hpo.trial_runner.cuda_context_is_dead", lambda: probed.append(True) or True
    )
    monkeypatch.setattr(
        objective.runner,
        "run",
        lambda bundle, trial, report=True: (_ for _ in ()).throw(optuna.TrialPruned("pruned")),
    )

    with pytest.raises(optuna.TrialPruned):
        objective(optuna.create_study(direction="maximize").ask())
    assert probed == []
