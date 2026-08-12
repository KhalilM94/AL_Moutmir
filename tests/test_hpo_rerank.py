from types import SimpleNamespace

import optuna
import pytest

from yg_eo_soilnet.hpo.export import OVERRIDES_ATTR
from yg_eo_soilnet.hpo.rerank import (
    RerankResult,
    choose_winner,
    rerank,
    rerank_frame,
    top_trials,
)


def _study(values, direction="minimize", prune=(), with_overrides=True):
    """A study whose trial i has value values[i] and a recorded overrides attribute."""
    study = optuna.create_study(direction=direction)

    def objective(trial):
        index = trial.number
        trial.suggest_float("x", 0.0, 1.0)
        if with_overrides:
            trial.set_user_attr(OVERRIDES_ATTR, {"model.dropout": 0.1 * index})
        if index in prune:
            raise optuna.TrialPruned()
        return values[index]

    study.optimize(objective, n_trials=len(values))
    return study


@pytest.fixture(autouse=True)
def seeds_applied(monkeypatch):
    """The seeds rerank actually applied, in order - and no real torch seeding during tests."""
    applied: list[int] = []
    monkeypatch.setattr("yg_eo_soilnet.hpo.objective.seed_everything", applied.append)
    return applied


class FakeObjective:
    """Stands in for TrialObjective: records what it was asked to run and returns scripted values."""

    def __init__(self, per_seed, seeds_applied, seed=42):
        self.seed = seed
        self.per_seed = per_seed  # callable(overrides, seed) -> float, or raises
        self.seeds_applied = seeds_applied
        self.calls: list[tuple[dict, int]] = []
        self.runner = SimpleNamespace(run=self._run)

    def build_bundle(self, overrides):
        return SimpleNamespace(overrides=dict(overrides))

    def _run(self, bundle, trial, *, report=True):
        assert report is False, "re-runs must not report to the pruner"
        # rerank seeds immediately before building, so the last applied seed is this run's.
        current = self.seeds_applied[-1]
        self.calls.append((bundle.overrides, current))
        return SimpleNamespace(value=self.per_seed(bundle.overrides, current))


# --- top_trials --------------------------------------------------------------


def test_top_trials_orders_by_the_study_direction():
    study = _study([0.9, 0.1, 0.5])

    assert [t.number for t in top_trials(study, 3)] == [1, 2, 0]


def test_top_trials_orders_descending_when_maximizing():
    study = _study([0.9, 0.1, 0.5], direction="maximize")

    assert [t.number for t in top_trials(study, 2)] == [0, 2]


def test_top_trials_skips_pruned_trials():
    study = _study([0.9, 0.1, 0.5], prune=(1,))

    assert 1 not in [t.number for t in top_trials(study, 5)]


def test_top_trials_caps_at_the_available_count():
    assert len(top_trials(_study([0.5, 0.6]), 10)) == 2


# --- rerank ------------------------------------------------------------------


def test_the_first_seed_is_the_trials_own(seeds_applied):
    """That is what makes the pass double as a reproducibility check."""
    study = _study([0.5, 0.6])
    objective = FakeObjective(lambda overrides, seed: 0.5, seeds_applied, seed=42)

    results = rerank(objective, study, top_k=1, seeds=3)

    assert results[0].seeds == [42, 43, 44]


def test_each_candidate_runs_once_per_seed(seeds_applied):
    study = _study([0.5, 0.6, 0.7])
    objective = FakeObjective(lambda overrides, seed: 0.5, seeds_applied)

    rerank(objective, study, top_k=2, seeds=3)

    assert len(objective.calls) == 6


def test_the_recorded_overrides_are_what_gets_rerun(seeds_applied):
    study = _study([0.5, 0.6])
    objective = FakeObjective(lambda overrides, seed: 0.5, seeds_applied)

    rerank(objective, study, top_k=1, seeds=1)

    assert objective.calls[0][0] == {"model.dropout": 0.0}  # trial 0 was the best


def test_a_trial_without_recorded_overrides_is_skipped(seeds_applied):
    study = _study([0.5, 0.6], with_overrides=False)
    objective = FakeObjective(lambda overrides, seed: 0.5, seeds_applied)
    logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)

    assert rerank(objective, study, top_k=2, seeds=1, logger=logger) == []


def test_one_failing_seed_does_not_end_the_pass(seeds_applied):
    study = _study([0.5, 0.6])

    def flaky(overrides, seed):
        if seed == 43:
            raise RuntimeError("CUDA hiccup")
        return 0.55

    objective = FakeObjective(flaky, seeds_applied)
    results = rerank(objective, study, top_k=1, seeds=3)

    assert results[0].values == [0.55, 0.55]  # the seed-43 run dropped out
    assert results[0].seeds == [42, 44]


def test_a_candidate_whose_every_rerun_fails_has_no_mean(seeds_applied):
    study = _study([0.5, 0.6])
    def always_fails(overrides, seed):
        raise RuntimeError("boom")

    objective = FakeObjective(always_fails, seeds_applied)
    logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)

    results = rerank(objective, study, top_k=1, seeds=2, logger=logger)

    assert results[0].mean is None
    assert choose_winner(results, "minimize") is None


def test_rerank_of_an_empty_study_is_empty(seeds_applied):
    assert rerank(FakeObjective(lambda o, s: 0.5, seeds_applied), _study([]), top_k=5, seeds=2) == []


# --- RerankResult ------------------------------------------------------------


def test_mean_and_std_over_seeds():
    result = RerankResult(trial_number=1, original_value=0.60, overrides={}, values=[0.62, 0.64, 0.66])

    assert result.mean == pytest.approx(0.64)
    assert result.std == pytest.approx(0.02)
    assert result.drift == pytest.approx(0.04)


def test_std_of_a_single_seed_is_zero_not_an_error():
    assert RerankResult(trial_number=1, original_value=0.6, overrides={}, values=[0.62]).std == 0.0


def test_reproduced_compares_the_first_seed_against_the_trial():
    close = RerankResult(trial_number=1, original_value=0.600, overrides={}, values=[0.601, 0.7])
    far = RerankResult(trial_number=1, original_value=0.600, overrides={}, values=[0.900, 0.7])

    assert close.reproduced is True
    assert far.reproduced is False


def test_reproduced_is_unknown_without_any_run():
    assert RerankResult(trial_number=1, original_value=0.6, overrides={}).reproduced is None


# --- choose_winner -----------------------------------------------------------


def test_the_winner_is_chosen_on_the_mean_not_the_headline():
    """The whole point: the study's best trial often does not survive re-running."""
    results = [
        RerankResult(trial_number=191, original_value=0.6297, overrides={}, values=[0.660, 0.658]),
        RerankResult(trial_number=83, original_value=0.6338, overrides={}, values=[0.640, 0.642]),
    ]

    winner = choose_winner(results, "minimize")

    assert winner.trial_number == 83  # worse headline, better mean


def test_the_winner_maximizes_when_the_study_does():
    results = [
        RerankResult(trial_number=1, original_value=0.9, overrides={}, values=[0.50]),
        RerankResult(trial_number=2, original_value=0.8, overrides={}, values=[0.70]),
    ]

    assert choose_winner(results, "maximize").trial_number == 2


def test_choose_winner_ignores_candidates_with_no_usable_run():
    results = [
        RerankResult(trial_number=1, original_value=0.5, overrides={}, values=[]),
        RerankResult(trial_number=2, original_value=0.9, overrides={}, values=[0.95]),
    ]

    assert choose_winner(results, "minimize").trial_number == 2


# --- the table ---------------------------------------------------------------


def test_the_frame_carries_a_row_per_candidate():
    results = [
        RerankResult(trial_number=5, original_value=0.60, overrides={}, values=[0.62, 0.64]),
        RerankResult(trial_number=9, original_value=0.61, overrides={}, values=[0.63]),
    ]
    frame = rerank_frame(results, metric="val_loss")

    assert list(frame["trial"]) == [5, 9]
    assert "original_val_loss" in frame.columns and "mean_val_loss" in frame.columns
    assert frame.loc[0, "drift"] == pytest.approx(0.03)


def test_an_empty_frame_is_still_typed():
    frame = rerank_frame([], metric="val_r2")

    assert frame.empty
    assert "mean_val_r2" in frame.columns
