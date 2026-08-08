import optuna
import pytest

from yg_eo_soilnet.hpo.plots import optimization_history, param_importances, write_study_artifacts
from yg_eo_soilnet.hpo.search_space import Objective
from yg_eo_soilnet.hpo.tracker import BLOCKS, GAP, ObjectiveTracker, best_value_or_none

MAXIMIZE = Objective(metric="val_r2", direction="maximize")
MINIMIZE = Objective(metric="val_loss", direction="minimize")


def _study(values, direction="maximize", prune_at=()):
    """A real study whose trials take the given values; indices in `prune_at` are pruned."""
    study = optuna.create_study(direction=direction)

    def objective(trial):
        index = trial.number
        trial.suggest_float("x", 0.0, 1.0)
        trial.set_user_attr("best_epoch", index * 2)
        trial.set_user_attr("epochs_run", index * 3)
        if index in prune_at:
            raise optuna.TrialPruned()
        return values[index]

    study.optimize(objective, n_trials=len(values))
    return study


def _tracked(values, **kwargs):
    study = _study(values, **kwargs)
    tracker = ObjectiveTracker(MINIMIZE if kwargs.get("direction") == "minimize" else MAXIMIZE)
    tracker.prime(study)
    return study, tracker


# --- best_value_or_none: the regression this was written for -----------------


def test_best_value_is_none_when_nothing_completed():
    """optuna raises here rather than returning None, so the obvious guard blows up."""
    study = optuna.create_study(direction="maximize")

    assert best_value_or_none(study) is None


def test_a_study_whose_first_trial_is_pruned_does_not_raise():
    study, tracker = _tracked([0.5, 0.7], prune_at=(0,))

    assert best_value_or_none(study) == pytest.approx(0.7)
    assert tracker.summary_line().startswith("best=0.7000")


# --- sparkline edge cases ----------------------------------------------------


def test_an_empty_history_has_no_sparkline():
    assert ObjectiveTracker(MAXIMIZE).sparkline() == ""


def test_a_history_of_only_pruned_trials_is_all_gaps():
    _, tracker = _tracked([0.1, 0.2, 0.3], prune_at=(0, 1, 2))

    assert tracker.sparkline() == GAP * 3


def test_a_constant_history_does_not_divide_by_zero():
    """min == max is a real early-study state, not an error."""
    _, tracker = _tracked([0.5, 0.5, 0.5])
    spark = tracker.sparkline()

    assert len(spark) == 3
    assert set(spark) == {BLOCKS[len(BLOCKS) // 2]}


def test_a_single_value_renders_one_block():
    _, tracker = _tracked([0.42])

    assert len(tracker.sparkline()) == 1
    assert tracker.sparkline() != GAP


def test_the_sparkline_ranks_low_to_high_and_marks_gaps():
    _, tracker = _tracked([0.0, 0.5, 1.0, 0.9], prune_at=(3,))
    spark = tracker.sparkline()

    assert spark[0] == BLOCKS[0]
    assert spark[2] == BLOCKS[-1]
    assert spark[1] not in (BLOCKS[0], BLOCKS[-1])
    assert spark[3] == GAP


def test_a_pruned_trial_is_a_gap_even_though_optuna_gives_it_a_value():
    """Optuna carries the last intermediate value onto a PRUNED trial, so `value` cannot be the
    test for completion - only the state can."""
    study = optuna.create_study(direction="maximize")

    def objective(trial):
        trial.suggest_float("x", 0.0, 1.0)
        trial.report(0.9, step=0)  # a value Optuna will keep on the pruned trial
        raise optuna.TrialPruned()

    study.optimize(objective, n_trials=1)
    tracker = ObjectiveTracker(MAXIMIZE)
    tracker.prime(study)

    assert study.trials[0].value is not None  # the trap
    assert tracker.records[0].is_complete is False
    assert tracker.sparkline() == GAP


def test_the_sparkline_is_capped_to_its_width():
    _, tracker = _tracked([i / 20 for i in range(20)])

    assert len(tracker.sparkline(width=5)) == 5


# --- accumulation ------------------------------------------------------------


def test_prime_seeds_a_resumed_study():
    _, tracker = _tracked([0.1, 0.2, 0.3])

    assert len(tracker.records) == 3
    assert tracker.best_trial == 2


def test_record_appends_and_refreshes_the_best():
    study = optuna.create_study(direction="maximize")
    tracker = ObjectiveTracker(MAXIMIZE)
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=2, callbacks=[tracker.record])

    assert len(tracker.records) == 2
    assert tracker.best_value == pytest.approx(study.best_value)


def test_counts_split_complete_from_pruned():
    _, tracker = _tracked([0.1, 0.2, 0.3, 0.4], prune_at=(1, 2))

    assert tracker.counts == {"complete": 2, "pruned": 2, "failed": 0}
    assert "pruned2" in tracker.summary_line()


def test_summary_line_reports_no_best_before_anything_completes():
    assert ObjectiveTracker(MAXIMIZE).summary_line().startswith("best=n/a")


# --- frames ------------------------------------------------------------------


def test_the_top_frame_ranks_by_the_objective_direction():
    study, tracker = _tracked([0.1, 0.9, 0.5])
    frame = tracker.top_frame(study, n=2)

    assert list(frame["trial"]) == [1, 2]
    assert list(frame.columns) == ["trial", "val_r2", "best_epoch", "epochs_run", "duration_s"]


def test_the_top_frame_ranks_ascending_when_minimizing():
    study, tracker = _tracked([0.9, 0.1, 0.5], direction="minimize")

    assert list(tracker.top_frame(study, n=2)["trial"]) == [1, 2]


def test_the_top_frame_excludes_pruned_trials():
    study, tracker = _tracked([0.1, 0.9], prune_at=(1,))

    assert list(tracker.top_frame(study)["trial"]) == [0]


def test_the_top_frame_is_empty_but_typed_with_no_completed_trials():
    study, tracker = _tracked([0.1], prune_at=(0,))
    frame = tracker.top_frame(study)

    assert frame.empty
    assert "val_r2" in frame.columns


def test_the_trials_frame_carries_every_trial():
    study, tracker = _tracked([0.1, 0.2], prune_at=(1,))

    assert len(tracker.trials_frame(study)) == 2


# --- plots and artifacts -----------------------------------------------------


def test_optimization_history_returns_a_figure():
    study, _ = _tracked([0.1, 0.5, 0.3])
    figure = optimization_history(study)

    assert hasattr(figure, "savefig")  # an Axes would not; optuna returns one and we adapt it


def test_plots_degrade_to_a_message_figure_when_there_is_no_data():
    empty = optuna.create_study(direction="maximize")

    assert hasattr(optimization_history(empty), "savefig")
    assert hasattr(param_importances(empty), "savefig")


def test_param_importances_needs_two_completed_trials():
    study, _ = _tracked([0.4])

    assert hasattr(param_importances(study), "savefig")


def test_write_study_artifacts_produces_all_three_files(tmp_path):
    study, tracker = _tracked([0.1, 0.5, 0.3])
    written = write_study_artifacts(study, tracker, tmp_path, log_to_mlflow=False)

    names = {path.name for path in written}
    assert names == {"trials.csv", "optimization_history.png", "param_importances.png"}
    assert all(path.exists() and path.stat().st_size > 0 for path in written)


def test_write_study_artifacts_creates_missing_directories(tmp_path):
    study, tracker = _tracked([0.2, 0.4])
    written = write_study_artifacts(study, tracker, tmp_path / "a" / "b", log_to_mlflow=False)

    assert written[0].parent.is_dir()
