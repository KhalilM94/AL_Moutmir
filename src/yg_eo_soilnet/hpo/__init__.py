"""Model-agnostic Optuna hyperparameter search over the Lightning registry.

Nothing here knows about a specific model. A trial deep-copies one registry entry, overwrites its
``init_args`` / ``datamodule_init_args`` / ``trainer_args``, and hands the result to the unmodified
LightningConfigFactory - so every model the registry can already build is tunable, including ones
added later.

Entry point: ``tune.py`` at the repository root.
"""

from yg_eo_soilnet.hpo.constraints import CONSTRAINTS, constraint
from yg_eo_soilnet.hpo.data import build_lightning_input
from yg_eo_soilnet.hpo.export import build_tuned_spec, export_best_config
from yg_eo_soilnet.hpo.objective import ObjectiveContext, TrialObjective
from yg_eo_soilnet.hpo.overrides import apply_overrides, validate_override_keys
from yg_eo_soilnet.hpo.plots import optimization_history, param_importances, write_study_artifacts
from yg_eo_soilnet.hpo.progress import MODES, StudyProgress, resolve_mode, tqdm_safe_logging
from yg_eo_soilnet.hpo.pruning import OptunaPruningCallback, metric_to_float
from yg_eo_soilnet.hpo.search_space import Distribution, Objective, SearchSpace
from yg_eo_soilnet.hpo.study import (
    create_or_load_study,
    default_study_name,
    reset_study,
    run_study,
    summarize,
)
from yg_eo_soilnet.hpo.tracker import ObjectiveTracker, TrialRecord, best_value_or_none
from yg_eo_soilnet.hpo.trial_runner import (
    TrialRunner,
    UnrecoverableAcceleratorError,
    cuda_context_is_dead,
    raise_if_accelerator_is_dead,
    silence_lightning,
)

__all__ = [
    "CONSTRAINTS",
    "MODES",
    "Distribution",
    "Objective",
    "ObjectiveContext",
    "ObjectiveTracker",
    "OptunaPruningCallback",
    "SearchSpace",
    "StudyProgress",
    "TrialObjective",
    "TrialRecord",
    "TrialRunner",
    "UnrecoverableAcceleratorError",
    "apply_overrides",
    "best_value_or_none",
    "build_lightning_input",
    "build_tuned_spec",
    "constraint",
    "cuda_context_is_dead",
    "create_or_load_study",
    "default_study_name",
    "export_best_config",
    "metric_to_float",
    "optimization_history",
    "param_importances",
    "raise_if_accelerator_is_dead",
    "reset_study",
    "resolve_mode",
    "run_study",
    "silence_lightning",
    "summarize",
    "tqdm_safe_logging",
    "validate_override_keys",
    "write_study_artifacts",
]
