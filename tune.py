"""Optuna hyperparameter search over any entry in the Lightning registry.

    python tune.py --entry soil_tabular --n-trials 200

Nothing here is specific to a model. The entry name selects a registry entry and a matching search
space; the trial mutates that entry and hands it to the ordinary LightningConfigFactory, so a model
added to the registry tomorrow is tunable by writing a search space for it and nothing else.

The winner is exported as a ready-to-run registry file, named for the study that produced it:

    LIGHTNING_MODEL_REGISTRY_PATH=configs/lightning/tuned/soil_tabular-7438eb_best.yml python main.py

A study is named `<entry>-<fingerprint of its search space>` and resumed by name, so re-running the
same command continues the sweep while editing the entry's search space starts a clean one. `--reset`
discards a study, and `--study-name` pins one across edits (which is then checked, not assumed).
"""

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path

import mlflow

from config import Config
from yg_eo_soilnet.data_manager import DataManager
from yg_eo_soilnet.datamodules.scikit.scikit_datamodule import ScikitDataModule
from yg_eo_soilnet.hpo.data import build_lightning_input
from yg_eo_soilnet.hpo.export import export_best_config
from yg_eo_soilnet.hpo.objective import ObjectiveContext, TrialObjective
from yg_eo_soilnet.hpo.progress import MODES, StudyProgress
from yg_eo_soilnet.hpo.rerank import choose_winner, rerank, rerank_frame
from yg_eo_soilnet.hpo.search_space import SearchSpace
from yg_eo_soilnet.hpo.study import (
    DEFAULT_STORAGE,
    create_or_load_study,
    default_study_name,
    reset_study,
    run_study,
    set_hpo_experiment,
    summarize,
)
from yg_eo_soilnet.hpo.tracker import ObjectiveTracker, best_value_or_none
from yg_eo_soilnet.hpo.trial_runner import UnrecoverableAcceleratorError, silence_lightning
from yg_eo_soilnet.logger.training_logger import TrainingLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune a Lightning registry entry with Optuna")
    parser.add_argument("--entry", required=True, help="Registry entry to tune, e.g. soil_tabular")
    parser.add_argument("--config-path", default="configs/main_config.yml", help="Path to the main YAML config file")
    parser.add_argument(
        "--target",
        default=None,
        help=(
            "Which target group to tune, when MULTI_TARGET_MODE fits several models. Required in "
            "that case: a study optimizes one objective, and picking a group silently would tune "
            "one target and export the result as though it described the run."
        ),
    )
    parser.add_argument(
        "--search-spaces",
        default="configs/lightning/search_spaces",
        help=(
            "Where the search spaces live: either a folder of one-model-per-file YAMLs (the "
            "default) or a single YAML holding several. Pointing at <name>.yml also picks up a "
            "<name>/ folder beside it, so both layouts load the same spaces."
        ),
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Trials to run NOW; on a resumed study they are added to it"
    )
    parser.add_argument("--timeout", type=float, default=None, help="Seconds; stops after the running trial")
    parser.add_argument(
        "--study-name",
        default=None,
        help="Defaults to <entry>-<search space fingerprint>, so an unedited space resumes and an "
        "edited one starts clean. Naming it yourself pins it across edits, which is checked.",
    )
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--reset", action="store_true", help="Delete the study before running, discarding its trials")
    parser.add_argument("--seed", type=int, default=None, help="Defaults to config.RANDOM_SEED")
    parser.add_argument(
        "--seed-repeats", type=int, default=1, help="Average the objective over this many seeds per trial"
    )
    parser.add_argument("--max-epochs", type=int, default=None, help="Override trainer.max_epochs for every trial")
    parser.add_argument(
        "--cache-datamodules",
        action="store_true",
        help="Reuse a datamodule across trials that do not change its arguments",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Raise on a failing trial instead of pruning it")
    parser.add_argument(
        "--progress",
        choices=MODES,
        default="auto",
        help="Progress display: bars on a terminal, one line per trial when piped (default: auto)",
    )
    parser.add_argument("--top-n", type=int, default=10, help="How many trials to list at the end")
    parser.add_argument("--verbose", action="store_true", help="Keep Lightning's per-trial logging")
    parser.add_argument("--no-mlflow", action="store_true", help="Write only to the Optuna study database")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Re-export the best trial of an existing study and exit; runs no trials and loads no data",
    )
    parser.add_argument(
        "--rerank-top",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Re-run the K best trials of an existing study over several seeds and export the winner "
            "on the averaged score, instead of the single best trial. Runs no new trials. Budget "
            "K x --rerank-seeds full training runs"
        ),
    )
    parser.add_argument(
        "--rerank-seeds",
        type=int,
        default=3,
        metavar="N",
        help="Seeds per candidate when re-ranking (default 3). The first is the trial's own seed",
    )
    parser.add_argument("--export-path", default=None, help="Defaults to configs/lightning/tuned/<entry>_best.yml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.verbose:
        silence_lightning()

    config = Config(config_path=args.config_path)
    logger = TrainingLogger(
        name=f"AlMoutmir HPO - {args.entry}",
        log_filename=f"hpo_{args.entry}",
        enable_file_logging=config.MAIN_FILE_LOGGING_ENABLED,
    ).get_logger()

    space = SearchSpace.from_yaml(args.search_spaces, args.entry)
    if args.max_epochs is not None:
        space.fixed["trainer.max_epochs"] = args.max_epochs

    if args.entry not in config.LIGHTNING_MODEL_REGISTRY:
        available = ", ".join(sorted(config.LIGHTNING_MODEL_REGISTRY))
        raise SystemExit(f"No entry '{args.entry}' in {config.lightning_registry_path}. Available: {available}")
    registry_entry = config.LIGHTNING_MODEL_REGISTRY[args.entry]

    study_name = args.study_name or default_study_name(args.entry, space)
    if args.reset:
        reset_study(study_name, args.storage, logger)

    if args.export_only:
        # Re-exporting needs the stored study, the registry entry and the objective - no data at
        # all. Skipping data preparation and study.optimize turns a re-export of a long finished
        # study into a few seconds instead of another full run.
        study = create_or_load_study(space, study_name, args.storage)
        tracker = ObjectiveTracker(space.objective)
        tracker.prime(study)
        logger.info(f"Re-exporting study '{study_name}' ({len(study.trials)} trials) without running any")
        report_and_export(study, space, tracker, deepcopy(registry_entry), args, config, logger)
        return

    # Before any data work: the sklearn splitter logs its split artifacts to MLflow unconditionally,
    # which auto-starts a run in whatever experiment is current. Unset, that is Default - whose
    # artifact location may not even be writable. Naming the run keeps those artifacts findable
    # instead of scattering an auto-named run per invocation.
    set_hpo_experiment()

    # Load and split exactly as main.py does, then build the datamodule payload once. Every trial
    # reuses it; rebuilding per trial would cost far more than the training it feeds.
    stage_start = time.perf_counter()
    with mlflow.start_run(run_name=f"{study_name}_data"):
        scikit_datamodule = ScikitDataModule(config, logger, DataManager(config, logger))
        split_data = scikit_datamodule.prepare()
        # `split_data` carries the run's shared split_plan, and build_lightning_input copies the
        # dict, so every trial's datamodule resolves the SAME split. Rebuilding it per trial would
        # re-fit the scaler and the vocabulary against a different train set and quietly invalidate
        # the objective - which is why val_size/test_size/seed stay unsearchable in overrides.py.
        mlflow.log_params(split_data["split_plan"].describe())
        data = build_lightning_input(
            args.entry, registry_entry, config, split_data, logger=logger, data_manager=scikit_datamodule.data_manager
        )
    logger.info(f"Data prepared in {time.perf_counter() - stage_start:.2f}s")

    context = ObjectiveContext.from_config(
        args.entry,
        config,
        data=data,
        target=args.target,
        logger=logger,
        data_manager=scikit_datamodule.data_manager,
        datamodule_cache={} if args.cache_datamodules else None,
    )
    tracker = ObjectiveTracker(space.objective)
    progress = StudyProgress(
        n_trials=args.n_trials,
        objective=space.objective,
        tracker=tracker,
        mode=args.progress,
        logger=logger,
    )
    objective = TrialObjective(
        context,
        space,
        seed=args.seed if args.seed is not None else int(config.RANDOM_SEED),
        seed_repeats=args.seed_repeats,
        fail_fast=args.fail_fast,
        progress=progress,
    )

    if args.rerank_top is not None:
        rerank_and_export(objective, space, args, config, logger, study_name)
        return

    logger.info(f"Tuning '{args.entry}' to {space.objective.direction} {space.objective.metric}")
    try:
        study = run_study(
            objective,
            space,
            study_name=study_name,
            storage=args.storage,
            n_trials=args.n_trials,
            timeout=args.timeout,
            use_mlflow=not args.no_mlflow,
            extra_params={"entry": args.entry, "target": context.target, "seed_repeats": args.seed_repeats},
            logger=logger,
            tracker=tracker,
            progress=progress,
            artifact_dir=Path("optuna_studies") / study_name,
        )
    except BaseException as error:
        # BaseException on purpose: Ctrl-C on a long sweep used to lose the export exactly the same
        # way a lost GPU did.
        handle_study_abort(error, space, tracker, context, args, config, logger, study_name)
        raise

    report_and_export(study, space, tracker, context.registry_entry, args, config, logger)


def rerank_and_export(objective, space, args, config, logger, study_name: str) -> None:
    """Re-run the shortlist over several seeds and export the winner on the averaged score.

    The study's headline value is the best of hundreds of trials, each itself the best epoch of a
    noisy run - a maximum over noise, so it overstates what a retrain will give. This picks the
    configuration that holds up across seeds and records the figure to actually expect.
    """
    study = create_or_load_study(space, study_name, args.storage)
    logger.info(
        f"Re-ranking the top {args.rerank_top} of {len(study.trials)} trials in '{study_name}' "
        f"over {args.rerank_seeds} seed(s) each - {args.rerank_top * args.rerank_seeds} training runs"
    )

    results = rerank(objective, study, top_k=args.rerank_top, seeds=args.rerank_seeds, logger=logger)
    if not results:
        raise SystemExit("No shortlisted trial could be re-run, so there is nothing to export.")

    frame = rerank_frame(results, metric=space.objective.metric)
    logger.info(f"Re-ranked shortlist:\n{frame.to_string(index=False)}")

    not_reproduced = [r.trial_number for r in results if r.reproduced is False]
    if not_reproduced:
        # Seed 0 of each candidate re-runs the trial's own seed, so this is a direct check that
        # seeding reaches the weights. A miss means a run cannot be reproduced from its record.
        logger.warning(
            f"Trials {not_reproduced} did not reproduce at their own seed - runs are not "
            f"reproducible from their recorded seed, so treat the re-ranked means as noisy."
        )

    winner = choose_winner(results, space.objective.direction)
    logger.info(
        f"Winner: trial {winner.trial_number} | headline {winner.original_value:.6f} -> "
        f"re-ranked {winner.mean:.6f} +- {winner.std:.6f}"
    )
    if winner.trial_number != study.best_trial.number:
        logger.info(
            f"That is NOT the study's best trial ({study.best_trial.number}, "
            f"{study.best_value:.6f}) - the headline ranking did not survive re-running."
        )

    artifact_dir = Path("optuna_studies") / study_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(artifact_dir / "rerank.csv", index=False)

    export_path = Path(args.export_path or f"configs/lightning/tuned/{study_name}_reranked.yml")
    export_best_config(
        study,
        args.entry,
        deepcopy(config.LIGHTNING_MODEL_REGISTRY[args.entry]),
        space.objective,
        export_path,
        registry_path=config.lightning_registry_path,
        rerank=winner,
    )
    logger.info(f"Exported to {export_path}")
    print(f"\nRe-ranked winner: trial {winner.trial_number}")
    print(f"  headline   {space.objective.metric} = {winner.original_value:.6f}  (best of {len(study.trials)} trials)")
    print(f"  expect     {space.objective.metric} = {winner.mean:.6f} +- {winner.std:.6f}  on a retrain")
    print(f"Tuned config: {export_path}")
    print(f"  rerank table: {artifact_dir / 'rerank.csv'}")


def handle_study_abort(error, space, tracker, context, args, config, logger, study_name: str) -> None:
    """Salvage an aborted study, then leave the caller to re-raise.

    A sweep that ran for hours must not lose its winner because trial N+1 crashed - which is exactly
    what a lost GPU driver did to a 266-trial study whose 174 completed trials never reached
    `configs/lightning/tuned/`. Every trial is in the storage the whole time, so the study is simply
    reloaded and exported.

    Exporting never raises: it runs while another exception is in flight, and masking that one with
    a failure to export would hide why the study stopped. The one exception raised deliberately is
    SystemExit for a dead accelerator, whose traceback names whatever CUDA call came next rather than
    the failure and is therefore misleading noise.
    """
    try:
        study = create_or_load_study(space, study_name, args.storage)
        if best_value_or_none(study) is None:
            logger.warning("Study aborted with no completed trial, so there is nothing to export.")
        else:
            logger.warning(f"Study aborted - exporting the best of its {len(study.trials)} trials anyway.")
            report_and_export(study, space, tracker, context.registry_entry, args, config, logger)
    except Exception as export_error:
        logger.error(f"Could not export after the study aborted: {export_error!r}", exc_info=True)

    if isinstance(error, UnrecoverableAcceleratorError):
        raise SystemExit(str(error)) from error


def report_and_export(study, space, tracker, registry_entry, args, config, logger) -> None:
    """Log the trial leaderboard and write the tuned registry file."""
    summary = summarize(study, space)
    logger.info(f"Study finished: {summary}")
    if summary["best_trial"] is None:
        raise SystemExit("No trial completed, so there is nothing to export.")

    top_frame = tracker.top_frame(study, n=args.top_n)
    logger.info(f"Top {len(top_frame)} trials:\n{top_frame.to_string(index=False)}")
    best_params = "\n".join(f"  {key} = {value}" for key, value in sorted(tracker.best_params(study).items()))
    logger.info(f"Best trial {study.best_trial.number} parameters:\n{best_params}")

    # Keyed on the STUDY, not the entry. Keying on the entry meant any throwaway study on the same
    # model - a smoke run, an A/B, or the same model under an edited search space - silently
    # overwrote the production file a long study had written. Study names carry the search-space
    # fingerprint, so two spaces on one entry now export side by side.
    export_path = Path(args.export_path or f"configs/lightning/tuned/{study.study_name}_best.yml")
    export_best_config(
        study,
        args.entry,
        registry_entry,
        space.objective,
        export_path,
        registry_path=config.lightning_registry_path,
    )
    logger.info(f"Best {space.objective.metric}={study.best_value:.6f} (trial {study.best_trial.number})")
    logger.info(f"Exported to {export_path}")
    print(f"\nBest {space.objective.metric} = {study.best_value:.6f}  (trial {study.best_trial.number})")
    print(f"Tuned config: {export_path}")
    print(f"Train it with:\n  LIGHTNING_MODEL_REGISTRY_PATH={export_path} python main.py")


if __name__ == "__main__":
    main()
