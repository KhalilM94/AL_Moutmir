from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.base import clone
import mlflow
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import traceback

from yg_eo_soilnet.logger import ChildRunLogger, TrainingLogger
from yg_eo_soilnet.datamodules.scikit.scikit_trainer_utils import (CVSplitter, PipelineBuilder, TargetNanFilter)
from yg_eo_soilnet.targets import split_target_names
from yg_eo_soilnet.tracking import start_child_run
from yg_eo_soilnet.uncertainty import (
    aggregate,
    bootstrap_indices,
    fit_calibrators,
    member_seeds,
    should_bootstrap,
    uncertainty_enabled_for,
)
from yg_eo_soilnet.uncertainty.intervals import (
    build_interval_estimators,
    needs_calibration_set,
    normalize_method,
)
from yg_eo_soilnet.uncertainty.predictors import EnsembleRegressor
from yg_eo_soilnet.utils import LogTransformer

# Tag distinguishing an ensemble member's run from a per-target evaluation run. Both are children of
# the same model run, and the parent leaderboard has to be able to tell them apart - see
# ParentRunLogger._collect_leaderboard, which would otherwise list five members in place of the model.
MEMBER_RUN_KIND = "ensemble_member"

class ModelTrainer:
    def __init__(
        self,
        config,
        columns_to_transform: Optional[List[str]] = None,
        enable_clustering: bool = False,
        split_strategy: str = 'kfold',
        seed: int = 42,
        n_splits: int = 5,
        logger= None,
        tuning_verbose: int = 0
    ):
        self.config = config
        self.columns_to_transform = columns_to_transform or []
        self.enable_clustering = enable_clustering
        self.split_strategy = split_strategy
        self.seed = seed
        self.n_splits = n_splits
        if logger is not None:
            self.logger = logger
        else:
            self.logger = TrainingLogger(
                enable_file_logging=getattr(config, 'SKLEARN_FILE_LOGGING_ENABLED', True),
            ).get_logger()
        self.tuning_verbose = tuning_verbose
        self.pipeline_builder = PipelineBuilder(
            tree_categorical_encoding=getattr(config, "TREE_CATEGORICAL_ENCODING", "ordinal"),
            tree_onehot_max_categories=getattr(config, "TREE_ONEHOT_MAX_CATEGORIES", None),
        )
        
        self.log_transformer = LogTransformer()

    def train(
        self,
        target: str,
        data: Dict,
        model_pipelines: Dict[str, Dict],
        targets: Optional[list] = None,
        ):
        """
        Train models for one TARGET GROUP using the provided data.

        `target` is the group's run label and `targets` the columns it covers. A group of one is
        the familiar single-output fit; a group of several fits one estimator against a 2-D y,
        which only estimators declaring `multi_target: native` in the registry can do - see
        yg_eo_soilnet.targets.
        """
        self._validate_model_pipelines(model_pipelines)
        target_names = [str(name) for name in (targets or split_target_names(target))] or [str(target)]
        self._guard_uniform_log_transform(target_names)

        # Which rows this family fits on, and which it keeps back to calibrate against. Decided ONCE
        # for the whole sklearn family rather than per model: two models in one run that fitted on
        # different row counts would sit on the same leaderboard axis with no sign that their
        # rmse_test is not a like-for-like comparison.
        fit_pool, calibration_data = self._resolve_fit_pool(data)
        X_train = fit_pool['X']
        X_train = X_train.astype({col: 'float64' for col in X_train.select_dtypes(include=['int64', 'int32']).columns})
        # A one-column selection stays a Series, so the single-target path is byte-for-byte what it
        # always was; several columns give the DataFrame a native multi-output estimator wants.
        y_train = self._select_targets(fit_pool['y'], target_names)
        X_test = data['X_test']
        # Must read X_test's own dtypes: X_train was already converted above, so keying off it
        # produced an empty mapping and left X_test integer-typed.
        X_test = X_test.astype({col: 'float64' for col in X_test.select_dtypes(include=['int64', 'int32']).columns})
        y_test = self._select_targets(data['y_test'], target_names)
        groups_train = data['groups_train'] if self.enable_clustering else None

        X_train, y_train, groups_train = TargetNanFilter().transform(X_train, y_train, groups_train)

        if X_test is not None and y_test is not None:
            X_test, y_test, _ = TargetNanFilter().transform(X_test, y_test)
        else:
            X_test, y_test = None, None

        # Same NaN filter as the fit pool and the test split: a calibration row whose target was
        # never measured contributes a residual of NaN, which fit_conformal would drop anyway - but
        # dropping it here keeps the row count it reports honest.
        if calibration_data is not None:
            X_calib = calibration_data['X'].astype(
                {col: 'float64' for col in calibration_data['X'].select_dtypes(include=['int64', 'int32']).columns}
            )
            y_calib = self._select_targets(calibration_data['y'], target_names)
            X_calib, y_calib, _ = TargetNanFilter().transform(X_calib, y_calib)
            calibration_data = {'X': X_calib, 'y': y_calib}

        min_feature_count = int(getattr(self.config, "MIN_FEATURE_COUNT", 10))
        min_valid_rows = max(5, min_feature_count, int(X_train.shape[1]))
        if y_train is not None and not y_train.empty and len(y_train) < min_valid_rows:
            self.logger.warning(
                f"Target {target} has only {len(y_train)} valid training rows after NaN filtering; "
                f"minimum recommended is {min_valid_rows} for {X_train.shape[1]} features."
            )
        if y_test is not None and hasattr(y_test, 'empty') and not y_test.empty and len(y_test) < min_valid_rows:
            self.logger.warning(
                f"Target {target} has only {len(y_test)} valid test rows after NaN filtering; "
                f"minimum recommended is {min_valid_rows} for {X_train.shape[1]} features."
            )

        # The whole featurized population, for the per-point prediction export. Its index still
        # keys into `point_ids`, so the export never has to pair rows by position. Deliberately NOT
        # run through TargetNanFilter: a point with an unmeasured target can still be predicted,
        # and dropping it here would put holes in an export whose whole purpose is completeness.
        export_data = None
        if data.get('X_all') is not None and data.get('point_ids') is not None:
            X_all = data['X_all']
            export_data = {
                'X': X_all.astype(
                    {col: 'float64' for col in X_all.select_dtypes(include=['int64', 'int32']).columns}
                ),
                'point_ids': data['point_ids'],
            }

        if not self._should_skip_target(y_train, y_test, target):

            # Uniform across the group; _guard_uniform_log_transform above refused a mixed one.
            is_log_target = target_names[0] in self.columns_to_transform
            mlflow_logger = ChildRunLogger()

            trained_models = 0
            for model_name, config in model_pipelines.items():
                try:
                    self.logger.info(f"Training {model_name} for {target}")

                    # The run is opened HERE, not inside the logger, so the grid search below runs
                    # inside it and its progress and system metrics attach to the right run. It also
                    # gives sklearn the same ownership rule as Lightning, where the trainer has
                    # always had to open the run before fit() so the loss curves had somewhere to go.
                    with start_child_run(f"{target}_{model_name}"):
                        self._train_one(
                            config=config,
                            model_name=model_name,
                            target=target,
                            target_names=target_names,
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            groups_train=groups_train,
                            is_log_target=is_log_target,
                            mlflow_logger=mlflow_logger,
                            calibration_data=calibration_data,
                            export_data=export_data,
                        )
                    trained_models += 1
                except Exception as e:
                    self.logger.warning(f"Training failed for {model_name} on {target}: {e}")
                    self.logger.debug(traceback.format_exc())
                    if getattr(self.config, "FAIL_ON_MODEL_ERROR", False):
                        raise

            # Without this, a target where every model failed still exits 0 with an empty MLflow run.
            if trained_models == 0 and getattr(self.config, "FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET", True):
                raise RuntimeError(
                    f"All {len(model_pipelines)} model(s) failed to train for target {target!r}; "
                    "see the warnings above. Set FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET: false to continue anyway."
                )

        else:
            self.logger.warning(f"Skipping training for target {target} due to insufficient data.")

    def _train_one(
        self,
        *,
        config,
        model_name,
        target,
        target_names,
        X_train,
        y_train,
        X_test,
        y_test,
        groups_train,
        is_log_target,
        mlflow_logger,
        calibration_data=None,
        export_data=None,
    ):
        """Fit one registry entry over one target group, inside an already-started run."""
        model_seed = int(config.get("random_seed", self.seed))

        cv_splitter = CVSplitter(
            cv_strategy=self.split_strategy,
            n_splits=self.n_splits,
            random_state=model_seed,
        )
        splits = cv_splitter.create_splits(X_train, y_train, groups_train)

        model = config["model"]
        params = config.get("params", {})
        modeltype = config.get("modeltype", "ml")
        if modeltype == "ml":
            is_tree_model = self.pipeline_builder._is_tree_based_model(model)
            categorical_encoding = "ordinal" if is_tree_model else "onehot"
            categorical_cols = [
                col for col in self.config.CATEGORICAL_FEATURES
                if col in X_train.columns
            ]
            numeric_cols = [
                col for col in X_train.columns
                if col not in categorical_cols
            ]
            # Build pipeline
            pipeline = self.pipeline_builder.build(
                model,
                is_log_target,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols,
            )

            # Adjust param grid if using TransformedTargetRegressor
            if is_log_target and bool(params):
                params = {
                    k.replace("model__", "model__regressor__") : v
                    for k, v in params.items()
                }
            search = GridSearchCV(
                estimator=clone(pipeline),
                param_grid= params if params is not None else {},
                cv=splits, refit=False,
                scoring= "neg_root_mean_squared_error",
                # -1 for ordinary estimators; entries whose model loads a large
                # checkpoint per worker set search_n_jobs to keep memory bounded.
                n_jobs=int(config.get("search_n_jobs", -1)),
                return_train_score=True,
                verbose=self.tuning_verbose
            )

            search.fit(X_train, y_train)
            best_params = search.best_params_ if params is not None else {}
            cv_results = pd.DataFrame(search.cv_results_)

            # The grid search stays a SINGLE pass whatever the ensemble size: the members share one
            # set of hyperparameters, so searching per member would multiply the most expensive part
            # of the run to answer a question already answered.
            if uncertainty_enabled_for(self.config, model_name):
                best_model = self._fit_ensemble(
                    pipeline=pipeline,
                    best_params=best_params,
                    model=model,
                    model_name=model_name,
                    target=target,
                    target_names=target_names,
                    X_train=X_train,
                    y_train=y_train,
                    groups_train=groups_train,
                    model_seed=model_seed,
                    splits=splits,
                    calibration_data=calibration_data,
                )
            else:
                best_model = clone(pipeline)
                if params is not None:
                    best_model.set_params(**best_params)
                best_model.fit(X_train, y_train)

            if not any(cv_results.get("params", [])):
                cv_results["params"] = [best_model.get_params()]
            #Evaluate model
            param_names = list(params.keys()) if params else []
            plot_func = {}
            if len(param_names) > 1:
                cv_plot = "yg_eo_soilnet.plot_utils.cv_parallel_coordinates"
            elif len(param_names) == 1:
                cv_plot = "yg_eo_soilnet.plot_utils.cv_val_curve"
            else:
                cv_plot = None  # No hyperparameters to plot
                param_names = list(best_model.get_params().keys())

            if cv_plot is not None:
                plot_func.update({cv_plot: {"args": [cv_results]}})
            mlflow_logger.log_child_run(
                config=self.config,
                search=search,
                cv_results=cv_results,
                best_model=best_model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                target=target,
                targets=target_names,
                param_names=param_names,
                model_name=model_name,
                plot_functions=plot_func,
                extra_params={
                    "categorical_encoding": categorical_encoding,
                },
                export_data=export_data,
                )
        else:
            raise ValueError(f"Unknown modeltype: {modeltype}")

    # --- uncertainty --------------------------------------------------------

    def _resolve_fit_pool(self, data: Dict) -> tuple[Dict, Optional[Dict]]:
        """``(fit_pool, calibration_data)`` - which rows are fitted on and which are held back.

        Normally the fit pool is ``X_train``, which is train u val: sklearn validates by k-fold
        INSIDE that pool, so it has never needed a separate validation holdout and the val rows
        would otherwise be wasted.

        Conformal calibration needs rows the model has not seen, so enabling it with
        ``calibration.source: val`` moves the family onto ``X_train_only`` and reserves ``X_val``.
        The alternative would be to calibrate on rows that were fitted on, which produces residuals
        that are too small, an interval that is too narrow, and a coverage guarantee that is void -
        and none of that would show up as an error. Paying ~15% of the training rows is the honest
        price; ``uncertainty_fit_pool`` is logged so the smaller rmse_test is never mistaken for a
        regression.
        """
        default_pool = {'X': data['X_train'], 'y': data['y_train']}

        if not bool(getattr(self.config, "UNCERTAINTY_ENABLED", False)):
            return default_pool, None
        # Gated on whether the chosen interval NEEDS held-out rows, not on whether one was asked
        # for at all. gaussian and sigma turn sigma into an interval by arithmetic, so reserving the
        # val split for them would cost ~15% of the training rows and buy nothing.
        if not needs_calibration_set(
            getattr(self.config, "UNCERTAINTY_INTERVAL_METHOD", "conformal")
        ):
            return default_pool, None
        if str(getattr(self.config, "UNCERTAINTY_CALIBRATION_SOURCE", "val")).lower() != "val":
            # cv_oof calibrates on out-of-fold residuals from the k-fold the grid search already
            # runs, so it keeps the full fit pool. See _calibration_from_out_of_fold.
            return default_pool, None

        train_only = data.get('X_train_only')
        val_features = data.get('X_val')
        if train_only is None or val_features is None or len(val_features) == 0:
            self.logger.warning(
                "uncertainty.calibration.source is 'val' but the split carries no separate val "
                "rows; falling back to the full fit pool and calibrating out-of-fold instead."
            )
            return default_pool, None

        self.logger.info(
            f"Uncertainty calibration reserves the val split: fitting on {len(train_only)} rows "
            f"and calibrating on {len(val_features)}."
        )
        return (
            {'X': train_only, 'y': data['y_train_only']},
            {'X': val_features, 'y': data['y_val']},
        )

    def _fit_ensemble(
        self,
        *,
        pipeline,
        best_params,
        model,
        model_name,
        target,
        target_names,
        X_train,
        y_train,
        groups_train,
        model_seed,
        splits,
        calibration_data,
    ) -> EnsembleRegressor:
        """Fit the members, calibrate them, and return them as one estimator.

        Each member gets its own MLflow child run, tagged ``run_kind=ensemble_member``. That tag is
        load-bearing: members are children of the model run, and the parent leaderboard collects the
        GRANDCHILDREN of the parent run in preference to their parent, so without a way to tell a
        member from a per-target evaluation run the leaderboard would list five members in place of
        the one model.
        """
        n_members = int(getattr(self.config, "UNCERTAINTY_N_MEMBERS", 5))
        stride = int(getattr(self.config, "UNCERTAINTY_SEED_STRIDE", 1000))
        seeds = member_seeds(model_seed, n_members, stride)
        bootstrap = should_bootstrap(
            model, str(getattr(self.config, "UNCERTAINTY_BOOTSTRAP", "auto"))
        )

        members = []
        for index, seed in enumerate(seeds):
            with start_child_run(
                f"{target}_{model_name}_member{index}",
                tags={
                    "run_kind": MEMBER_RUN_KIND,
                    "target": target,
                    "model_name": model_name,
                    "ensemble_member": str(index),
                    "ensemble_seed": str(seed),
                },
            ):
                member = clone(pipeline)
                if best_params:
                    member.set_params(**best_params)
                self._seed_member(member, seed)

                member_X, member_y = X_train, y_train
                if bootstrap:
                    positions = bootstrap_indices(len(X_train), seed)
                    member_X = X_train.iloc[positions]
                    member_y = y_train.iloc[positions]

                member.fit(member_X, member_y)
                mlflow.log_params(
                    {
                        "ensemble_member": index,
                        "ensemble_seed": seed,
                        "ensemble_bootstrapped": bootstrap,
                        "ensemble_n_rows": len(member_X),
                    }
                )
            members.append(member)

        # Which pool these members actually fitted on. Logged because it changes what rmse_test
        # means: a train-only ensemble trained on ~15% fewer rows than every non-uncertainty run in
        # the experiment, and comparing the two without knowing that reads as a regression.
        mlflow.log_params(
            {
                "uncertainty_fit_pool": "train_only" if calibration_data is not None else "train_val",
                "uncertainty_calibration_source": getattr(
                    self.config, "UNCERTAINTY_CALIBRATION_SOURCE", "val"
                ),
                "uncertainty_n_train_rows": len(X_train),
            }
        )

        ensemble = EnsembleRegressor(
            members=members,
            target_names=target_names,
            member_seeds=seeds,
            bootstrapped=bootstrap,
        )
        ensemble.calibrators = self._calibrate(
            ensemble=ensemble,
            pipeline=pipeline,
            best_params=best_params,
            target_names=target_names,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            splits=splits,
            calibration_data=calibration_data,
        )
        self._warn_if_ensemble_collapsed(ensemble, X_train, model_name, bootstrap)
        return ensemble

    def _calibrate(
        self,
        *,
        ensemble,
        pipeline,
        best_params,
        target_names,
        X_train,
        y_train,
        groups_train,
        splits,
        calibration_data,
    ) -> dict:
        """One interval estimator per target, of whichever kind the config asked for.

        Only the conformal branch touches data. gaussian and sigma turn sigma into an interval by
        arithmetic alone, so they are built here without a calibration set and without the
        out-of-fold pass below.
        """
        method = normalize_method(getattr(self.config, "UNCERTAINTY_INTERVAL_METHOD", "conformal"))
        alpha = float(getattr(self.config, "UNCERTAINTY_ALPHA", 0.05))

        if not needs_calibration_set(method):
            return build_interval_estimators(
                method,
                target_names,
                alpha=alpha,
                k=float(getattr(self.config, "UNCERTAINTY_INTERVAL_K", 1.0)),
            )

        if calibration_data is not None:
            prediction = ensemble.predict_uncertainty(calibration_data['X'])
            return fit_calibrators(
                prediction,
                calibration_data['y'],
                target_names,
                alpha=alpha,
                logger=self.logger,
            )

        return self._calibration_from_out_of_fold(
            ensemble=ensemble,
            pipeline=pipeline,
            best_params=best_params,
            target_names=target_names,
            X_train=X_train,
            y_train=y_train,
            splits=splits,
            alpha=alpha,
        )

    def _calibration_from_out_of_fold(
        self,
        *,
        ensemble,
        pipeline,
        best_params,
        target_names,
        X_train,
        y_train,
        splits,
        alpha,
    ) -> dict:
        """Calibrate on out-of-fold residuals, keeping every row in the fit pool.

        One documented approximation, and it is worth being explicit about because it is the reason
        `val` is the default. The RESIDUALS come from a single pipeline refitted per fold, so each
        is measured on a model that saw ~80% of the pool; the SIGMA comes from the full ensemble.
        The two do not describe the same model. The mismatch runs in the conservative direction -
        out-of-fold residuals are larger than the full model's, so the interval errs wide - but the
        exact split-conformal guarantee does not carry over. Read this as a well-behaved heuristic,
        not as the proven bound `val` gives.
        """
        estimator = clone(pipeline)
        if best_params:
            estimator.set_params(**best_params)

        out_of_fold = np.asarray(
            cross_val_predict(estimator, X_train, y_train, cv=splits, n_jobs=1)
        )
        if out_of_fold.ndim == 1:
            out_of_fold = out_of_fold.reshape(-1, 1)

        sigma = ensemble.predict_uncertainty(X_train).total_std
        prediction = aggregate([out_of_fold], [np.zeros_like(out_of_fold)])
        # aggregate over one member reports zero spread; the sigma the interval scales is the
        # ensemble's, measured on the same rows.
        prediction = type(prediction)(
            mean=prediction.mean,
            epistemic_std=sigma,
            aleatoric_std=np.zeros_like(sigma),
        )

        return fit_calibrators(
            prediction, y_train, target_names, alpha=alpha, logger=self.logger
        )

    @staticmethod
    def _seed_member(member, seed: int) -> None:
        """Push a member's seed into the estimator step of its pipeline.

        Two possible paths because the pipeline's `model` step is the estimator directly, or a
        TransformedTargetRegressor wrapping it when the target is logged - the same pair the
        param-grid rewrite in _train_one has to handle. Guarded on the key actually existing:
        an estimator with no random_state is seeded by the bootstrap instead.
        """
        available = member.get_params(deep=True)
        for key in ("model__random_state", "model__regressor__random_state"):
            if key in available:
                member.set_params(**{key: int(seed)})
                return

    def _warn_if_ensemble_collapsed(self, ensemble, X_train, model_name: str, bootstrap: bool) -> None:
        """Say so when the members are identical, instead of reporting perfect confidence.

        A zero standard deviation is indistinguishable from a supremely confident model in every
        artifact downstream: the bars vanish, picp reads 0 or 1, and nothing says the ensemble never
        formed. This is the check that turns that into a line in the log.
        """
        sample = X_train.iloc[: min(len(X_train), 256)]
        spread = float(np.max(ensemble.predict_uncertainty(sample).epistemic_std))
        if spread > 0.0:
            return
        self.logger.warning(
            f"Every ensemble member of {model_name} predicts identically (epistemic std is exactly "
            f"0), so the ensemble carries no model uncertainty"
            + (
                ". The members were bootstrapped, so this points at an estimator whose fit does not "
                "depend on the sampled rows."
                if bootstrap
                else " because bootstrapping is disabled and this estimator ignores its seed. Set "
                "uncertainty.bootstrap: auto."
            )
        )

    @staticmethod
    def _select_targets(frame: pd.DataFrame, target_names: list):
        """The group's columns, as a Series for one target and a DataFrame for several.

        The squeeze matters: every estimator in the registry takes a 1-D y, and handing a
        one-column DataFrame instead would change the single-target path that has always worked.
        """
        if len(target_names) == 1:
            return frame[target_names[0]]
        return frame[list(target_names)]

    def _guard_uniform_log_transform(self, target_names: list) -> None:
        """Refuse a joint group whose targets disagree about the log transform.

        ``TransformedTargetRegressor`` wraps the whole estimator, so the transform is a property of
        the FIT, not of a column. A group mixing logged and unlogged targets cannot be expressed;
        saying so beats silently applying one target's choice to the other.
        """
        if len(target_names) < 2:
            return
        logged = [name for name in target_names if name in self.columns_to_transform]
        if logged and len(logged) != len(target_names):
            raise ValueError(
                f"Targets {sorted(target_names)} are fitted jointly but disagree about the log "
                f"transform: {sorted(logged)} are in COLUMNS_TO_TRANSFORM and the rest are not. "
                "One fit applies one transform. Either align COLUMNS_TO_TRANSFORM or set "
                "MULTI_TARGET_MODE: per_target."
            )

    def _should_skip_target(self, y_train: pd.Series, y_test: Optional[pd.Series], target: str) -> bool:
        n_train = len(y_train) if y_train is not None else 0
        n_test = len(y_test) if y_test is not None else 0
        self.logger.info(f"Training for target: {target} — {n_train} train samples, {n_test} test samples.")
        if y_train is None or y_train.empty or (y_test is not None and hasattr(y_test, 'empty') and y_test.empty):
            self.logger.warning(f"Skipping {target} — no valid data after filtering NaNs.")
            return True
        return False

    def _validate_model_pipelines(self, model_pipelines: Dict[str, Dict]):
        for model_name, config in model_pipelines.items():
            if "model" not in config or "params" not in config:
                raise ValueError(f"Model pipeline '{model_name}' must have 'model' and 'params' keys.")

