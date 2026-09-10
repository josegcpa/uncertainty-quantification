import numpy as np
import optuna
from typing import Literal, Optional
from scipy.stats import mode
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score

from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from .ensemble import CVEnsembleClassifier, CVEnsembleRegressor
from .logger import get_logger

logger = get_logger("uncertainty_quantification")

MIN_TREES = 100
MAX_TREES = 200


class CustomCalibratedClassifierCV(CalibratedClassifierCV):
    def predict_with_method(self, X, method_name: str, *args, **kwargs):
        is_proba = "proba" in method_name
        preds = []
        for cal_est in self.calibrated_classifiers_:
            base_est = getattr(
                cal_est, "estimator", getattr(cal_est, "base_estimator", None)
            )
            if hasattr(base_est, method_name):
                fn = getattr(base_est, method_name)
                preds.append(fn(X, *args, **kwargs))
        if len(preds) > 0:
            if is_proba:
                return sum(preds) / len(preds)
            else:
                modes, _ = mode(np.stack(preds), axis=0, keepdims=False)
                return modes
        raise RuntimeError(f"Estimator does not support {method_name}")

    def virtual_ensembles_predict(self, X, virtual_ensembles_count=10):
        return self.predict_with_method(
            X,
            "virtual_ensembles_predict",
            virtual_ensembles_count=virtual_ensembles_count,
        )

    def predict_proba_raw(self, X):
        return self.predict_with_method(X, "predict_proba_raw")


class CustomPipeline:
    def __init__(
        self,
        task: Literal["classification", "regression"],
        method: Literal["catboost", "xgboost", "lightgbm", "rf"],
        posterior_sampling: bool = False,
        cv_ensemble: bool = False,
        cv_ensemble_splits: int = 5,
        calibration: bool = False,
        feature_selection_method: Literal["kbest"] = "kbest",
        n_features: int = 10,
        optimize_hyperparameters: bool = False,
        virtual_ensembles_count: Optional[int] = None,
        ensemble_output_format: Optional[str] = None,
        n_trials: int = 50,
        n_jobs: int = 1,
        seed: int = 42,
    ):
        self.task = task
        self.method = method
        self.posterior_sampling = posterior_sampling
        self.cv_ensemble = cv_ensemble
        self.cv_ensemble_splits = cv_ensemble_splits
        self.calibration = calibration
        self.feature_selection_method = feature_selection_method
        self.n_features = n_features
        self.optimize_hyperparameters = optimize_hyperparameters
        self.virtual_ensembles_count = virtual_ensembles_count
        self.ensemble_output_format = ensemble_output_format
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.seed = seed

        self.pipeline = None

        if (
            self.virtual_ensembles_count is not None
            and self.method != "catboost"
        ):
            raise ValueError(
                "virtual_ensembles_count is only available for method='catboost'"
            )

        if (
            self.virtual_ensembles_count is not None
            and not self.posterior_sampling
        ):
            logger.warning(
                "virtual_ensembles_count requested but posterior_sampling is "
                "disabled; virtual ensembles are degenerate without posterior "
                "sampling, enabling posterior_sampling"
            )
            self.posterior_sampling = True

        self._build_pipeline()

    def _build_pipeline(self):
        logger.info(f"Building pipeline for {self.task} using {self.method}")

        if self.posterior_sampling and self.method != "catboost":
            raise ValueError(
                "posterior_sampling is only available for method='catboost'"
            )

        base_estimator = self._get_estimator()

        if self.calibration and self.task == "classification":
            logger.info("Applying isotonic calibration to each fold model")
            cv = StratifiedKFold(
                n_splits=3, shuffle=True, random_state=self.seed
            )
            base_estimator = CustomCalibratedClassifierCV(
                base_estimator, method="isotonic", cv=cv
            )

        if self.cv_ensemble:
            logger.info(
                f"Using CV ensemble with {self.cv_ensemble_splits} splits"
            )
            if self.task == "classification":
                base_estimator = CVEnsembleClassifier(
                    base_estimator,
                    n_splits=self.cv_ensemble_splits,
                    output_format=self.ensemble_output_format,
                    proba_output_format="mean",
                    random_state=self.seed,
                )
            else:
                base_estimator = CVEnsembleRegressor(
                    base_estimator,
                    n_splits=self.cv_ensemble_splits,
                    output_format=self.ensemble_output_format,
                    random_state=self.seed,
                )

        selector = self._get_selector()

        steps = [("feature_selection", selector), ("estimator", base_estimator)]

        self.pipeline = Pipeline(steps)

    def _get_selector(self, n_features: Optional[int] = None):
        if self.feature_selection_method == "kbest":
            k = n_features if n_features is not None else self.n_features
            score_func = (
                f_classif if self.task == "classification" else f_regression
            )
            return SelectKBest(score_func=score_func, k=k)
        raise ValueError(
            f"Unknown feature selection method: {self.feature_selection_method}"
        )

    def _get_estimator(self, **hyperparams):
        if self.method == "catboost":
            kwargs = dict(hyperparams)
            kwargs["random_seed"] = self.seed
            kwargs["thread_count"] = 1
            if self.posterior_sampling:
                kwargs["posterior_sampling"] = True
                kwargs.setdefault("iterations", 100)
                kwargs.setdefault("l2_leaf_reg", 1.0)

            if self.task == "classification":
                return CatBoostClassifier(verbose=False, **kwargs)
            return CatBoostRegressor(verbose=False, **kwargs)

        elif self.method == "xgboost":
            params = {
                "n_jobs": 1,
                "random_state": self.seed,
                **hyperparams,
            }
            if self.task == "classification":
                return XGBClassifier(eval_metric="logloss", **params)
            return XGBRegressor(**params)

        elif self.method == "lightgbm":
            params = {
                "verbose": -1,
                "n_jobs": 1,
                "random_state": self.seed,
                **hyperparams,
            }
            if self.task == "classification":
                return LGBMClassifier(**params)
            return LGBMRegressor(**params)

        elif self.method == "lr":
            params = {
                "l1_ratio": 0.99,
                "random_state": self.seed,
                "max_iter": 10000,
                "solver": "saga",
                **hyperparams,
            }
            if self.task == "classification":
                return LogisticRegression(**params)
            return LinearRegression(**params)

        elif self.method == "rf":
            params = {
                "n_estimators": 100,
                "n_jobs": 1,
                "random_state": self.seed,
                **hyperparams,
            }
            if self.task == "classification":
                return RandomForestClassifier(**params)
            return RandomForestRegressor(**params)

        raise ValueError(f"Unknown method {self.method}")

    def fit(self, X, y):
        logger.info("Fitting pipeline")

        if self.optimize_hyperparameters:
            self._optimize(X, y)
        else:
            self.pipeline.fit(X, y)
        return self

    def _optimize(self, X, y):
        logger.info("Running hyperparameter optimization using Optuna")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {}
            n_features = self.n_features

            if self.feature_selection_method == "kbest":
                n_features = trial.suggest_int(
                    "n_features", 10, self.n_features
                )

            if self.method == "rf":
                params["n_estimators"] = trial.suggest_int(
                    "n_estimators", MIN_TREES, MAX_TREES
                )
                params["max_depth"] = trial.suggest_int("max_depth", 5, 20)
            if self.method == "lr":
                params["C"] = trial.suggest_float("C", 1e-7, 10.0, log=True)
                params["l1_ratio"] = trial.suggest_float(
                    "l1_ratio", 1e-7, 1.0, log=True
                )
            elif self.method in ["lightgbm", "xgboost"]:
                params["n_estimators"] = trial.suggest_int(
                    "n_estimators", MIN_TREES, MAX_TREES
                )
                params["learning_rate"] = trial.suggest_float(
                    "learning_rate", 1e-3, 0.5, log=True
                )
            elif self.method == "catboost":
                params["iterations"] = trial.suggest_int(
                    "iterations", MIN_TREES, MAX_TREES
                )
                if self.posterior_sampling:
                    params["l2_leaf_reg"] = trial.suggest_float(
                        "l2_leaf_reg", 0.1, 1.0, log=True
                    )
                params["learning_rate"] = trial.suggest_float(
                    "learning_rate", 1e-3, 0.5, log=True
                )

            selector = self._get_selector(n_features=n_features)

            estimator = self._get_estimator(**params)

            pipe = Pipeline(
                [("feature_selection", selector), ("estimator", estimator)]
            )

            metric = "accuracy" if self.task == "classification" else "r2"

            if self.task == "classification":
                cv = StratifiedKFold(
                    n_splits=3, shuffle=True, random_state=self.seed
                )
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=self.seed)

            scores = cross_val_score(pipe, X, y, cv=cv, scoring=metric)
            return scores.mean()

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)

        best_params = study.best_params
        best_n_features = best_params.pop("n_features", self.n_features)

        # Replace base estimator with optimized hyperparams, but wrapped in
        # whatever decorators (cv_ensemble, calibration) were requested.
        base_estimator = self._get_estimator(**best_params)

        if self.calibration and self.task == "classification":
            cv = StratifiedKFold(
                n_splits=3, shuffle=True, random_state=self.seed
            )
            base_estimator = CustomCalibratedClassifierCV(
                base_estimator, method="isotonic", cv=cv
            )

        if self.cv_ensemble:
            if self.task == "classification":
                base_estimator = CVEnsembleClassifier(
                    base_estimator,
                    n_splits=self.cv_ensemble_splits,
                    output_format=self.ensemble_output_format,
                    proba_output_format="mean",
                    random_state=self.seed,
                )
            else:
                base_estimator = CVEnsembleRegressor(
                    base_estimator,
                    n_splits=self.cv_ensemble_splits,
                    output_format=self.ensemble_output_format,
                    random_state=self.seed,
                )

        selector = self._get_selector(n_features=best_n_features)

        self.pipeline = Pipeline(
            [("feature_selection", selector), ("estimator", base_estimator)]
        )
        self.pipeline.fit(X, y)

    def predict(self, X, *args, **kwargs):
        logger.info("Predicting")

        if self.cv_ensemble:
            X_reduced = self.pipeline.named_steps[
                "feature_selection"
            ].transform(X)
            return self.pipeline.named_steps["estimator"].predict(
                X_reduced, *args, **kwargs
            )

        return self.pipeline.predict(X)

    def predict_proba(self, X, *args, **kwargs):
        logger.info("Predicting probabilities")

        if self.cv_ensemble:
            X_reduced = self.pipeline.named_steps[
                "feature_selection"
            ].transform(X)
            return self.pipeline.named_steps["estimator"].predict_proba(
                X_reduced, *args, **kwargs
            )

        return self.pipeline.predict_proba(X)

    def predict_proba_raw(self, X, *args, **kwargs):
        logger.info("Predicting raw probabilities")

        if not self.cv_ensemble:
            raise ValueError(
                "predict_proba_raw requires cv_ensemble=True; "
                "use predict_proba for standard probability predictions"
            )

        X_reduced = self.pipeline.named_steps["feature_selection"].transform(X)
        return self.pipeline.named_steps["estimator"].predict_proba_raw(
            X_reduced, *args, **kwargs
        )

    def virtual_ensembles_predict(self, X):
        """Wrapper around feature processing and calling virtual_ensembles_count for CatBoost"""
        if self.method != "catboost":
            raise ValueError(
                "virtual_ensembles_predict is only available for CatBoost"
            )
        if self.virtual_ensembles_count is None:
            raise ValueError(
                "virtual_ensembles_count must be set during initialization"
            )

        logger.info(
            f"Predicting with {self.virtual_ensembles_count} virtual ensembles"
        )

        # apply feature selection
        X_reduced = self.pipeline.named_steps["feature_selection"].transform(X)
        estimator = self.pipeline.named_steps["estimator"]

        if hasattr(estimator, "virtual_ensembles_predict"):
            return estimator.virtual_ensembles_predict(
                X_reduced, virtual_ensembles_count=self.virtual_ensembles_count
            )
        raise RuntimeError(
            "Estimator does not support virtual_ensembles_predict"
        )


def get_pipeline(**kwargs) -> CustomPipeline:
    return CustomPipeline(**kwargs)
