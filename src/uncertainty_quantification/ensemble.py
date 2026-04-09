from typing import List, Literal, Any, Dict
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold


class CVEnsembleEstimator(BaseEstimator):
    def __init__(
        self,
        base_estimator: BaseEstimator,
        task: Literal["classification", "regression"] = "classification",
        n_splits: int = 5,
        output_format: Literal["raw", "majority", "mean"] = None,
        random_state: int = 42,
    ):
        self.base_estimator = base_estimator
        self.task = task
        self.n_splits = n_splits
        self.output_format = output_format
        self.random_state = random_state
        self.estimators_: List[BaseEstimator] = []

    def fit(self, X, y, **fit_params):
        self.estimators_ = []
        if self.task == "classification":
            self.classes_ = np.unique(y)
            cv = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
        else:
            cv = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )

        for train_idx, val_idx in cv.split(X, y):
            X_train, y_train = (
                X[train_idx] if isinstance(X, np.ndarray) else X.iloc[train_idx]
            ), (
                y[train_idx] if isinstance(y, np.ndarray) else y.iloc[train_idx]
            )
            estimator = clone(self.base_estimator)
            estimator.fit(X_train, y_train, **fit_params)
            self.estimators_.append(estimator)
        return self

    def predict(self, X, *args, **kwargs):
        output_format = self.output_format
        if output_format is None:
            output_format = (
                "majority" if self.task == "classification" else "mean"
            )

        all_preds = []
        for estimator in self.estimators_:
            all_preds.append(estimator.predict(X, *args, **kwargs))

        all_preds = np.array(all_preds)  # shape: (n_splits, n_samples)

        if output_format == "raw":
            return all_preds.T  # shape: (n_samples, n_splits)
        elif output_format == "majority":
            if self.task == "regression":
                raise ValueError(
                    "Majority voting is not supported for regression tasks."
                )
            # Simple majority voting
            from scipy.stats import mode

            modes, _ = mode(all_preds, axis=0, keepdims=False)
            return modes
        elif output_format == "mean":
            return np.mean(all_preds, axis=0)
        else:
            raise ValueError(f"Unknown output_format: {output_format}")

    def virtual_ensembles_predict(self, X, virtual_ensembles_count=10):
        all_preds = []
        for estimator in self.estimators_:
            if hasattr(estimator, "virtual_ensembles_predict"):
                all_preds.append(
                    estimator.virtual_ensembles_predict(
                        X, virtual_ensembles_count=virtual_ensembles_count
                    )
                )
            else:
                all_preds.append(
                    estimator.predict(
                        X, virtual_ensembles_count=virtual_ensembles_count
                    )
                )

        if len(all_preds) > 0:
            return (
                np.concatenate(all_preds, axis=1)
                if all_preds[0].ndim > 1
                else np.column_stack(all_preds)
            )
        raise RuntimeError(
            "Estimator does not support virtual_ensembles_predict"
        )


class CVEnsembleClassifier(ClassifierMixin, CVEnsembleEstimator):
    def __init__(
        self,
        base_estimator,
        n_splits=5,
        output_format=None,
        proba_output_format="raw",
        random_state=42,
    ):
        super().__init__(
            base_estimator=base_estimator,
            task="classification",
            n_splits=n_splits,
            output_format=output_format,
            random_state=random_state,
        )
        self.proba_output_format = proba_output_format

    def predict_proba(self, X, *args, **kwargs):
        all_probas = []
        for estimator in self.estimators_:
            all_probas.append(estimator.predict_proba(X, *args, **kwargs))
        all_probas = np.array(all_probas)
        if self.proba_output_format == "raw":
            return np.transpose(all_probas, (1, 0, 2))
        return np.mean(all_probas, axis=0)

    def predict_proba_raw(self, X, *args, **kwargs):
        all_probas = []
        for estimator in self.estimators_:
            all_probas.append(estimator.predict_proba(X, *args, **kwargs))
        all_probas = np.array(all_probas)
        return np.transpose(all_probas, (1, 0, 2))


class CVEnsembleRegressor(RegressorMixin, CVEnsembleEstimator):
    def __init__(
        self, base_estimator, n_splits=5, output_format=None, random_state=42
    ):
        super().__init__(
            base_estimator=base_estimator,
            task="regression",
            n_splits=n_splits,
            output_format=output_format,
            random_state=random_state,
        )
