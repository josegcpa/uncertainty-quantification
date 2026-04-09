import pytest
import numpy as np
from uncertainty_quantification.pipeline import get_pipeline
from sklearn.datasets import make_classification, make_regression


def test_pipeline_basic_classification():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    pipeline = get_pipeline(
        task="classification",
        method="rf",
        feature_selection_method="kbest",
        n_features=5,
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == 100


def test_pipeline_basic_regression():
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    pipeline = get_pipeline(
        task="regression",
        method="lightgbm",
        feature_selection_method="kbest",
        n_features=5,
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == 100


def test_posterior_sampling_exception():
    with pytest.raises(ValueError):
        get_pipeline(
            task="classification", method="rf", posterior_sampling=True
        )


def test_virtual_ensembles_exception():
    with pytest.raises(ValueError):
        get_pipeline(
            task="classification", method="rf", virtual_ensembles_count=5
        )


def test_cv_ensemble_classification():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    pipeline = get_pipeline(
        task="classification",
        method="catboost",
        cv_ensemble=True,
        ensemble_output_format="majority",
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == 100


def test_calibration():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    pipeline = get_pipeline(
        task="classification", method="xgboost", calibration=True
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == 100


def test_virtual_ensembles_predict():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    pipeline = get_pipeline(
        task="classification", method="catboost", virtual_ensembles_count=5
    )
    pipeline.fit(X, y)
    preds = pipeline.virtual_ensembles_predict(X)
    assert preds.shape[1] == 5
    assert preds.shape[0] == 100


def test_virtual_ensembles_predict_with_calibration():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    pipeline = get_pipeline(
        task="classification",
        method="catboost",
        virtual_ensembles_count=5,
        calibration=True,
    )
    pipeline.fit(X, y)
    preds = pipeline.virtual_ensembles_predict(X)
    assert (
        preds.shape[1] == 5 or preds.shape[1] == 5 * 3
    )  # 3 cv folds in isotonic calibration by default
    assert preds.shape[0] == 100


def test_virtual_ensembles_predict_with_cv_ensemble():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    pipeline = get_pipeline(
        task="classification",
        method="catboost",
        virtual_ensembles_count=5,
        cv_ensemble=True,
    )
    pipeline.fit(X, y)
    preds = pipeline.virtual_ensembles_predict(X)
    assert preds.shape[1] == 5 or preds.shape[1] == 5 * 5  # 5 cv splits
    assert preds.shape[0] == 100


def test_virtual_ensembles_predict_with_cv_ensemble_and_calibration():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    pipeline = get_pipeline(
        task="classification",
        method="catboost",
        virtual_ensembles_count=5,
        cv_ensemble=True,
        calibration=True,
    )
    pipeline.fit(X, y)
    preds = pipeline.virtual_ensembles_predict(X)
    assert preds.shape[0] == 100
