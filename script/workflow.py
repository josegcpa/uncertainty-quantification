import argparse
import numpy as np
import pandas as pd
import os
import joblib
import time

from sklearn.model_selection import train_test_split
from uncertainty_quantification.pipeline import get_pipeline
from uncertainty_quantification.data import load_dataset, DataCatalog
from uncertainty_quantification.logger import get_logger

logger = get_logger("workflow")

DATASETS = DataCatalog().list_datasets()
TEST_PROP = 0.2


def main():
    parser = argparse.ArgumentParser(
        description="Uncertainty Quantification Workflow CLI"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=["all", *DATASETS],
        help="List of dataset names (e.g. dataset1 dataset2)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        choices=["catboost", "xgboost", "lightgbm", "rf"],
        help="List of ML methods to train",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "regression"],
        help="Task type",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=20,
        help="Number of features to select",
    )
    parser.add_argument(
        "--feature-selection",
        type=str,
        default="kbest",
        help="Feature selection method",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable Optuna hyperparameter optimization",
    )
    parser.add_argument(
        "--cv-ensemble",
        action="store_true",
        help="Use cross-validation ensembling",
    )
    parser.add_argument(
        "--cv-ensemble-splits",
        type=int,
        default=5,
        help="Number of splits for cross-validation ensembling",
    )
    parser.add_argument(
        "--calibration", action="store_true", help="Apply isotonic calibration"
    )
    parser.add_argument(
        "--posterior-sampling",
        action="store_true",
        help="Use posterior sampling (CatBoost only)",
    )
    parser.add_argument(
        "--virtual-ensembles-count",
        type=int,
        default=10,
        help="Number of virtual ensembles (CatBoost only)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs for parallel processing",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save trained models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save models and data",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Results will be saved to {args.output_dir}/")

    # i) set random seed globally
    np.random.seed(args.random_seed)
    logger.info(f"Random seed set to {args.random_seed}")

    if "all" in args.datasets:
        datasets = DATASETS
    else:
        datasets = args.datasets

    for dataset_name in datasets:
        logger.info(f"--- Processing dataset: {dataset_name} ---")
        try:
            df = load_dataset(dataset_name)
            if "ID" in df.columns:
                df = df.drop(columns=["ID"])
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

        except Exception as e:
            logger.error(
                f"Failed to load dataset '{dataset_name}': {e}. Using synthetic data fallback."
            )
            if args.task == "classification":
                from sklearn.datasets import make_classification

                X, y = make_classification(
                    n_samples=500, n_features=20, random_state=args.random_seed
                )
            else:
                from sklearn.datasets import make_regression

                X, y = make_regression(
                    n_samples=500, n_features=20, random_state=args.random_seed
                )

            # Save synthetic data
            df = pd.DataFrame(X)
            df["target"] = y
            dataset_path = os.path.join(
                args.output_dir, f"{dataset_name}_synthetic_data.csv"
            )
            df.to_csv(dataset_path, index=False)

        # ii) Split train/test for ALL data consistently using the specified random seed
        logger.info("Splitting data into train and test sets consistently")
        X_train, X_test, y_train, y_test, idx_train, idx_test = (
            train_test_split(
                X,
                y,
                np.arange(0, X.shape[0]),
                test_size=TEST_PROP,
                random_state=args.random_seed,
                stratify=y,
            )
        )

        for method in args.methods:
            model_filename = (
                f"{dataset_name}_{method}_seed{args.random_seed}.joblib"
            )
            model_path = os.path.join(args.output_dir, model_filename)
            if os.path.exists(model_path):
                logger.warning(
                    "Model %s already exists, skipping...", model_path
                )
                continue
            logger.info(f"--- Fitting model: {method} ---")
            # Constrain unsupported argument combinations to avoid crashes
            if args.posterior_sampling and method != "catboost":
                logger.warning(
                    f"Skipping {method} because posterior_sampling requires catboost"
                )
                continue

            vec = args.virtual_ensembles_count
            if vec is not None and method != "catboost":
                logger.warning(
                    f"Ignoring virtual_ensembles_count for {method} (requires catboost)"
                )
                vec = None

            start_time = time.time()
            pipeline = get_pipeline(
                task=args.task,
                method=method,
                posterior_sampling=args.posterior_sampling,
                cv_ensemble=args.cv_ensemble,
                cv_ensemble_splits=args.cv_ensemble_splits,
                calibration=args.calibration,
                feature_selection_method=args.feature_selection,
                n_features=args.n_features,
                optimize_hyperparameters=args.optimize,
                virtual_ensembles_count=vec,
                n_jobs=args.n_jobs,
                seed=args.random_seed,
            )

            pipeline.fit(X_train, y_train)
            end_time = time.time()

            output = {}

            if args.save_model:
                output["pipeline"] = pipeline

            logger.info("Calculating raw predictions")
            preds = pipeline.predict(X_test)
            proba_preds = pipeline.predict_proba_raw(X_test)

            if vec and method == "catboost":
                virtual_preds = pipeline.virtual_ensembles_predict(X_test)
                logger.info(
                    f"Virtual ensemble predictions shape: {virtual_preds.shape}"
                )

                output["virtual_preds"] = virtual_preds

            logger.info(
                f"Predictions shape for {method} on {dataset_name}: {preds.shape}"
            )
            output["preds"] = preds
            output["y_test"] = y_test
            output["proba_preds"] = proba_preds
            output["idx_test"] = idx_test
            output["idx_train"] = idx_train
            output["time_elapsed"] = end_time - start_time
            joblib.dump(output, model_path, compress=9)


if __name__ == "__main__":
    main()
