import argparse
import numpy as np
import os
import joblib
import time

from uncertainty_quantification.pipeline import get_pipeline
from uncertainty_quantification.data import (
    get_data_split,
    DataCatalog,
    FOLDS_DIR,
)
from uncertainty_quantification.logger import get_logger

logger = get_logger("workflow")


def main():
    catalog = DataCatalog()
    datasets = catalog.list_datasets()

    parser = argparse.ArgumentParser(
        description="Uncertainty Quantification Workflow CLI"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=["all", *datasets],
        help="List of dataset names (e.g. dataset1 dataset2)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        choices=["catboost", "xgboost", "lightgbm", "rf", "lr"],
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
        default=50,
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
        default=None,
        help="Number of virtual ensembles (CatBoost only)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials to run",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of jobs for parallel processing",
    )
    parser.add_argument(
        "--test-prop",
        type=float,
        default=0.2,
        help="Testing proportion",
    )
    parser.add_argument(
        "--save-model",
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
    os.makedirs(FOLDS_DIR, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Results will be saved to {args.output_dir}/")

    # i) set random seed globally
    np.random.seed(args.random_seed)
    logger.info(f"Random seed set to {args.random_seed}")

    if "all" in args.datasets:
        selected_datasets = datasets
    else:
        selected_datasets = args.datasets

    for dataset_name in selected_datasets:
        logger.info(f"--- Processing dataset: {dataset_name} ---")
        # ii) Split train/test for ALL data consistently using the specified random seed
        logger.info("Splitting data into train and test sets consistently")
        X_train, X_test, y_train, y_test, idx_train, idx_test = get_data_split(
            dataset_name, args.random_seed, task=args.task
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
                n_trials=args.n_trials,
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

            if args.task == "classification":
                output["proba_preds"] = pipeline.predict_proba(X_test)
                if args.cv_ensemble:
                    output["proba_preds_raw"] = pipeline.predict_proba_raw(
                        X_test
                    )

            if vec is not None and method == "catboost":
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
            output["idx_test"] = idx_test
            output["idx_train"] = idx_train
            output["time_elapsed"] = end_time - start_time
            joblib.dump(output, model_path, compress=9)


if __name__ == "__main__":
    main()
