import os

from uncertainty_quantification.data import (
    get_data_split,
    FOLDS_DIR,
    DataCatalog,
)
from uncertainty_quantification.logger import get_logger

logger = get_logger("data_splitting")

DATASETS = DataCatalog().list_datasets()

os.makedirs(FOLDS_DIR, exist_ok=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=["all", *DATASETS],
        help="List of dataset names (e.g. dataset1 dataset2)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--test-prop",
        type=float,
        default=0.2,
        help="Testing proportion",
    )

    args = parser.parse_args()

    if "all" in args.datasets:
        datasets = DATASETS
    else:
        datasets = args.datasets

    for dataset in datasets:
        get_data_split(dataset, args.random_seed, args.test_prop)
