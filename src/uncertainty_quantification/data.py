import os
import requests
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split

from .logger import get_logger

logger = get_logger("uncertainty_quantification")

RADMLBENCH_URL = "https://raw.githubusercontent.com/aydindemircioglu/radMLBench"
METADATA_URL = f"{RADMLBENCH_URL}/main/radMLBench/metadata.yaml"
DATASETS_BASE_URL = f"{RADMLBENCH_URL}/main/datasets/"
DOWNLOAD_DIR = "data/raw"
FOLDS_DIR = "data/splits"


class DataCatalog:
    def __init__(
        self,
        metadata_url: str = METADATA_URL,
        datasets_base_url: str = DATASETS_BASE_URL,
    ):
        self.metadata_url = metadata_url
        self.datasets_base_url = datasets_base_url
        self.metadata: List[Dict[str, Any]] = []

    def fetch_metadata(self):
        logger.info(f"Fetching metadata from {self.metadata_url}")
        metadata_path = f"{DOWNLOAD_DIR}/metadata.yaml"
        download_if_unavailable(metadata_path, self.metadata_url)
        with open(metadata_path) as o:
            self.metadata = yaml.safe_load(o)
        return self.metadata

    def list_datasets(self) -> List[str]:
        if not self.metadata:
            self.fetch_metadata()
        if isinstance(self.metadata, dict):
            return sorted(list(self.metadata.keys()))
        elif isinstance(self.metadata, list):
            datasets = []
            for item in self.metadata:
                if "dataset" in item:
                    datasets.append(item["dataset"])
                elif "name" in item:
                    datasets.append(item["name"])
            return sorted(datasets)
        return []


def download_if_unavailable(path: str, url: str):
    if not os.path.exists(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            logger.error(f"Failed to download {path}: {e}")
            raise


def load_dataset(
    dataset_name: str, task: str = "classification"
) -> pd.DataFrame:
    logger.info(f"Loading dataset {dataset_name} from radMLBench")
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    path = f"{DOWNLOAD_DIR}/{dataset_name}.gz"
    url = f"{DATASETS_BASE_URL}/{dataset_name}.gz"
    download_if_unavailable(path, url)
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise


def get_data_split(
    dataset_name: str,
    random_seed: int,
    test_prop: float = 0.2,
    task: str = "classification",
):
    splits_path = os.path.join(
        FOLDS_DIR,
        f"{dataset_name}.{random_seed}.joblib",
    )
    if os.path.exists(splits_path):
        logger.info("Found split in %s", splits_path)
        X_train, X_test, y_train, y_test, idx_train, idx_test = joblib.load(
            splits_path
        )
    else:
        try:
            df = load_dataset(dataset_name)
            id_cols = [
                c for c in df.columns if c.lower() in ("id", "unnamed: 0")
            ]
            if id_cols:
                df = df.drop(columns=id_cols)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

        except Exception as e:
            raise ValueError(f"Failed to load dataset '{dataset_name}': {e}.")

        stratify = y if task == "classification" else None
        X_train, X_test, y_train, y_test, idx_train, idx_test = (
            train_test_split(
                X,
                y,
                np.arange(0, X.shape[0]),
                test_size=test_prop,
                random_state=random_seed,
                stratify=stratify,
            )
        )
        joblib.dump(
            (X_train, X_test, y_train, y_test, idx_train, idx_test),
            splits_path,
        )
        logger.info("Saved split to %s", splits_path)
    return X_train, X_test, y_train, y_test, idx_train, idx_test
