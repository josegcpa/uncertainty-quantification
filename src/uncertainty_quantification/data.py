import os
import requests
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from .logger import get_logger

logger = get_logger("uncertainty_quantification")

METADATA_URL = "https://raw.githubusercontent.com/aydindemircioglu/radMLBench/main/radMLBench/metadata.yaml"
DATASETS_BASE_URL = "https://raw.githubusercontent.com/aydindemircioglu/radMLBench/main/datasets/"
DOWNLOAD_DIR = "data/raw"


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
        # The metadata is typically a list of dicts, but might be a dict. Handle safely.
        if isinstance(self.metadata, dict):
            return list(self.metadata.keys())
        elif isinstance(self.metadata, list):
            datasets = []
            for item in self.metadata:
                if "dataset" in item:
                    datasets.append(item["dataset"])
                elif "name" in item:
                    datasets.append(item["name"])
            return datasets
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
    url = os.path.join(DOWNLOAD_DIR, f"{dataset_name}.gz")
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise
