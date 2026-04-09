import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":
    results_path = Path("results")

    all_df = []

    for result in tqdm(list(results_path.rglob("*joblib"))):
        file_name = result.name
        approach = result.parent.name.split("-")
        if approach[1] == "calibration":
            calibration = True
            nfolds = int(approach[2])
            seed = int(approach[3])
        else:
            calibration = False
            nfolds = int(approach[1])
            seed = int(approach[2])
        approach = approach[0]

        dataset, model, seed = file_name.replace(".joblib", "").split("_")

        results = joblib.load(result)

        preds = results.get("preds", None)
        y_test = results.get("y_test", None)
        proba_preds = results.get("proba_preds", None)[..., 1]
        virtual_preds = results.get("virtual_preds", None)

        if len(proba_preds.shape) > 1:
            proba_var = np.var(proba_preds, axis=1)
            proba_min = np.min(proba_preds, axis=1)
            proba_max = np.max(proba_preds, axis=1)
            proba_preds = np.mean(proba_preds, axis=1)
        else:
            proba_var = np.ones_like(proba_preds) * np.nan
            proba_min = np.ones_like(proba_preds) * np.nan
            proba_max = np.ones_like(proba_preds) * np.nan

        if virtual_preds is not None:
            virtual_preds = np.squeeze(virtual_preds, 2)
            virtual_preds = 1 / (1 + np.exp(-virtual_preds))
            virtual_var = np.var(virtual_preds, axis=1)
            virtual_min = np.min(virtual_preds, axis=1)
            virtual_max = np.max(virtual_preds, axis=1)
            virtual_preds = np.mean(virtual_preds, axis=1)
        else:
            virtual_var = np.ones_like(proba_preds) * np.nan
            virtual_min = np.ones_like(proba_preds) * np.nan
            virtual_max = np.ones_like(proba_preds) * np.nan
            virtual_preds = np.ones_like(proba_preds) * np.nan

        data = pd.DataFrame(
            {
                "y_test": y_test,
                "proba_preds": proba_preds,
                "proba_var": proba_var,
                "proba_min": proba_min,
                "proba_max": proba_max,
                "virtual_var": virtual_var,
                "virtual_min": virtual_min,
                "virtual_max": virtual_max,
            }
        )
        data["approach"] = approach
        data["model"] = model
        data["dataset"] = dataset
        data["seed"] = seed
        data["calibration"] = calibration
        data["nfolds"] = nfolds

        all_df.append(data)

    all_df = pd.concat(all_df)
    Path("data").mkdir(exist_ok=True)
    all_df.to_csv("data/results.csv", index=False)
