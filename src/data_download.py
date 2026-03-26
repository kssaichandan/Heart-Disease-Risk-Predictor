import os
import urllib.request

import pandas as pd

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def get_raw_dataset_path():
    return os.path.join(get_project_root(), "data", "raw", "heart_cleveland.csv")


def download_data():
    output_path = get_raw_dataset_path()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print("Dataset already exists — skipping download")
        return output_path

    try:
        urllib.request.urlretrieve(DATASET_URL, output_path)
        dataframe = pd.read_csv(output_path, header=None, names=COLUMN_NAMES)
        dataframe.to_csv(output_path, index=False)
        print(f"Dataset downloaded to {output_path}")
        return output_path
    except Exception as exc:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"Unable to download dataset: {exc}") from exc


if __name__ == "__main__":
    download_data()
