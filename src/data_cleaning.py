import os

import numpy as np
import pandas as pd

from src.data_download import get_project_root, get_raw_dataset_path


def get_cleaned_dataset_path():
    return os.path.join(
        get_project_root(),
        "data",
        "processed",
        "heart_cleaned.csv",
    )


def _cap_outliers_iqr(dataframe, column_name):
    q1 = dataframe[column_name].quantile(0.25)
    q3 = dataframe[column_name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    dataframe[column_name] = dataframe[column_name].clip(lower=lower_bound, upper=upper_bound)


def clean_data():
    input_path = get_raw_dataset_path()
    output_path = get_cleaned_dataset_path()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        dataframe = pd.read_csv(input_path)
        dataframe = dataframe.replace("?", np.nan)

        for column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

        for column in ["ca", "thal"]:
            dataframe[column] = dataframe[column].fillna(dataframe[column].median())

        dataframe = dataframe.drop_duplicates().reset_index(drop=True)
        dataframe["target"] = (dataframe["target"] > 0).astype(int)

        for column in ["chol", "trestbps", "thalach", "oldpeak"]:
            _cap_outliers_iqr(dataframe, column)

        if dataframe.isna().sum().sum() > 0:
            dataframe = dataframe.fillna(dataframe.median(numeric_only=True))

        dataframe.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved to {output_path}")
        return output_path
    except Exception as exc:
        raise RuntimeError(f"Unable to clean dataset: {exc}") from exc


if __name__ == "__main__":
    clean_data()
