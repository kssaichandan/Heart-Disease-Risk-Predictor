import os

import pandas as pd

from src.data_cleaning import get_cleaned_dataset_path
from src.data_download import COLUMN_NAMES, get_project_root

CATEGORICAL_COLUMNS = {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"}


def get_user_data_path():
    return os.path.join(
        get_project_root(),
        "data",
        "user",
        "user_training_data.csv",
    )


def load_user_training_data():
    path = get_user_data_path()
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMN_NAMES)

    dataframe = pd.read_csv(path)
    missing_columns = [column for column in COLUMN_NAMES if column not in dataframe.columns]
    for column in missing_columns:
        dataframe[column] = 0
    return dataframe[COLUMN_NAMES]


def append_user_training_row(row):
    path = get_user_data_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    dataframe = pd.DataFrame([[row[column] for column in COLUMN_NAMES]], columns=COLUMN_NAMES)
    write_header = not os.path.exists(path)
    dataframe.to_csv(path, mode="a", header=write_header, index=False)
    return path


def get_user_row_count():
    return len(load_user_training_data())


def preprocess_user_training_row(row):
    cleaned_path = get_cleaned_dataset_path()
    reference_frame = pd.read_csv(cleaned_path) if os.path.exists(cleaned_path) else pd.DataFrame(columns=COLUMN_NAMES)

    processed_row = {}
    imputed_fields = []

    for column in COLUMN_NAMES:
        value = row.get(column)
        if pd.isna(value):
            value = None

        if value is None:
            if column == "target":
                raise ValueError("Target must be 0 or 1.")

            if reference_frame.empty:
                fill_value = 0
            elif column in CATEGORICAL_COLUMNS:
                mode_series = reference_frame[column].mode(dropna=True)
                fill_value = mode_series.iloc[0] if not mode_series.empty else 0
            else:
                fill_value = reference_frame[column].median()

            processed_row[column] = int(round(fill_value)) if column in CATEGORICAL_COLUMNS else round(float(fill_value), 4)
            imputed_fields.append(column)
            continue

        numeric_value = float(value)
        if column in CATEGORICAL_COLUMNS:
            processed_row[column] = int(round(numeric_value))
        else:
            processed_row[column] = round(numeric_value, 4)

    if processed_row["target"] not in {0, 1}:
        raise ValueError("Target must be 0 or 1.")

    return processed_row, imputed_fields


def clear_user_training_data():
    path = get_user_data_path()
    if os.path.exists(path):
        os.remove(path)
    return path
