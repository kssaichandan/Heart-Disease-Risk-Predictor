import os

import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.utils import shuffle

from src.data_cleaning import OUTLIER_CAP_COLUMNS, cap_outliers_iqr, get_cleaned_dataset_path
from src.data_download import get_project_root
from src.user_data import load_user_training_data

RANDOM_STATE = 42
TARGET_SIZE = 1100


def get_augmented_dataset_path():
    return os.path.join(
        get_project_root(),
        "data",
        "processed",
        "heart_augmented.csv",
    )


def _balance_features_target(features, target, target_size=TARGET_SIZE):
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    smote_features, smote_target = smote.fit_resample(features, target)
    smote_features = pd.DataFrame(smote_features, columns=features.columns)
    smote_target = pd.Series(smote_target, name="target")

    if len(smote_target) < target_size:
        majority_target = target_size // 2
        minority_target = target_size - majority_target
        sampler = RandomOverSampler(
            sampling_strategy={0: majority_target, 1: minority_target},
            random_state=RANDOM_STATE,
        )
        final_features, final_target = sampler.fit_resample(smote_features, smote_target)
    else:
        final_features, final_target = smote_features, smote_target

    balanced = pd.concat(
        [
            pd.DataFrame(final_features, columns=features.columns).reset_index(drop=True),
            pd.Series(final_target, name="target").reset_index(drop=True),
        ],
        axis=1,
    )
    balanced = shuffle(balanced, random_state=RANDOM_STATE).reset_index(drop=True)

    if len(balanced) > target_size:
        balanced = balanced.groupby("target", group_keys=False).apply(
            lambda frame: frame.sample(
                n=(target_size // 2) if frame.name == 0 else (target_size - (target_size // 2)),
                random_state=RANDOM_STATE,
            )
        )
        balanced = shuffle(balanced, random_state=RANDOM_STATE).reset_index(drop=True)

    return balanced


def augment_training_split(features_df, target_series, target_size=TARGET_SIZE, persist=True):
    """SMOTE + RandomOverSampler balance applied ONLY to a training split.

    Re-applies the IQR outlier cap to numeric columns first so that any user-added
    rows merged upstream are bounded the same way as cleaned data.
    """
    try:
        features = features_df.reset_index(drop=True).copy()
        target = target_series.reset_index(drop=True).copy()

        for column in OUTLIER_CAP_COLUMNS:
            if column in features.columns:
                cap_outliers_iqr(features, column)

        balanced = _balance_features_target(features, target, target_size=target_size)

        if persist:
            output_path = get_augmented_dataset_path()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            balanced.to_csv(output_path, index=False)
            print(f"Augmented training set saved to {output_path} with {len(balanced)} rows")

        augmented_features = balanced.drop(columns=["target"])
        augmented_target = balanced["target"]
        return augmented_features, augmented_target
    except Exception as exc:
        raise RuntimeError(f"Unable to augment training split: {exc}") from exc


def augment_data():
    input_path = get_cleaned_dataset_path()
    output_path = get_augmented_dataset_path()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        dataframe = pd.read_csv(input_path)
        user_dataframe = load_user_training_data()
        if not user_dataframe.empty:
            dataframe = pd.concat([dataframe, user_dataframe], ignore_index=True)
            dataframe = dataframe.drop_duplicates().reset_index(drop=True)

        features = dataframe.drop(columns=["target"])
        target = dataframe["target"]

        for column in OUTLIER_CAP_COLUMNS:
            if column in features.columns:
                cap_outliers_iqr(features, column)

        balanced = _balance_features_target(features, target)
        balanced.to_csv(output_path, index=False)
        print(f"Augmented dataset saved to {output_path} with {len(balanced)} rows")
        return output_path
    except Exception as exc:
        raise RuntimeError(f"Unable to augment dataset: {exc}") from exc


if __name__ == "__main__":
    augment_data()
