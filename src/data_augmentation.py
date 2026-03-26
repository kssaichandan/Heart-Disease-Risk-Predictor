import os

import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.utils import shuffle

from src.data_cleaning import get_cleaned_dataset_path
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

        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        smote_features, smote_target = smote.fit_resample(features, target)
        smote_features = pd.DataFrame(smote_features, columns=features.columns)
        smote_target = pd.Series(smote_target, name="target")

        if len(smote_target) < TARGET_SIZE:
            majority_target = TARGET_SIZE // 2
            minority_target = TARGET_SIZE - majority_target
            sampler = RandomOverSampler(
                sampling_strategy={0: majority_target, 1: minority_target},
                random_state=RANDOM_STATE,
            )
            final_features, final_target = sampler.fit_resample(smote_features, smote_target)
        else:
            final_features, final_target = smote_features, smote_target

        augmented = pd.concat(
            [
                pd.DataFrame(final_features, columns=features.columns),
                pd.Series(final_target, name="target"),
            ],
            axis=1,
        )
        augmented = shuffle(augmented, random_state=RANDOM_STATE).reset_index(drop=True)

        if len(augmented) > TARGET_SIZE:
            augmented = augmented.groupby("target", group_keys=False).apply(
                lambda frame: frame.sample(
                    n=(TARGET_SIZE // 2) if frame.name == 0 else (TARGET_SIZE - (TARGET_SIZE // 2)),
                    random_state=RANDOM_STATE,
                )
            )
            augmented = shuffle(augmented, random_state=RANDOM_STATE).reset_index(drop=True)

        augmented.to_csv(output_path, index=False)
        print(f"Augmented dataset saved to {output_path} with {len(augmented)} rows")
        return output_path
    except Exception as exc:
        raise RuntimeError(f"Unable to augment dataset: {exc}") from exc


if __name__ == "__main__":
    augment_data()
