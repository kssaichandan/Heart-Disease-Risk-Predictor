import json
import os
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.ann_model import train_ann_model
from src.data_augmentation import augment_data, get_augmented_dataset_path
from src.data_cleaning import clean_data, get_cleaned_dataset_path
from src.data_download import COLUMN_NAMES, download_data, get_project_root
from src.evaluate import evaluate_models
from src.fuzzy_logic import train_fuzzy_model
from src.genetic_algorithm import run_genetic_algorithm
from src.random_forest import train_random_forest
from src.stacking_model import evaluate_meta_learner, train_meta_learner
from src.svm_model import train_svm_model
from src.user_data import load_user_training_data

RANDOM_STATE = 42
FULL_TRAIN_MODE = "full"
FAST_TRAIN_MODE = "fast"


def _get_models_dir():
    path = os.path.join(get_project_root(), "models")
    os.makedirs(path, exist_ok=True)
    return path


def _get_scaler_path():
    return os.path.join(_get_models_dir(), "scaler.pkl")


def _get_metadata_path():
    return os.path.join(_get_models_dir(), "model_metadata.json")


def _save_metadata(all_features, selected_indices, healthy_profile, training_summary, training_mode):
    metadata = {
        "all_features": all_features,
        "selected_indices": selected_indices,
        "selected_feature_names": [all_features[index] for index in selected_indices],
        "healthy_average": {key: round(float(value), 4) for key, value in healthy_profile.items()},
        "training_summary": training_summary,
        "training_mode": training_mode,
    }
    with open(_get_metadata_path(), "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def _load_existing_metadata():
    if not os.path.exists(_get_metadata_path()):
        return {}
    with open(_get_metadata_path(), "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def _extract_reusable_params(model, keys):
    model_params = model.get_params()
    return {key: model_params[key] for key in keys if key in model_params}


def _resolve_selected_indices(mode, scaled_features, target_values, feature_columns, existing_metadata):
    if mode == FAST_TRAIN_MODE:
        selected_indices = existing_metadata.get("selected_indices")
        if selected_indices:
            print("Fast retrain: reusing previously selected GA features.")
            return selected_indices
        print("Fast retrain: no saved GA features found, falling back to full GA selection.")
    return run_genetic_algorithm(scaled_features, target_values, feature_columns)


def _parse_mode(argv):
    if len(argv) <= 1:
        return FULL_TRAIN_MODE
    mode = argv[1].strip().lower()
    if mode in {FULL_TRAIN_MODE, FAST_TRAIN_MODE}:
        return mode
    raise ValueError("Training mode must be 'full' or 'fast'.")


def main(mode=FULL_TRAIN_MODE):
    try:
        download_data()
        clean_data()
        augment_data()

        existing_metadata = _load_existing_metadata()
        cleaned_dataframe = pd.read_csv(get_cleaned_dataset_path())
        augmented_dataframe = pd.read_csv(get_augmented_dataset_path())

        feature_columns = COLUMN_NAMES[:-1]
        features = augmented_dataframe[feature_columns]
        target = augmented_dataframe["target"]

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        joblib.dump(scaler, _get_scaler_path())

        selected_indices = _resolve_selected_indices(
            mode,
            scaled_features,
            target.to_numpy(),
            feature_columns,
            existing_metadata,
        )
        selected_feature_names = [feature_columns[index] for index in selected_indices]

        row_indices = augmented_dataframe.index.to_numpy()
        train_indices, test_indices = train_test_split(
            row_indices,
            test_size=0.2,
            stratify=target,
            random_state=RANDOM_STATE,
        )

        x_train = scaled_features[train_indices][:, selected_indices]
        x_test = scaled_features[test_indices][:, selected_indices]
        y_train = target.iloc[train_indices].to_numpy()
        y_test = target.iloc[test_indices].to_numpy()
        x_train_raw = augmented_dataframe.iloc[train_indices][["age", "chol", "thalach"]].reset_index(drop=True)
        x_test_raw = augmented_dataframe.iloc[test_indices][feature_columns].reset_index(drop=True)

        ann_epochs = 25 if mode == FAST_TRAIN_MODE else 100
        _, ann_predictions, ann_probabilities = train_ann_model(
            x_train,
            y_train,
            x_test,
            y_test,
            epochs=ann_epochs,
        )

        rf_fixed_params = None
        svm_fixed_params = None
        if mode == FAST_TRAIN_MODE:
            if os.path.exists(os.path.join(_get_models_dir(), "rf_model.pkl")):
                existing_rf_model = joblib.load(os.path.join(_get_models_dir(), "rf_model.pkl"))
                rf_fixed_params = _extract_reusable_params(
                    existing_rf_model,
                    ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"],
                )
            if os.path.exists(os.path.join(_get_models_dir(), "svm_model.pkl")):
                existing_svm_model = joblib.load(os.path.join(_get_models_dir(), "svm_model.pkl"))
                svm_fixed_params = _extract_reusable_params(existing_svm_model, ["C", "gamma"])

        rf_model, rf_predictions, rf_probabilities = train_random_forest(
            x_train,
            y_train,
            x_test,
            selected_feature_names,
            fast_mode=(mode == FAST_TRAIN_MODE),
            fixed_params=rf_fixed_params,
        )
        fuzzy_model = train_fuzzy_model()
        fuzzy_probabilities = fuzzy_model.predict_proba(x_test_raw)
        fuzzy_predictions = (fuzzy_probabilities >= 0.5).astype(int)
        svm_model, svm_predictions, svm_probabilities = train_svm_model(
            x_train,
            y_train,
            x_test,
            fast_mode=(mode == FAST_TRAIN_MODE),
            fixed_params=svm_fixed_params,
        )

        meta_model = train_meta_learner(
            x_train_selected=x_train,
            y_train=y_train,
            x_train_fuzzy=x_train_raw,
            rf_params=rf_model.get_params(),
            svm_params=svm_model.get_params(),
            cv_splits=3 if mode == FAST_TRAIN_MODE else 5,
            ann_epochs=25 if mode == FAST_TRAIN_MODE else 100,
        )
        stacked_predictions, stacked_probabilities = evaluate_meta_learner(
            meta_model,
            {
                "ANN": ann_probabilities,
                "Random Forest": rf_probabilities,
                "SVM": svm_probabilities,
                "Fuzzy Logic": fuzzy_probabilities,
            },
            y_test,
        )

        healthy_profile = (
            cleaned_dataframe.loc[cleaned_dataframe["target"] == 0, feature_columns]
            .mean()
            .to_dict()
        )
        user_training_dataframe = load_user_training_data()
        training_summary = {
            "raw_rows": len(pd.read_csv(os.path.join(get_project_root(), "data", "raw", "heart_cleveland.csv"))),
            "cleaned_rows": len(cleaned_dataframe),
            "user_rows_used": len(user_training_dataframe),
            "training_input_total": len(cleaned_dataframe) + len(user_training_dataframe),
            "augmented_rows": len(augmented_dataframe),
        }
        _save_metadata(feature_columns, selected_indices, healthy_profile, training_summary, mode)

        model_outputs = {
            "ANN": {"predictions": ann_predictions, "probabilities": ann_probabilities},
            "Random Forest": {"predictions": rf_predictions, "probabilities": rf_probabilities},
            "Fuzzy Logic": {"predictions": fuzzy_predictions, "probabilities": fuzzy_probabilities},
            "SVM": {"predictions": svm_predictions, "probabilities": svm_probabilities},
            "Stacked Ensemble": {"predictions": stacked_predictions, "probabilities": stacked_probabilities},
        }
        evaluate_models(y_test, model_outputs)

        print(
            f"{'Fast' if mode == FAST_TRAIN_MODE else 'Full'} training complete. "
            "Saved scaler, GA features, base models, meta model, metrics, and result charts."
        )
    except Exception as exc:
        print(f"Training failed: {exc}")
        raise


if __name__ == "__main__":
    main(_parse_mode(sys.argv))
