import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".cache", "matplotlib"),
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.ann_model import get_ann_model_path
from src.data_augmentation import get_augmented_dataset_path
from src.data_cleaning import get_cleaned_dataset_path
from src.data_download import COLUMN_NAMES, get_project_root, get_raw_dataset_path
from src.fuzzy_logic import get_fuzzy_model_path
from src.random_forest import get_rf_model_path
from src.stacking_model import get_meta_model_path
from src.svm_model import get_svm_model_path
from src.user_data import (
    append_user_training_row,
    clear_user_training_data,
    get_user_row_count,
    load_user_training_data,
    preprocess_user_training_row,
)

RISK_WEIGHTS = {
    "ANN": 0.35,
    "Random Forest": 0.30,
    "SVM": 0.25,
    "Fuzzy Logic": 0.10,
}
THAL_FORWARD_MAP = {1.0: 3.0, 2.0: 6.0, 3.0: 7.0}
THAL_REVERSE_MAP = {3.0: 1, 6.0: 2, 7.0: 3}
CP_FORWARD_MAP = {0.0: 1.0, 1.0: 2.0, 2.0: 3.0, 3.0: 4.0}
CP_REVERSE_MAP = {1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3}
SLOPE_FORWARD_MAP = {0.0: 1.0, 1.0: 2.0, 2.0: 3.0}
SLOPE_REVERSE_MAP = {1.0: 0, 2.0: 1, 3.0: 2}
DATASET_PAGE_SIZE = 20
CATEGORICAL_FEATURES = {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}

FEATURE_TOOLTIPS = {
    "age": "Patient age in years.",
    "sex": "Biological sex encoded as male or female.",
    "cp": "Chest pain type observed during evaluation.",
    "trestbps": "Resting blood pressure in mmHg.",
    "chol": "Serum cholesterol in mg/dl.",
    "fbs": "Fasting blood sugar greater than 120 mg/dl.",
    "restecg": "Resting electrocardiographic result.",
    "thalach": "Maximum heart rate achieved during testing.",
    "exang": "Exercise induced angina.",
    "oldpeak": "ST depression induced by exercise relative to rest.",
    "slope": "Slope of the peak exercise ST segment.",
    "ca": "Number of major vessels colored by fluoroscopy.",
    "thal": "Thalassemia test outcome category.",
}

ADVICE_BY_LEVEL = {
    "Low": "Your predicted risk is low. Maintain regular exercise, a balanced diet, and routine checkups.",
    "Moderate": "Your predicted risk is moderate. Consider a medical review soon and focus on blood pressure, cholesterol, and activity levels.",
    "High": "Your risk is high. Please consult a cardiologist immediately for a complete evaluation.",
}

app = Flask(__name__)
ARTIFACTS = {}
LOAD_ERROR = None
TRAINING_LOCK = threading.Lock()
TRAINING_THREAD = None
ARTIFACT_LOCK = threading.Lock()
DEFAULT_TRAINING_ESTIMATE_SECONDS = 300
FAST_TRAINING_ESTIMATE_SECONDS = 120

TRAINING_STATUS = {
    "state": "idle",
    "message": "Training has not started from the website yet.",
    "mode": None,
    "started_at": None,
    "finished_at": None,
    "estimated_total_seconds": DEFAULT_TRAINING_ESTIMATE_SECONDS,
    "duration_seconds": None,
    "logs": [],
}

NOISY_LOG_PATTERNS = [
    "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR",
    "port.cc:153",
    "oneDNN custom operations are on",
    "cpu_feature_guard.cc:227",
    "TensorFlow GPU support is not available on native Windows",
    "This TensorFlow binary is optimized to use available CPU instructions",
    "To enable the following instructions",
    "You are saving your model as an HDF5 file",
    "This file format is considered legacy",
    "FutureWarning:",
    "Passing `palette` without assigning `hue` is deprecated",
    "src\\evaluate.py",
    "tf.reset_default_graph is deprecated",
    "triggered tf.function retracing",
]


def _models_dir():
    return os.path.join(get_project_root(), "models")


def _metadata_path():
    return os.path.join(_models_dir(), "model_metadata.json")


def _scaler_path():
    return os.path.join(_models_dir(), "scaler.pkl")


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _serialize_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        rounded = round(float(value), 4)
        if rounded.is_integer():
            return int(rounded)
        return rounded
    return value


def _serialize_rows(dataframe):
    serialized_rows = []
    for _, row in dataframe.iterrows():
        serialized_rows.append({column: _serialize_value(row[column]) for column in dataframe.columns})
    return serialized_rows


def _load_dataset_by_name(dataset_name):
    dataset_map = {
        "raw": get_raw_dataset_path(),
        "cleaned": get_cleaned_dataset_path(),
        "augmented": get_augmented_dataset_path(),
        "user": None,
    }

    if dataset_name == "user":
        return load_user_training_data()

    dataset_path = dataset_map.get(dataset_name)
    if dataset_path is None or not os.path.exists(dataset_path):
        return pd.DataFrame(columns=COLUMN_NAMES)
    return pd.read_csv(dataset_path)


def _get_dataset_counts():
    training_summary = ARTIFACTS.get("training_summary", {})
    cleaned_rows = len(_load_dataset_by_name("cleaned"))
    user_rows = get_user_row_count()
    return {
        "raw": len(_load_dataset_by_name("raw")),
        "cleaned": cleaned_rows,
        "augmented": len(_load_dataset_by_name("augmented")),
        "user": user_rows,
        "training_input_total": cleaned_rows + user_rows,
        "trained_input_total": training_summary.get("training_input_total", cleaned_rows),
        "trained_user_rows": training_summary.get("user_rows_used", 0),
    }


def _build_dataset_payload(dataset_name, page, page_size):
    dataframe = _load_dataset_by_name(dataset_name)
    page_size = max(5, min(page_size, 100))
    total_rows = len(dataframe)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))

    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    page_frame = dataframe.iloc[start_index:end_index].reset_index(drop=True)

    return {
        "dataset": dataset_name,
        "columns": list(dataframe.columns),
        "rows": _serialize_rows(page_frame),
        "page": page,
        "page_size": page_size,
        "total_rows": total_rows,
        "total_pages": total_pages,
        "counts": _get_dataset_counts(),
    }


def _build_feature_importance(patient_scaled, healthy_scaled):
    feature_names = ARTIFACTS["all_features"]
    selected_names = ARTIFACTS["selected_feature_names"]
    rf_importances = dict(
        zip(selected_names, ARTIFACTS["rf_model"].feature_importances_.tolist())
    )
    contribution_map = {}

    for index, feature_name in enumerate(feature_names):
        base_weight = rf_importances.get(feature_name, 0.0)
        deviation = abs(float(patient_scaled[index]) - float(healthy_scaled[index]))
        contribution_map[feature_name] = base_weight * (0.5 + deviation)

    total = sum(contribution_map.values())
    if total <= 0:
        selected_weight = 1.0 / max(len(selected_names), 1)
        contribution_map = {
            feature_name: selected_weight if feature_name in selected_names else 0.0
            for feature_name in feature_names
        }
    else:
        contribution_map = {
            feature_name: round(value / total, 4)
            for feature_name, value in sorted(
                contribution_map.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }

    return contribution_map


def _risk_level(probability):
    if probability < 0.30:
        return "Low"
    if probability <= 0.60:
        return "Moderate"
    return "High"


def _safe_probability(value):
    return max(0.0, min(1.0, float(value)))


def _prepare_patient_payload(payload):
    patient_record = {}
    for feature_name in COLUMN_NAMES[:-1]:
        raw_value = payload.get(feature_name)
        if raw_value in (None, ""):
            patient_record[feature_name] = None
            continue

        numeric_value = float(raw_value)
        if np.isnan(numeric_value):
            patient_record[feature_name] = None
            continue

        patient_record[feature_name] = numeric_value

    if patient_record["cp"] in CP_FORWARD_MAP:
        patient_record["cp"] = CP_FORWARD_MAP[patient_record["cp"]]

    if patient_record["slope"] in SLOPE_FORWARD_MAP:
        patient_record["slope"] = SLOPE_FORWARD_MAP[patient_record["slope"]]

    if patient_record["thal"] in THAL_FORWARD_MAP:
        patient_record["thal"] = THAL_FORWARD_MAP[patient_record["thal"]]

    cleaned_dataframe = _load_dataset_by_name("cleaned")
    imputed_fields = []

    for feature_name in COLUMN_NAMES[:-1]:
        if patient_record[feature_name] is not None:
            continue

        if cleaned_dataframe.empty:
            fill_value = ARTIFACTS.get("healthy_average", {}).get(feature_name, 0)
        elif feature_name in CATEGORICAL_FEATURES:
            mode_series = cleaned_dataframe[feature_name].mode(dropna=True)
            fill_value = mode_series.iloc[0] if not mode_series.empty else 0
        else:
            fill_value = cleaned_dataframe[feature_name].median()

        patient_record[feature_name] = float(fill_value)
        imputed_fields.append(feature_name)

    return patient_record, imputed_fields


def _prepare_training_row(payload):
    patient_record = {}
    for feature_name in COLUMN_NAMES[:-1]:
        raw_value = payload.get(feature_name)
        if raw_value in (None, ""):
            patient_record[feature_name] = None
            continue

        numeric_value = float(raw_value)
        if np.isnan(numeric_value):
            patient_record[feature_name] = None
            continue

        patient_record[feature_name] = numeric_value

    if patient_record["cp"] in CP_FORWARD_MAP:
        patient_record["cp"] = CP_FORWARD_MAP[patient_record["cp"]]

    if patient_record["slope"] in SLOPE_FORWARD_MAP:
        patient_record["slope"] = SLOPE_FORWARD_MAP[patient_record["slope"]]

    if patient_record["thal"] in THAL_FORWARD_MAP:
        patient_record["thal"] = THAL_FORWARD_MAP[patient_record["thal"]]

    if "target" not in payload:
        raise ValueError("Missing field: target")

    target = int(float(payload["target"]))
    if target not in {0, 1}:
        raise ValueError("Target must be 0 or 1")

    patient_record["target"] = target
    return preprocess_user_training_row(patient_record)


def _prepare_form_sample(record):
    sample = {}
    for feature_name in COLUMN_NAMES[:-1]:
        value = float(record[feature_name])
        if feature_name == "cp":
            value = float(CP_REVERSE_MAP.get(value, value))
        elif feature_name == "slope":
            value = float(SLOPE_REVERSE_MAP.get(value, value))
        elif feature_name == "thal":
            value = float(THAL_REVERSE_MAP.get(value, value))

        if feature_name in CATEGORICAL_FEATURES:
            sample[feature_name] = int(round(value))
        elif feature_name == "oldpeak":
            sample[feature_name] = round(value, 1)
        else:
            sample[feature_name] = round(value, 2)

    return sample


def _sample_patient(kind):
    dataframe = _load_dataset_by_name("cleaned")
    if dataframe.empty:
        raise RuntimeError("Cleaned dataset is not available yet.")

    if kind == "healthy":
        subset = dataframe.loc[dataframe["target"] == 0]
    elif kind == "at_risk":
        subset = dataframe.loc[dataframe["target"] == 1]
    else:
        subset = dataframe

    if subset.empty:
        raise RuntimeError("No sample rows available for the selected mode.")

    sample_row = subset.sample(n=1).iloc[0]
    return _prepare_form_sample(sample_row)


def _set_training_status(**kwargs):
    with TRAINING_LOCK:
        TRAINING_STATUS.update(kwargs)


def _is_noisy_log(line):
    return any(pattern in line for pattern in NOISY_LOG_PATTERNS)


def _user_facing_training_message(line):
    if line.startswith("Dataset already exists"):
        return "Dataset already available. Starting preprocessing."
    if line.startswith("Dataset downloaded"):
        return "Dataset download completed."
    if line.startswith("Cleaned dataset saved"):
        return "Data cleaning completed."
    if line.startswith("Augmented dataset saved"):
        return "Data augmentation completed."
    if line.startswith("GA generation"):
        return "Feature selection is running."
    if line.startswith("Selected GA features:"):
        return "Feature selection completed."
    if line.startswith("Fast retrain: reusing previously selected GA features."):
        return "Fast retrain reused the saved GA-selected features."
    if line.startswith("Random Forest reused params"):
        return "Random Forest reused the previously tuned parameters."
    if line.startswith("Random Forest best params:"):
        return "Random Forest tuning completed."
    if line.startswith("SVM reused params"):
        return "SVM reused the previously tuned parameters."
    if line.startswith("SVM best params:"):
        return "SVM tuning completed."
    if line.startswith("Meta-learner trust split:"):
        return "Stacking meta-learner trained successfully."
    if line.startswith("Stacked ensemble accuracy:"):
        return "Stacked ensemble validation completed."
    if line.startswith("Training complete.") or line.startswith("Fast training complete.") or line.startswith("Full training complete."):
        return "Retraining completed successfully. New artifacts are active."
    if line.startswith("Training failed:"):
        return "Retraining failed. Check the latest log lines below."
    return None


def _display_training_log(line):
    if line.startswith("Dataset already exists"):
        return "Dataset already available. Skipping download."
    if line.startswith("Dataset downloaded"):
        return "Dataset download completed."
    if line.startswith("Cleaned dataset saved"):
        return "Cleaned dataset saved."
    if line.startswith("Augmented dataset saved"):
        return "Augmented dataset prepared."
    if line.startswith("Fast retrain: reusing previously selected GA features."):
        return "Fast retrain reused the saved GA-selected features."
    if line.startswith("GA generation"):
        generation_text = line.split(":", 1)[0].replace("GA generation", "").strip()
        if generation_text.startswith("01/") or generation_text.startswith("10/") or generation_text.startswith("20/") or generation_text.startswith("30/") or generation_text.startswith("40/") or generation_text.startswith("50/"):
            return f"Feature selection progress {generation_text}."
        return None
    if line.startswith("Selected GA features:"):
        return "Feature selection completed."
    if line.startswith("Random Forest reused params"):
        return "Random Forest reused the previously tuned parameters."
    if line.startswith("Random Forest best params:"):
        return "Random Forest tuning completed."
    if line.startswith("SVM reused params"):
        return "SVM reused the previously tuned parameters."
    if line.startswith("SVM best params:"):
        return "SVM tuning completed."
    if line.startswith("Meta-learner trust split:"):
        return line
    if line.startswith("Stacked ensemble accuracy:"):
        return line
    if line.startswith("Training complete.") or line.startswith("Fast training complete.") or line.startswith("Full training complete."):
        return "Retraining completed successfully. New artifacts are active."
    if line.startswith("Training failed:") or line.startswith("Traceback") or line.startswith("RuntimeError"):
        return line
    return None


def _run_training_job(mode):
    global TRAINING_THREAD

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = os.environ["MPLCONFIGDIR"]
    command = [sys.executable, "train.py", mode]
    process = subprocess.Popen(
        command,
        cwd=get_project_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    logs = []
    _set_training_status(
        state="running",
        mode=mode,
        message=(
            "Fast retraining started. This usually finishes sooner than full retraining."
            if mode == "fast"
            else "Full retraining started. This can take a few minutes."
        ),
        started_at=_utc_now_iso(),
        finished_at=None,
        duration_seconds=None,
        logs=[
            (
                "Fast retraining started. This usually finishes sooner than full retraining."
                if mode == "fast"
                else "Full retraining started. This can take a few minutes."
            )
        ],
    )

    try:
        if process.stdout is not None:
            for line in process.stdout:
                clean_line = line.rstrip()
                if clean_line:
                    user_message = _user_facing_training_message(clean_line)
                    display_line = None if _is_noisy_log(clean_line) else _display_training_log(clean_line)
                    if display_line:
                        if not logs or logs[-1] != display_line:
                            logs.append(display_line)
                            logs = logs[-80:]

                    status_update = {"logs": logs.copy()}
                    if user_message:
                        status_update["message"] = user_message
                    _set_training_status(**status_update)

        return_code = process.wait()
        if return_code == 0:
            ensure_artifacts_loaded(force_reload=True)
            started_at = datetime.fromisoformat(TRAINING_STATUS["started_at"])
            duration_seconds = max(
                1,
                int((datetime.now(timezone.utc) - started_at.astimezone(timezone.utc)).total_seconds()),
            )
            _set_training_status(
                state="completed",
                message="Retraining completed successfully. New artifacts are active.",
                mode=mode,
                finished_at=_utc_now_iso(),
                duration_seconds=duration_seconds,
                estimated_total_seconds=duration_seconds,
                logs=logs[-80:],
            )
        else:
            _set_training_status(
                state="failed",
                message="Retraining failed. Check the latest log lines below.",
                mode=mode,
                finished_at=_utc_now_iso(),
                logs=logs[-80:],
            )
    except Exception as exc:
        _set_training_status(
            state="failed",
            message=f"Retraining crashed: {exc}",
            mode=mode,
            finished_at=_utc_now_iso(),
            logs=logs[-80:] + [f"Runtime error: {exc}"],
        )
    finally:
        TRAINING_THREAD = None


def _start_retraining(status_message, mode="full"):
    global TRAINING_THREAD

    if TRAINING_THREAD is not None and TRAINING_THREAD.is_alive():
        raise RuntimeError("Retraining is already running.")

    estimated_total_seconds = (
        FAST_TRAINING_ESTIMATE_SECONDS if mode == "fast" else DEFAULT_TRAINING_ESTIMATE_SECONDS
    )

    _set_training_status(
        state="queued",
        message=status_message,
        mode=mode,
        started_at=_utc_now_iso(),
        finished_at=None,
        duration_seconds=None,
        estimated_total_seconds=estimated_total_seconds,
        logs=[status_message],
    )
    TRAINING_THREAD = threading.Thread(target=_run_training_job, args=(mode,), daemon=True)
    TRAINING_THREAD.start()

    with TRAINING_LOCK:
        return dict(TRAINING_STATUS)


def ensure_artifacts_loaded(force_reload=False):
    with ARTIFACT_LOCK:
        if force_reload or not ARTIFACTS:
            load_artifacts()
    return LOAD_ERROR is None


def load_artifacts():
    global ARTIFACTS, LOAD_ERROR
    try:
        from tensorflow.keras.models import load_model

        with open(_metadata_path(), "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

        ARTIFACTS = {
            "all_features": metadata["all_features"],
            "selected_indices": metadata["selected_indices"],
            "selected_feature_names": metadata["selected_feature_names"],
            "healthy_average": metadata["healthy_average"],
            "training_summary": metadata.get("training_summary", {}),
            "scaler": joblib.load(_scaler_path()),
            "ann_model": load_model(get_ann_model_path(), compile=False),
            "rf_model": joblib.load(get_rf_model_path()),
            "fuzzy_model": joblib.load(get_fuzzy_model_path()),
            "svm_model": joblib.load(get_svm_model_path()),
            "meta_model": joblib.load(get_meta_model_path()),
        }
        LOAD_ERROR = None
    except Exception as exc:
        ARTIFACTS = {}
        LOAD_ERROR = (
            "Model artifacts are not ready yet. Run `python train.py` before starting predictions. "
            f"Details: {exc}"
        )


@app.route("/")
def index():
    ensure_artifacts_loaded()
    return render_template(
        "index.html",
        model_ready=LOAD_ERROR is None,
        load_error=LOAD_ERROR,
        feature_tooltips=FEATURE_TOOLTIPS,
        dataset_counts=_get_dataset_counts(),
    )


@app.route("/predict", methods=["POST"])
def predict():
    ensure_artifacts_loaded()
    if LOAD_ERROR is not None:
        return jsonify({"error": LOAD_ERROR}), 503

    try:
        payload = request.get_json(silent=True) or {}
        patient_record, imputed_fields = _prepare_patient_payload(payload)

        patient_frame = pd.DataFrame([patient_record], columns=ARTIFACTS["all_features"])
        patient_scaled = ARTIFACTS["scaler"].transform(patient_frame)[0]
        selected_scaled = np.array([patient_scaled[ARTIFACTS["selected_indices"]]])

        ann_probability = _safe_probability(
            ARTIFACTS["ann_model"].predict(selected_scaled, verbose=0)[0][0]
        )
        rf_probability = _safe_probability(
            ARTIFACTS["rf_model"].predict_proba(selected_scaled)[0][1]
        )
        svm_probability = _safe_probability(
            ARTIFACTS["svm_model"].predict_proba(selected_scaled)[0][1]
        )
        fuzzy_probability = _safe_probability(
            ARTIFACTS["fuzzy_model"].predict_one(patient_record)
        )

        model_predictions = {
            "ANN": round(ann_probability, 4),
            "Random Forest": round(rf_probability, 4),
            "Fuzzy Logic": round(fuzzy_probability, 4),
            "SVM": round(svm_probability, 4),
        }
        stacked_model_input = np.array([[ann_probability, rf_probability, svm_probability, fuzzy_probability]])
        final_probability = round(
            _safe_probability(ARTIFACTS["meta_model"].predict_proba(stacked_model_input)[0][1]),
            4,
        )
        risk_level = _risk_level(final_probability)

        healthy_frame = pd.DataFrame(
            [ARTIFACTS["healthy_average"]],
            columns=ARTIFACTS["all_features"],
        )
        healthy_scaled = ARTIFACTS["scaler"].transform(healthy_frame)[0]
        feature_importance = _build_feature_importance(patient_scaled, healthy_scaled)

        response = {
            "probability": final_probability,
            "risk_level": risk_level,
            "model_predictions": model_predictions,
            "feature_importance": feature_importance,
            "advice": ADVICE_BY_LEVEL[risk_level],
            "imputed_fields": imputed_fields,
            "healthy_average": ARTIFACTS["healthy_average"],
            "profile_comparison": {
                "patient": {
                    feature: round(float(value), 4)
                    for feature, value in zip(ARTIFACTS["all_features"], patient_scaled)
                },
                "healthy": {
                    feature: round(float(value), 4)
                    for feature, value in zip(ARTIFACTS["all_features"], healthy_scaled)
                },
            },
        }
        return jsonify(response)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify(
            {
                "error": (
                    "We could not generate a prediction right now. "
                    f"Please verify your inputs and try again. Details: {exc}"
                )
            }
        ), 500


@app.route("/api/dataset", methods=["GET"])
def dataset():
    dataset_name = request.args.get("name", "augmented")
    if dataset_name not in {"raw", "cleaned", "augmented", "user"}:
        return jsonify({"error": "Unsupported dataset name."}), 400

    try:
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", DATASET_PAGE_SIZE))
    except ValueError:
        return jsonify({"error": "Page and page_size must be integers."}), 400

    return jsonify(_build_dataset_payload(dataset_name, page, page_size))


@app.route("/api/generate-sample", methods=["GET"])
def generate_sample():
    kind = request.args.get("kind", "random")
    if kind not in {"healthy", "at_risk", "random"}:
        return jsonify({"error": "Unsupported sample mode."}), 400

    try:
        return jsonify({"sample": _sample_patient(kind)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/training-data", methods=["POST"])
def save_training_data():
    try:
        payload = request.get_json(silent=True) or {}
        training_row, imputed_fields = _prepare_training_row(payload)
        append_user_training_row(training_row)
        counts = _get_dataset_counts()
        message = "Training row saved successfully."
        if imputed_fields:
            friendly_names = [feature.replace("_", " ").title() for feature in imputed_fields]
            message += f" Missing values were auto-filled for: {', '.join(friendly_names)}."
        return jsonify(
            {
                "message": message,
                "user_rows": get_user_row_count(),
                "counts": counts,
                "imputed_fields": imputed_fields,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Unable to save training row: {exc}"}), 500


@app.route("/api/training-data", methods=["DELETE"])
def clear_training_data():
    if TRAINING_THREAD is not None and TRAINING_THREAD.is_alive():
        return jsonify({"error": "Cannot remove website data while retraining is running."}), 409

    try:
        payload = request.get_json(silent=True) or {}
        retrain_mode = str(payload.get("retrain_mode", "fast")).strip().lower()
        if retrain_mode not in {"none", "fast", "full"}:
            return jsonify({"error": "retrain_mode must be 'none', 'fast', or 'full'."}), 400

        trained_user_rows = ARTIFACTS.get("training_summary", {}).get("user_rows_used", 0)
        clear_user_training_data()
        counts = _get_dataset_counts()
        response_payload = {
            "message": "Website-added rows were removed.",
            "counts": counts,
        }
        if trained_user_rows > 0 and retrain_mode in {"fast", "full"}:
            response_payload["message"] = (
                f"Website-added rows were removed. {retrain_mode.title()} retraining has started so the active models forget them too."
            )
            response_payload["status"] = _start_retraining(
                f"Rows removed. {retrain_mode.title()} retraining started so the active models forget website-added data.",
                mode=retrain_mode,
            )
        elif trained_user_rows > 0:
            response_payload["message"] = (
                "Website-added rows were removed. The current saved models still include previously trained website data until you run Fast Retrain or Full Retrain."
            )
        return jsonify(
            response_payload
        )
    except Exception as exc:
        return jsonify({"error": f"Unable to remove website-added rows: {exc}"}), 500


@app.route("/api/retrain", methods=["POST"])
def retrain():
    try:
        payload = request.get_json(silent=True) or {}
        mode = str(payload.get("mode", "full")).strip().lower()
        if mode not in {"full", "fast"}:
            return jsonify({"error": "Retraining mode must be 'full' or 'fast'."}), 400
        status_message = (
            "Fast retraining request accepted. Worker is starting..."
            if mode == "fast"
            else "Full retraining request accepted. Worker is starting..."
        )
        status = _start_retraining(status_message, mode=mode)
        return jsonify({"message": "Retraining started.", "status": status})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    except Exception as exc:
        return jsonify({"error": f"Unable to start retraining: {exc}"}), 500


@app.route("/api/retrain/status", methods=["GET"])
def retrain_status():
    with TRAINING_LOCK:
        status = dict(TRAINING_STATUS)
    status["user_rows"] = get_user_row_count()
    status["dataset_counts"] = _get_dataset_counts()
    return jsonify(status)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
