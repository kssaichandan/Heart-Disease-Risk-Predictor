import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.ann_model import build_ann_model
from src.data_download import get_project_root
from src.fuzzy_logic import FuzzyRiskModel
from src.random_forest import build_random_forest_estimator
from src.svm_model import build_svm_estimator

RANDOM_STATE = 42


def _get_tensorflow():
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    return tf


def get_meta_model_path():
    return os.path.join(get_project_root(), "models", "meta_model.pkl")


class ANNStackingEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=100, batch_size=32, random_state=RANDOM_STATE):
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, x, y):
        tf = _get_tensorflow()
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.random_state)
        self.model_ = build_ann_model(x.shape[1])
        self.model_.fit(
            x,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0,
        )
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x):
        positive = self.model_.predict(x, verbose=0).flatten()
        return np.column_stack([1 - positive, positive])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class FuzzyStackingEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = FuzzyRiskModel()

    def fit(self, x, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x):
        if isinstance(x, pd.DataFrame):
            frame = x.copy()
        else:
            frame = pd.DataFrame(x, columns=["age", "chol", "thalach"])

        positive = self.model.predict_proba(frame)
        return np.column_stack([1 - positive, positive])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def _normalize_coefficients(coefficients):
    weights = np.abs(coefficients)
    total = np.sum(weights)
    if total == 0:
        return [25.0, 25.0, 25.0, 25.0]
    return [round(float(weight / total * 100), 2) for weight in weights]


def train_meta_learner(
    x_train_selected,
    y_train,
    x_train_fuzzy,
    rf_params,
    svm_params,
    cv_splits=5,
    ann_epochs=100,
):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    ann_oof = cross_val_predict(
        ANNStackingEstimator(epochs=ann_epochs),
        x_train_selected,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    rf_oof = cross_val_predict(
        build_random_forest_estimator(**rf_params),
        x_train_selected,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    svm_oof = cross_val_predict(
        build_svm_estimator(**svm_params),
        x_train_selected,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    fuzzy_oof = cross_val_predict(
        FuzzyStackingEstimator(),
        x_train_fuzzy,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    stacked_train = np.column_stack([ann_oof, rf_oof, svm_oof, fuzzy_oof])
    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    meta_model.fit(stacked_train, y_train)

    os.makedirs(os.path.dirname(get_meta_model_path()), exist_ok=True)
    joblib.dump(meta_model, get_meta_model_path())

    coefficient_weights = _normalize_coefficients(meta_model.coef_[0])
    print(
        "Meta-learner trust split: "
        f"ANN {coefficient_weights[0]}%, "
        f"RF {coefficient_weights[1]}%, "
        f"SVM {coefficient_weights[2]}%, "
        f"Fuzzy {coefficient_weights[3]}%"
    )

    return meta_model


def evaluate_meta_learner(meta_model, base_test_probabilities, y_test):
    stacked_test = np.column_stack(
        [
            base_test_probabilities["ANN"],
            base_test_probabilities["Random Forest"],
            base_test_probabilities["SVM"],
            base_test_probabilities["Fuzzy Logic"],
        ]
    )
    probabilities = meta_model.predict_proba(stacked_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Stacked ensemble accuracy: {accuracy:.4f}")
    return predictions, probabilities
