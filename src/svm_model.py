import os

import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from src.data_download import get_project_root

RANDOM_STATE = 42


def get_svm_model_path():
    return os.path.join(get_project_root(), "models", "svm_model.pkl")


def build_svm_estimator(**overrides):
    estimator_config = {
        "kernel": "rbf",
        "probability": True,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    }
    estimator_config.update(overrides)
    return SVC(**estimator_config)


def train_svm_model(x_train, y_train, x_valid, fast_mode=False, fixed_params=None):
    try:
        if fast_mode:
            fast_params = fixed_params or {"C": 25, "gamma": "scale"}
            model = build_svm_estimator(**fast_params)
            model.fit(x_train, y_train)
            best_params = fast_params
        else:
            estimator = build_svm_estimator()
            parameter_grid = {
                "C": [0.5, 1, 10, 25],
                "gamma": ["scale", 0.1, 0.01, 0.001],
            }
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=parameter_grid,
                scoring="accuracy",
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                n_jobs=1,
                refit=True,
            )
            grid_search.fit(x_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_

        os.makedirs(os.path.dirname(get_svm_model_path()), exist_ok=True)
        joblib.dump(model, get_svm_model_path())

        probabilities = model.predict_proba(x_valid)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        print(
            "SVM "
            f"{'reused params (fast mode)' if fast_mode else 'best params'}:",
            best_params,
        )
        return model, predictions, probabilities
    except Exception as exc:
        raise RuntimeError(f"Unable to train SVM model: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit("Run train.py to train the SVM model.")
