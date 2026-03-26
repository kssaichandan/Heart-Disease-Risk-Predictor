import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data_download import get_project_root

RANDOM_STATE = 42


def get_rf_model_path():
    return os.path.join(get_project_root(), "models", "rf_model.pkl")


def build_random_forest_estimator(**overrides):
    estimator_config = {
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
        "n_jobs": 1,
    }
    estimator_config.update(overrides)
    return RandomForestClassifier(**estimator_config)


def train_random_forest(x_train, y_train, x_valid, feature_names, fast_mode=False, fixed_params=None):
    try:
        if fast_mode:
            fast_params = fixed_params or {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 5,
                "min_samples_leaf": 1,
            }
            model = build_random_forest_estimator(**fast_params)
            model.fit(x_train, y_train)
            best_params = fast_params
        else:
            estimator = build_random_forest_estimator()
            parameter_grid = {
                "n_estimators": [200],
                "max_depth": [None, 8, 12],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
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

        os.makedirs(os.path.dirname(get_rf_model_path()), exist_ok=True)
        joblib.dump(model, get_rf_model_path())

        probabilities = model.predict_proba(x_valid)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        ranked_features = sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
        print(
            "Random Forest "
            f"{'reused params (fast mode)' if fast_mode else 'best params'}:",
            best_params,
        )
        print("Random Forest feature importance:")
        for feature_name, importance in ranked_features:
            print(f"  {feature_name}: {importance:.4f}")

        return model, predictions, probabilities
    except Exception as exc:
        raise RuntimeError(f"Unable to train Random Forest model: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit("Run train.py to train the Random Forest model.")
