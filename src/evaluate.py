import json
import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), ".cache", "matplotlib"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from src.data_download import get_project_root


def _get_results_dir():
    path = os.path.join(get_project_root(), "results")
    os.makedirs(path, exist_ok=True)
    return path


def _get_metrics_path():
    path = os.path.join(get_project_root(), "models", "metrics.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def evaluate_models(y_true, model_outputs):
    try:
        os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
        metrics = {}

        print("-" * 72)
        print(f"{'Model':<18}{'Accuracy':>12}{'Precision':>12}{'Recall':>12}{'F1':>12}")
        print("-" * 72)

        for model_name, output in model_outputs.items():
            predictions = output["predictions"]
            probabilities = output["probabilities"]
            model_metrics = {
                "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
                "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
                "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
            }
            metrics[model_name] = model_metrics
            print(
                f"{model_name:<18}"
                f"{model_metrics['accuracy']:>12.4f}"
                f"{model_metrics['precision']:>12.4f}"
                f"{model_metrics['recall']:>12.4f}"
                f"{model_metrics['f1']:>12.4f}"
            )

        print("-" * 72)

        with open(_get_metrics_path(), "w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)

        results_dir = _get_results_dir()
        model_names = list(model_outputs.keys())
        accuracy_values = [metrics[name]["accuracy"] for name in model_names]

        plt.figure(figsize=(8, 5))
        sns.barplot(x=model_names, y=accuracy_values, hue=model_names, palette="crest", legend=False)
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "accuracy_comparison.png"))
        plt.close()

        grid_columns = 2
        grid_rows = int(np.ceil(len(model_names) / grid_columns))
        figure, axes = plt.subplots(grid_rows, grid_columns, figsize=(10, 4 * grid_rows))
        axes = np.atleast_1d(axes).flatten()
        for axis, model_name in zip(axes, model_names):
            matrix = confusion_matrix(y_true, model_outputs[model_name]["predictions"])
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
            axis.set_title(model_name)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
        for axis in axes[len(model_names):]:
            axis.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "confusion_matrices.png"))
        plt.close()

        plt.figure(figsize=(8, 5))
        for model_name in model_names:
            fpr, tpr, _ = roc_curve(y_true, model_outputs[model_name]["probabilities"])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="#777777")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "roc_curves.png"))
        plt.close()

        return metrics
    except Exception as exc:
        raise RuntimeError(f"Unable to evaluate models: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit("Run train.py to evaluate models.")
