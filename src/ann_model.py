import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), ".cache", "matplotlib"),
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.data_download import get_project_root

RANDOM_STATE = 42


def _get_tf_modules():
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input

    tf.get_logger().setLevel("ERROR")
    return tf, Sequential, Dense, Dropout, Input


def _get_matplotlib_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def get_ann_model_path():
    return os.path.join(get_project_root(), "models", "ann_model.h5")


def build_ann_model(input_dim):
    tf, Sequential, Dense, Dropout, Input = _get_tf_modules()

    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_ann_model(x_train, y_train, x_valid, y_valid, epochs=100, batch_size=32):
    try:
        tf, _, _, _, _ = _get_tf_modules()
        plt = _get_matplotlib_pyplot()
        os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
        tf.keras.utils.set_random_seed(RANDOM_STATE)

        model = build_ann_model(x_train.shape[1])

        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        )

        os.makedirs(os.path.dirname(get_ann_model_path()), exist_ok=True)
        model.save(get_ann_model_path())

        results_dir = os.path.join(get_project_root(), "results")
        os.makedirs(results_dir, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("ANN Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "ann_training_history.png"))
        plt.close()

        probabilities = model.predict(x_valid, verbose=0).flatten()
        predictions = (probabilities >= 0.5).astype(int)
        return model, predictions, probabilities
    except Exception as exc:
        raise RuntimeError(f"Unable to train ANN model: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit("Run train.py to train the ANN model.")
