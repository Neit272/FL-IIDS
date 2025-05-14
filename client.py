import os
import flwr as fl
import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, List, Optional

from utils.config import USE_DEM, USE_LGB_LLS
from utils.data_loader import get_data
from utils.model_loader import get_model
from utils.memory import DynamicExampleMemory
from utils.losses import (
    calculate_lls_loss,
    calculate_lgb_proxy_loss,
    GAMMA_LLS,
    GAMMA_LGB,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

CLASS_MAPPING = {
    "Normal": 0,
    "Analysis": 1,
    "Backdoor": 2,
    "DoS": 3,
    "Exploits": 4,
    "Fuzzers": 5,
    "Generic": 6,
    "Reconnaissance": 7,
    "Shellcode": 8,
}
TASK_CLASSES = {
    1: [CLASS_MAPPING[c] for c in ["Normal", "Analysis", "Backdoor"]],
    2: [CLASS_MAPPING[c] for c in ["DoS", "Exploits", "Fuzzers"]],
    3: [CLASS_MAPPING[c] for c in ["Generic", "Reconnaissance", "Shellcode"]],
}
NUM_CLASSES = len(CLASS_MAPPING)


class FLIIDSModel(tf.keras.Model):
    def __init__(self, original_model, num_classes):
        super().__init__(inputs=original_model.inputs, outputs=original_model.outputs)
        self.original_model = original_model
        self.num_classes = num_classes
        self.best_old_model_weights = None
        self.old_class_indices = None
        self.gamma_lls = GAMMA_LLS
        self.gamma_lgb = GAMMA_LGB
        self.old_model_inference = None

    def compile(self, optimizer, metrics, **kwargs):
        super().compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(name="compiled_loss"),
            metrics=metrics,
            **kwargs
        )
        if self.best_old_model_weights is not None:
            self.old_model_inference = tf.keras.models.clone_model(self.original_model)
            self.old_model_inference.set_weights(self.best_old_model_weights)
            self.old_model_inference.trainable = False

    def set_incremental_params(self, old_weights, old_indices):
        self.best_old_model_weights = old_weights
        self.old_class_indices = (
            tf.constant(old_indices, dtype=tf.int32) if old_indices else None
        )
        if self.best_old_model_weights is not None:
            if self.old_model_inference is None:
                self.old_model_inference = tf.keras.models.clone_model(
                    self.original_model
                )
            self.old_model_inference.set_weights(self.best_old_model_weights)
            self.old_model_inference.trainable = False


    @tf.function
    def train_step(self, data):
        x, y_true, is_old_flags = data

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred_current = self(x, training=True)

            # 🔹 [1] Always use a primary CCE loss
            cce_loss_fn = tf.keras.losses.CategoricalCrossentropy()
            primary_loss = cce_loss_fn(y_true, y_pred_current)

            # 🔹 [2] Optionally compute LGB loss
            if USE_LGB_LLS:
                loss_lgb = calculate_lgb_proxy_loss(
                    y_true, y_pred_current, {"is_old": is_old_flags}
                )
            else:
                loss_lgb = tf.constant(0.0)

            # 🔹 [3] Optionally compute LLS loss
            if (
                USE_LGB_LLS
                and self.old_model_inference is not None
                and self.old_class_indices is not None
            ):
                y_pred_old = self.old_model_inference(x, training=False)
                loss_lls = calculate_lls_loss(
                    y_true, y_pred_current, y_pred_old, self.old_class_indices
                )
            else:
                loss_lls = tf.constant(0.0)

            # 🔹 [4] Final loss
            total_loss = (
                primary_loss
                + self.gamma_lgb * loss_lgb
                + self.gamma_lls * loss_lls
                + sum(self.losses)
            )

        # Backward pass
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics
        self.compiled_loss(y_true, y_pred_current)
        self.compiled_metrics.update_state(y_true, y_pred_current)

        return {m.name: m.result() for m in self.metrics}


class Client(fl.client.NumPyClient):
    def __init__(self, client_id="0", dem_max_size=500):
        self.client_id = client_id
        self.X_train_full, self.Y_train_full, self.X_test, self.Y_test = get_data()
        self.Y_train_full = self.Y_train_full.astype(int)
        self.Y_test = self.Y_test.astype(int)
        self.num_classes = NUM_CLASSES

        self.base_model = get_model(self.X_train_full.shape[1:], self.num_classes)
        self.model = FLIIDSModel(self.base_model, self.num_classes)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.CategoricalCrossentropy(name="loss"),
            ],
        )

        self.memory = DynamicExampleMemory(max_size=dem_max_size) if USE_DEM else None
        self.current_task_id = 0
        self.best_old_model_weights = None
        self.all_trained_class_indices = []

    def get_parameters(self, config):
        return self.model.get_weights()

    def _load_data_for_task(self, task_id):
        if task_id not in TASK_CLASSES:
            return np.array([]), np.array([])
        classes = TASK_CLASSES[task_id]
        idx = np.isin(self.Y_train_full, classes)
        return self.X_train_full[idx], self.Y_train_full[idx]

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        round_id = config.get("server_round", 1)
        total_rounds = config.get("num_rounds", 9)
        batch_size = config.get("batch_size", 64)
        epochs = config.get("local_epochs", 1)

        num_tasks = len(TASK_CLASSES)
        rounds_per_task = max(1, total_rounds // num_tasks)
        new_task_id = min(((round_id - 1) // rounds_per_task) + 1, num_tasks)

        if new_task_id != self.current_task_id:
            if self.current_task_id > 0:
                self.best_old_model_weights = self.model.get_weights()
            self.current_task_id = new_task_id

            if USE_DEM and self.memory is not None:
                self.memory.update_task(self.current_task_id, {})

            self.all_trained_class_indices = sorted(
                set(
                    cls
                    for i in range(1, self.current_task_id + 1)
                    for cls in TASK_CLASSES.get(i, [])
                )
            )

        x_new, y_new = self._load_data_for_task(self.current_task_id)

        # sample from memory (if enabled)
        if USE_DEM and self.memory is not None:
            x_mem, y_mem = self.memory.sample(batch_size=batch_size // 2)
        else:
            x_mem, y_mem = None, None

        if x_mem is not None and len(x_mem) > 0:
            x_combined = np.concatenate((x_new, x_mem))
            y_combined = np.concatenate((y_new, y_mem))
            is_old = np.concatenate(
                (np.zeros(len(x_new), dtype=bool), np.ones(len(x_mem), dtype=bool))
            )
        else:
            x_combined = x_new
            y_combined = y_new
            is_old = np.zeros(len(x_combined), dtype=bool)

        y_one_hot = tf.keras.utils.to_categorical(
            y_combined, num_classes=self.num_classes
        )
        dataset = tf.data.Dataset.from_tensor_slices((x_combined, y_one_hot, is_old))
        dataset = (
            dataset.shuffle(len(x_combined))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        old_classes = [
            i
            for i in self.all_trained_class_indices
            if i not in TASK_CLASSES[self.current_task_id]
        ]
        self.model.set_incremental_params(self.best_old_model_weights, old_classes)

        history = self.model.fit(dataset, epochs=epochs, verbose=0)

        if USE_DEM and self.memory is not None and len(x_new) > 0:
            self.memory.store_samples(x_new, y_new)

        loss = history.history.get("loss", [0.0])[-1]
        acc = history.history.get("accuracy", [0.0])[-1]
        return (
            self.model.get_weights(),
            len(x_combined),
            {"loss": float(loss), "accuracy": float(acc)},
        )

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        y_test_one_hot = tf.keras.utils.to_categorical(
            self.Y_test, num_classes=self.num_classes
        )
        results = self.model.evaluate(self.X_test, y_test_one_hot, verbose=0)
        return (
            float(results[0]),
            len(self.X_test),
            {
                "loss": float(results[0]),
                "accuracy": float(results[1]),
            },
        )


def create_client(cid):
    return Client(client_id=cid)


if __name__ == "__main__":
    client_id = os.getenv("CLIENT_ID", "client_0")
    server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1:8080")
    client = Client(client_id=client_id)
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client(),
    )
