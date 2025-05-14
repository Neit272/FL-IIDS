import numpy as np
import random


class DynamicExampleMemory:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.data = {}  # {class_label: list of (x, loss)}
        self.class_allocations = {}  # {class_label: allocated_size}
        self.current_task_id = 0
        print(f"DEM Initialized with max_size: {max_size}")

    def update_task(self, task_id: int, task_class_counts: dict):
        """Update memory allocation when a new task starts."""
        print(f"DEM Updating for Task {task_id}")
        self.current_task_id = task_id

        all_seen_classes = list(self.data.keys())
        if all_seen_classes:
            per_class = self.max_size // len(all_seen_classes)
            for cls in all_seen_classes:
                self.class_allocations[cls] = per_class
        print(f"DEM class allocations (simplified): {self.class_allocations}")

    def store_samples(self, x_data: np.ndarray, y_data: np.ndarray, losses=None):
        """
        Store new samples into memory.
        - x_data: (N, feature_dim)
        - y_data: (N,)
        - losses: optional list of loss values for each sample (N,)
        """
        if losses is None:
            losses = [0.0] * len(y_data)  # fallback if not provided

        for x, y, l in zip(x_data, y_data, losses):
            y = int(y)
            if y not in self.data:
                self.data[y] = []

            self.data[y].append((x, float(l)))  # store as tuple

            # Sort by loss descending
            self.data[y].sort(key=lambda e: e[1], reverse=True)

            # Trim to class allocation limit
            max_c = self.class_allocations.get(y, self.max_size // len(self.data))
            if len(self.data[y]) > max_c:
                self.data[y] = self.data[y][:max_c]

        print(f"DEM: stored samples. Current summary: {self.get_memory_summary()}")

    def sample(self, batch_size: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Sample a random minibatch from memory."""
        all_samples_x = []
        all_samples_y = []
        for label, samples in self.data.items():
            if samples:
                for x, _ in samples:
                    all_samples_x.append(x)
                    all_samples_y.append(label)

        if not all_samples_x:
            return None, None

        indices = np.random.choice(
            len(all_samples_x), min(batch_size, len(all_samples_x)), replace=False
        )
        x_batch = np.array([all_samples_x[i] for i in indices])
        y_batch = np.array([all_samples_y[i] for i in indices])
        return x_batch, y_batch

    def get_memory_summary(self) -> dict:
        return {label: len(samples) for label, samples in self.data.items()}
