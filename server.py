import flwr as fl
import os
import json
from utils import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def weighted_average(metrics):
    accs = [
        num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m
    ]
    losses = [num_examples * m["loss"] for num_examples, m in metrics if "loss" in m]
    total = sum(num_examples for num_examples, _ in metrics)

    if total == 0:
        return {"accuracy": 0.0, "loss": 0.0}

    return {
        "accuracy": sum(accs) / total if accs else 0.0,
        "loss": sum(losses) / total if losses else 0.0,
    }


def fit_config(server_round: int):
    return {
        "server_round": server_round,
        "local_epochs": 1,
        "batch_size": 64,
    }


def get_server_strategy():
    return fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        on_fit_config_fn=None, # I let this None cause i haven't find the most suitable config yet
    )


def save_log(history):
    os.makedirs("logs", exist_ok=True)
    filepath = os.path.join("logs", f"results_{config.MODE}.json")
    with open(filepath, "w") as f:
        json.dump(history.metrics_distributed, f, indent=4)
    print(f"[server.py] Log saved to {filepath}")


if __name__ == "__main__":
    print(f"[server.py] Starting server in mode: {config.MODE.upper()}")
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=get_server_strategy(),
        config=fl.server.ServerConfig(num_rounds=3),
    )
    final_round, acc = history.metrics_distributed["accuracy"][-1]
    print(f"[server.py] After {final_round} rounds, accuracy = {acc:.3%}")
    save_log(history)
