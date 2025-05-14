import os
import argparse
import json
import flwr as fl
from utils.config import apply_config
from server import get_server_strategy
from client import Client

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def create_client_fn():
    def client_fn(cid):
        return Client(client_id=cid)

    return client_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["baseline", "dem", "full"], default="baseline"
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--clients", type=int, default=3)
    args = parser.parse_args()

    # Apply mode flags
    apply_config(args.mode)
    print(f"[simulation.py] Running FL-IIDS in mode: {args.mode.upper()}")

    history = fl.simulation.start_simulation(
        client_fn=create_client_fn(),
        num_clients=args.clients,
        strategy=get_server_strategy(),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )

    final_round, acc = history.metrics_distributed["accuracy"][-1]
    print(f"After {final_round} rounds, global evaluation accuracy = {acc:.3%}")
    with open(f"logs/results_{args.mode}.json", "w") as f:
        json.dump(history.metrics_distributed, f, indent=4)
