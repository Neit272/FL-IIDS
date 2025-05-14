import json
import matplotlib.pyplot as plt

MODES = ["baseline", "dem", "full"]
COLORS = {"baseline": "blue", "dem": "orange", "full": "green"}


def load_metrics(mode):
    with open(f"logs/results_{mode}.json", "r") as f:
        data = json.load(f)
    rounds = [r for r, _ in data.get("accuracy", [])]
    accs = [a for _, a in data.get("accuracy", [])]
    losses = [l for _, l in data.get("loss", [])] if "loss" in data else [0] * len(accs)
    return rounds, accs, losses


def plot_metric(metric_name, ylabel):
    plt.figure()
    for mode in MODES:
        rounds, accs, losses = load_metrics(mode)
        values = accs if metric_name == "accuracy" else losses
        plt.plot(rounds, values, label=mode.upper(), color=COLORS[mode])
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(f"{metric_name.capitalize()} vs Round")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"logs/{metric_name}_comparison.png")
    plt.show()


if __name__ == "__main__":
    plot_metric("accuracy", "Accuracy")
    plot_metric("loss", "Loss")
