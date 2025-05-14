import flwr as fl
import os

# Make tensorflow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def weighted_average(metrics):
    total_examples = 0
    # Khởi tạo federated_metrics chỉ với các keys có giá trị là số từ metrics đầu tiên
    # và đảm bảo chúng ta không cố gắng tổng hợp các giá trị không phải số
    federated_metrics = {}
    if metrics and metrics[0][1]:
        for k_init, v_init in metrics[0][1].items():
            if isinstance(v_init, (int, float)):  # Chỉ khởi tạo nếu là số
                federated_metrics[k_init] = 0.0  # Dùng float để nhất quán

    if not federated_metrics:  # Nếu không có metric số nào
        return {}

    for num_examples, m in metrics:
        for k, v in m.items():
            if k in federated_metrics and isinstance(
                v, (int, float)
            ):  # Chỉ xử lý các key đã khởi tạo và là số
                federated_metrics[k] += num_examples * v
        total_examples += num_examples

    if total_examples == 0:
        return {k: 0.0 for k in federated_metrics}  # Tránh chia cho 0

    return {k: v / total_examples for k, v in federated_metrics.items()}


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,
        "local_epochs": 1,  # Số epoch huấn luyện local mỗi round
        "batch_size": 64,
        "num_rounds": 9,  # Tổng số rounds FL
        # Thêm các config khác nếu cần
    }
    return config


def get_server_strategy():
    def weighted_average(metrics):
        accs = [num_examples * m["accuracy"] for num_examples, m in metrics]
        losses = [num_examples * m["loss"] for num_examples, m in metrics]
        total = sum(num_examples for num_examples, _ in metrics)
        return {
            "accuracy": sum(accs) / total,
            "loss": sum(losses) / total,
        }

    return fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        on_fit_config_fn=None,
    )


if __name__ == "__main__":
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=get_server_strategy(),
        config=fl.server.ServerConfig(num_rounds=3),
    )
    final_round, acc = history.metrics_distributed["accuracy"][-1]
    print(f"After {final_round} rounds of training the accuracy is {acc:.3%}")
