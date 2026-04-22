"""demo: A Flower / PyTorch app."""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

import hivewatch
from hivewatch.emitters import WandbEmitter, MLflowEmitter

from demo.task import Net, get_weights


# ── Metrics aggregation ───────────────────────────────────────────────────────

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate per-client metrics weighted by number of samples.
    Used by both fit_metrics_aggregation_fn and evaluate_metrics_aggregation_fn.
    """
    total = sum(num for num, _ in metrics)
    if total == 0:
        return {}

    def wavg(key: str) -> float:
        return sum(num * m.get(key, 0.0) for num, m in metrics) / total

    return {
        "accuracy":       wavg("accuracy"),
        "local_accuracy": wavg("local_accuracy"),
        "local_loss":     wavg("local_loss"),
    }


# ── HiveWatch strategy ───────────────────────────────────────────────────────────

class HivewatchStrategy(FedAvg):
    """
    FedAvg subclass that injects hivewatch logging hooks.
    All aggregation logic lives in FedAvg — we only add logging.

    Clients should return these keys in their fit() metrics dict:
        local_loss, local_accuracy, gradient_norm,
        bytes_sent, train_time_sec, cpu_pct, ram_mb,
        lat, lng, city, country  (optional, for geo map)
    """

    def aggregate_fit(self, server_round, results, failures):
        # Log client dropouts
        for failure in failures:
            cid = failure[0].cid if isinstance(failure, tuple) else "unknown"
            hivewatch.log_dropout(server_round, cid, reason="fit failure")

        # Log per-client updates
        for proxy, fit_res in results:
            m = fit_res.metrics or {}
            hivewatch.log_client_update(
                client_id      = m.get("client_id", proxy.cid),
                round          = server_round,
                local_loss     = m.get("local_loss"),
                local_accuracy = m.get("local_accuracy"),
                num_samples    = fit_res.num_examples,
                gradient_norm  = m.get("gradient_norm"),
                bytes_sent     = m.get("bytes_sent", 0),
                train_time_sec = m.get("train_time_sec"),
                cpu_pct        = m.get("cpu_pct"),
                ram_mb         = m.get("ram_mb"),
                lat            = m.get("lat"),
                lng            = m.get("lng"),
                city           = m.get("city"),
                country        = m.get("country"),
            )

        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        if aggregated:
            loss, metrics = aggregated
            hivewatch.log_round(
                round           = server_round,
                global_accuracy = float(metrics.get("accuracy", 0.0)),
                global_loss     = float(loss) if loss else 0.0,
                num_stragglers  = len(failures),
            )

        return aggregated


# ── Server function ───────────────────────────────────────────────────────────

def server_fn(context: Context):
    num_rounds   = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Init hivewatch
    hivewatch.init(
        algorithm = "FedAvg",
        config    = dict(context.run_config),
        emitters  = [
            WandbEmitter(
                project = "my-fl-project",
                log_geo = True,
            ),
            MLflowEmitter(
                tracking_uri = "http://localhost:5000",
                experiment   = "my-fl-project-mlflow",
                log_geo      = True,
                mlflow_system_metrics  = True,
                system_metrics_sampling_interval = 5,
            ),
        ],
    )

    ndarrays   = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = HivewatchStrategy(
        fraction_fit                    = fraction_fit,
        fraction_evaluate               = 1.0,
        min_available_clients           = 2,
        initial_parameters              = parameters,
        fit_metrics_aggregation_fn      = weighted_average,
        evaluate_metrics_aggregation_fn = weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)