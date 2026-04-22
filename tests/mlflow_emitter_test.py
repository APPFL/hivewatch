from __future__ import annotations

import os
import types

import pytest

from hivewatch.emitters.mlflow_emitter import MLflowEmitter
from hivewatch.schema import ClientUpdate, RoundSummary


class FakeMetric:
    def __init__(self, key: str, value: float, timestamp: int, step: int):
        self.key = key
        self.value = value
        self.timestamp = timestamp
        self.step = step


class FakeMlflowClient:
    def __init__(self):
        self.log_batches: list[dict] = []
        self.tags: list[tuple[str, str, str]] = []
        self.artifacts: list[tuple[str, str, str]] = []

    def log_batch(self, run_id: str, metrics: list[FakeMetric]):
        self.log_batches.append({"run_id": run_id, "metrics": metrics})

    def set_tag(self, run_id: str, key: str, value: str):
        self.tags.append((run_id, key, value))

    def log_artifact(self, run_id: str, path: str, artifact_path: str):
        self.artifacts.append((run_id, path, artifact_path))


class FakeMlflowModule:
    def __init__(self):
        self.set_tracking_uri_calls: list[str] = []
        self.set_experiment_calls: list[str] = []
        self.start_run_calls: list[dict] = []
        self.log_params_calls: list[dict] = []
        self.enable_system_metrics_logging_calls = 0
        self.end_run_calls = 0
        self.active_run_value = None
        self.client = FakeMlflowClient()
        self.tracking = types.SimpleNamespace(MlflowClient=lambda: self.client)
        self.entities = types.SimpleNamespace(Metric=FakeMetric)

    def set_tracking_uri(self, uri: str):
        self.set_tracking_uri_calls.append(uri)

    def enable_system_metrics_logging(self):
        self.enable_system_metrics_logging_calls += 1

    def active_run(self):
        return self.active_run_value

    def set_experiment(self, experiment: str):
        self.set_experiment_calls.append(experiment)

    def start_run(self, **kwargs):
        self.start_run_calls.append(kwargs)
        run_id = kwargs.get("run_id", "fresh-run-id")
        return types.SimpleNamespace(info=types.SimpleNamespace(run_id=run_id))

    def log_params(self, params: dict):
        self.log_params_calls.append(params)

    def end_run(self):
        self.end_run_calls += 1


@pytest.fixture
def fake_mlflow(monkeypatch):
    module = FakeMlflowModule()
    monkeypatch.setitem(__import__("sys").modules, "mlflow", module)
    return module


def metric_map(batch: dict) -> dict[str, float]:
    return {metric.key: metric.value for metric in batch["metrics"]}


def make_client(**overrides) -> ClientUpdate:
    data = {
        "client_id": "client-1",
        "round": 3,
        "local_accuracy": 0.9,
        "local_loss": 0.2,
        "num_samples": 10,
        "gradient_norm": 1.2,
        "gradient_magnitude": 0.5,
        "sparsity": 0.1,
        "compression_ratio": 0.8,
        "bytes_sent": 2_000_000,
        "bytes_received": 500_000,
        "network_latency_ms": 25.0,
        "train_time_sec": 4.0,
        "base_round": 1,
        "cpu_pct": 55.0,
        "ram_mb": 256.0,
        "gpu_util_pct": 70.0,
        "gpu_vram_mb": 1024.0,
        "lat": 42.0,
        "lng": -71.0,
        "city": "Boston",
        "country": "US",
        "status": "active",
    }
    data.update(overrides)
    return ClientUpdate(**data)


def test_mlflow_on_init_starts_new_run_and_logs_flattened_config(fake_mlflow):
    emitter = MLflowEmitter(
        tracking_uri="http://mlflow.local",
        experiment="exp-name",
        tags={"team": "research"},
    )

    emitter.on_init("hivewatch-run", "FedAvg", {"optimizer": {"lr": 0.1}, "epochs": 2})

    assert fake_mlflow.set_tracking_uri_calls == ["http://mlflow.local"]
    assert fake_mlflow.set_experiment_calls == ["exp-name"]
    assert fake_mlflow.start_run_calls == [
        {
            "run_name": "hivewatch-run",
            "tags": {
                "hivewatch/run_id": "hivewatch-run",
                "hivewatch/algorithm": "FedAvg",
                "team": "research",
            },
        }
    ]
    assert fake_mlflow.log_params_calls == [{"optimizer.lr": "0.1", "epochs": "2"}]
    assert emitter._run_id == "fresh-run-id"
    assert emitter._client is fake_mlflow.client


def test_mlflow_on_init_can_resume_run_and_enable_system_metrics(fake_mlflow, monkeypatch):
    monkeypatch.delenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", raising=False)
    emitter = MLflowEmitter(
        resume_run_id="existing-run",
        mlflow_system_metrics=True,
        system_metrics_sampling_interval=7,
    )

    emitter.on_init("hivewatch-run", "FedAvg", {"ignored": True})

    assert fake_mlflow.enable_system_metrics_logging_calls == 1
    assert os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] == "7"
    assert fake_mlflow.start_run_calls == [{"run_id": "existing-run"}]
    assert fake_mlflow.set_experiment_calls == []
    assert fake_mlflow.log_params_calls == []
    assert emitter._run_id == "existing-run"


def test_mlflow_on_round_logs_round_client_system_and_geo_metrics(fake_mlflow):
    emitter = MLflowEmitter(log_geo=True)
    emitter.on_init("hivewatch-run", "FedAvg", {})
    client = make_client()
    summary = RoundSummary(
        round=3,
        global_accuracy=0.95,
        global_loss=0.1,
        num_selected=2,
        num_completed=1,
        num_stragglers=1,
        num_failures=0,
        total_bytes_up=2_000_000,
        total_bytes_down=500_000,
        round_duration_sec=12.0,
        gradient_divergence=0.3,
        aggregation_time_sec=1.5,
        algorithm_metadata={"mu": 0.01, "strategy": "fedavg"},
    )

    emitter.on_round(summary, [client])

    assert len(fake_mlflow.client.log_batches) == 1
    batch = fake_mlflow.client.log_batches[0]
    metrics = metric_map(batch)
    assert batch["run_id"] == "fresh-run-id"
    assert metrics["round/accuracy"] == pytest.approx(0.95)
    assert metrics["round/participation_rate"] == pytest.approx(0.5)
    assert metrics["comm/total_bytes_mb"] == pytest.approx(2.5)
    assert metrics["agg/algo/mu"] == pytest.approx(0.01)
    assert "agg/algo/strategy" not in metrics
    assert metrics["client.client-1.accuracy"] == pytest.approx(0.9)
    assert metrics["client.client-1.staleness"] == pytest.approx(2.0)
    assert metrics["sys.client-1.gpu_vram_mb"] == pytest.approx(1024.0)
    assert ("fresh-run-id", "geo.client-1.city", "Boston") in fake_mlflow.client.tags
    assert ("fresh-run-id", "geo.client-1.country", "US") in fake_mlflow.client.tags


def test_mlflow_client_update_checkpoint_and_finish(fake_mlflow):
    emitter = MLflowEmitter(log_geo=True)
    emitter.on_init("hivewatch-run", "FedAvg", {})
    emitter.on_client_update(make_client(round=None))
    assert fake_mlflow.client.log_batches == []

    emitter.on_client_update(make_client(round=4, status="failed"))
    assert len(fake_mlflow.client.log_batches) == 1
    update_metrics = metric_map(fake_mlflow.client.log_batches[0])
    assert update_metrics["client.client-1.status"] == pytest.approx(3.0)

    emitter.on_checkpoint(4, "/tmp/model.pt", {"format": "pt"})
    assert fake_mlflow.client.artifacts == [
        ("fresh-run-id", "/tmp/model.pt", "checkpoints/round_4")
    ]
    checkpoint_metrics = metric_map(fake_mlflow.client.log_batches[1])
    assert checkpoint_metrics["event/checkpoint"] == pytest.approx(1.0)

    emitter.finish()
    assert fake_mlflow.end_run_calls == 1
    assert emitter._run is None
    assert emitter._run_id is None
    assert emitter._client is None
