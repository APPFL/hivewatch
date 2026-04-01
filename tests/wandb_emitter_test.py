from __future__ import annotations

import types

import pytest

from fedviz.emitters.wandb_emitter import WandbEmitter
from fedviz.schema import ClientUpdate, RoundSummary


class FakeArtifact:
    def __init__(self, name: str, type: str, metadata: dict):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[str] = []

    def add_file(self, path: str):
        self.files.append(path)


class FakeTable:
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.rows: list[tuple] = []

    def add_data(self, *values):
        self.rows.append(values)


class FakeRun:
    def __init__(self):
        self.logged_artifacts: list[FakeArtifact] = []
        self.finish_calls = 0

    def log_artifact(self, artifact: FakeArtifact):
        self.logged_artifacts.append(artifact)

    def finish(self):
        self.finish_calls += 1


class FakeWandbModule:
    def __init__(self):
        self.run = None
        self.init_calls: list[dict] = []
        self.define_metric_calls: list[tuple] = []
        self.log_calls: list[dict] = []
        self.alert_calls: list[dict] = []
        self.AlertLevel = types.SimpleNamespace(WARN="warn", ERROR="error")

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        self.run = FakeRun()
        return self.run

    def define_metric(self, *args, **kwargs):
        self.define_metric_calls.append((args, kwargs))

    def log(self, payload: dict, step: int | None = None):
        self.log_calls.append({"payload": payload, "step": step})

    def alert(self, **kwargs):
        self.alert_calls.append(kwargs)

    def Artifact(self, name: str, type: str, metadata: dict):
        return FakeArtifact(name=name, type=type, metadata=metadata)

    def Table(self, columns: list[str]):
        return FakeTable(columns=columns)


@pytest.fixture
def fake_wandb(monkeypatch):
    module = FakeWandbModule()
    monkeypatch.setitem(__import__("sys").modules, "wandb", module)
    return module


def make_client(**overrides) -> ClientUpdate:
    data = {
        "client_id": "client-1",
        "round": 5,
        "local_accuracy": 0.88,
        "local_loss": 0.22,
        "num_samples": 32,
        "gradient_norm": 0.7,
        "gradient_magnitude": 0.3,
        "sparsity": 0.05,
        "compression_ratio": 0.9,
        "bytes_sent": 1_500_000,
        "bytes_received": 250_000,
        "network_latency_ms": 12.0,
        "train_time_sec": 2.5,
        "base_round": 4,
        "cpu_pct": 45.0,
        "ram_mb": 128.0,
        "gpu_util_pct": 65.0,
        "gpu_vram_mb": 512.0,
        "lat": 35.0,
        "lng": -80.0,
        "city": "Charlotte",
        "country": "US",
        "status": "idle",
    }
    data.update(overrides)
    return ClientUpdate(**data)


def test_wandb_on_init_starts_run_and_registers_metric_namespaces(fake_wandb):
    emitter = WandbEmitter(
        project="fedviz-project",
        entity="team",
        tags=["mnist"],
        config={"lr": 0.1},
        mode="offline",
    )

    emitter.on_init("fedviz-run", "FedAvg", {"epochs": 3})

    assert fake_wandb.init_calls == [
        {
            "project": "fedviz-project",
            "entity": "team",
            "group": "FedAvg",
            "job_type": "federated-training",
            "name": "fedviz-run",
            "tags": ["mnist"],
            "config": {
                "fedviz/run_id": "fedviz-run",
                "fedviz/algorithm": "FedAvg",
                "epochs": 3,
                "lr": 0.1,
            },
            "mode": "offline",
        }
    ]
    assert fake_wandb.define_metric_calls[0] == (("round",), {})
    assert len(fake_wandb.define_metric_calls) == 7


def test_wandb_on_init_adopts_existing_run(fake_wandb):
    fake_wandb.run = FakeRun()
    emitter = WandbEmitter()

    emitter.on_init("fedviz-run", "FedAvg", {})

    assert fake_wandb.init_calls == []
    assert emitter._run is fake_wandb.run


def test_wandb_on_round_logs_metrics_geo_and_finish_flushes_table(fake_wandb):
    emitter = WandbEmitter(log_geo=True)
    emitter.on_init("fedviz-run", "FedAvg", {})
    client = make_client()
    summary = RoundSummary(
        round=5,
        global_accuracy=0.91,
        global_loss=0.19,
        num_selected=4,
        num_completed=2,
        num_stragglers=1,
        num_failures=1,
        total_bytes_up=1_500_000,
        total_bytes_down=250_000,
        round_duration_sec=6.0,
        gradient_divergence=0.15,
        aggregation_time_sec=0.6,
        algorithm_metadata={"mu": 0.01},
    )

    emitter.on_round(summary, [client])

    logged = fake_wandb.log_calls[0]
    assert logged["step"] == 5
    payload = logged["payload"]
    assert payload["round"] == 5
    assert payload["round/participation_rate"] == pytest.approx(0.5)
    assert payload["comm/total_bytes_mb"] == pytest.approx(1.75)
    assert payload["agg/algo/mu"] == pytest.approx(0.01)
    assert payload["client/client-1/status"] == 1
    assert payload["sys/client-1/gpu_vram_mb"] == pytest.approx(512.0)
    assert emitter._geo_table is not None
    assert emitter._geo_table.rows == [
        (5, "client-1", 35.0, -80.0, "Charlotte", "US", 0.88, 0.22, "idle")
    ]

    emitter.finish()
    assert fake_wandb.log_calls[-1]["payload"] == {"geo/client_locations": emitter._geo_table}
    assert emitter._run.finish_calls == 1


def test_wandb_client_update_dropout_comm_failure_and_checkpoint(fake_wandb):
    emitter = WandbEmitter()
    emitter.on_init("fedviz-run", "FedAvg", {})

    emitter.on_client_update(make_client(round=None))
    assert fake_wandb.log_calls == []

    emitter.on_client_update(make_client(round=6, status="failed"))
    update = fake_wandb.log_calls[0]
    assert update["step"] == 6
    assert update["payload"]["client/client-1/status"] == 3

    emitter.on_dropout(6, "client-1", "timeout")
    emitter.on_comm_failure(6, "client-1", None)
    assert fake_wandb.alert_calls == [
        {
            "title": "Client dropout: client-1",
            "text": "Round 6 — timeout",
            "level": "warn",
        },
        {
            "title": "Comm failure: client-1",
            "text": "Round 6 — unknown",
            "level": "error",
        },
    ]

    emitter.on_checkpoint(6, "/tmp/model.bin", {"format": "bin"})
    artifact = emitter._run.logged_artifacts[0]
    assert artifact.name == "model-round-6"
    assert artifact.metadata == {"format": "bin"}
    assert artifact.files == ["/tmp/model.bin"]
    assert fake_wandb.log_calls[-1]["payload"] == {"round": 6, "event/checkpoint": 1}
