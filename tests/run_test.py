from __future__ import annotations

import math

from hivewatch.run import HivewatchRun


class RecorderEmitter:
    def __init__(self):
        self.inits: list[tuple] = []
        self.client_updates: list = []
        self.rounds: list[tuple] = []
        self.finished = 0

    def on_init(self, run_id: str, algorithm: str, config: dict):
        self.inits.append((run_id, algorithm, config))

    def on_client_update(self, client):
        self.client_updates.append(client)

    def on_round(self, summary, clients):
        self.rounds.append((summary, clients))

    def finish(self):
        self.finished += 1


def test_hivewatch_run_derives_round_metrics_from_client_updates():
    emitter = RecorderEmitter()
    run = HivewatchRun(
        run_id="run-1234",
        algorithm="FedAvg",
        config={"epochs": 2},
        emitters=[emitter],
        verbose=False,
    )

    run.round_start(2)
    run.log_client_update(
        "client-1",
        round=2,
        gradient_norm=1.0,
        bytes_sent=100,
        bytes_received=200,
        status="active",
    )
    run.log_client_update(
        "client-2",
        round=2,
        gradient_norm=3.0,
        bytes_sent=400,
        bytes_received=500,
        status="failed",
    )
    run.log_round(2, global_accuracy=0.8, global_loss=0.4)

    assert emitter.inits == [("run-1234", "FedAvg", {"epochs": 2})]
    assert len(emitter.client_updates) == 2
    summary, clients = emitter.rounds[0]
    assert len(clients) == 2
    assert summary.num_selected == 2
    assert summary.num_completed == 1
    assert summary.total_bytes_up == 500
    assert summary.total_bytes_down == 700
    assert math.isclose(summary.gradient_divergence, math.sqrt(2.0))
    assert summary.round_duration_sec is not None

    run.finish()
    assert emitter.finished == 1
