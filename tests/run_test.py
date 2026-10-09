from __future__ import annotations

import math
import threading
import time

import pytest

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
    assert math.isclose(summary.gradient_norm_dispersion, math.sqrt(2.0))
    assert summary.round_duration_sec is not None

    run.finish()
    assert emitter.finished == 1


def test_log_round_accepts_deprecated_gradient_divergence_kwarg():
    emitter = RecorderEmitter()
    run = HivewatchRun(
        run_id="run-5678",
        algorithm="FedAvg",
        config={},
        emitters=[emitter],
        verbose=False,
    )

    run.round_start(1)
    with pytest.warns(DeprecationWarning, match="gradient_norm_dispersion"):
        run.log_round(1, gradient_divergence=0.25)

    summary, _ = emitter.rounds[0]
    assert summary.gradient_norm_dispersion == 0.25
    with pytest.warns(DeprecationWarning, match="gradient_norm_dispersion"):
        assert summary.gradient_divergence == 0.25


def test_concurrent_logging_records_every_client_update():
    inner = RecorderEmitter()
    run = HivewatchRun(run_id="run-conc", algorithm="FedAvg", config={}, emitters=[inner], verbose=False)
    run.round_start(0)

    def worker(tid: int):
        for i in range(200):
            run.log_client_update(f"client-{tid}-{i}", round=0)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    captured = {}

    class SummaryEmitter:
        def on_round(self, summary, clients):
            captured["n"] = len(clients)

    run.emitters.append(SummaryEmitter())
    run.log_round(0)

    assert captured["n"] == 1600
    assert len(inner.client_updates) == 1600


def test_round_and_client_dispatch_never_overlap_across_threads():
    class OverlapDetector:
        def __init__(self):
            self.active = 0
            self.overlaps = 0
            self.guard = threading.Lock()

        def _enter(self):
            with self.guard:
                self.active += 1
                if self.active > 1:
                    self.overlaps += 1
            time.sleep(0.0005)
            with self.guard:
                self.active -= 1

        def on_client_update(self, client):
            self._enter()

        def on_round(self, summary, clients):
            self._enter()

    detector = OverlapDetector()
    run = HivewatchRun(run_id="run-overlap", algorithm="FedAvg", config={}, emitters=[detector], verbose=False)

    def log_clients():
        for i in range(200):
            run.log_client_update(f"client-{i}", round=i % 20)

    def log_rounds():
        for r in range(20):
            run.round_start(r)
            run.log_round(r)

    threads = [threading.Thread(target=log_clients), threading.Thread(target=log_rounds)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert detector.overlaps == 0
