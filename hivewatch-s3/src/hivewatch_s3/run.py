from __future__ import annotations

import logging
import statistics
import threading
import time
from typing import Dict, List, Optional

from .schema import ClientUpdate, RoundSummary, new_run_id

logger = logging.getLogger("hivewatch_s3")


class HiveWatchRun:
    def __init__(
        self,
        *,
        run_id: str,
        algorithm: str,
        config: dict,
        emitters: list,
        verbose: bool,
    ):
        self.run_id = run_id
        self.algorithm = algorithm
        self.config = config
        self.emitters = emitters
        self.verbose = verbose
        self._access_lock = threading.Lock()
        self._round_start_times: Dict[int, float] = {}
        self._pending_clients: Dict[int, List[ClientUpdate]] = {}

        for emitter in self.emitters:
            if hasattr(emitter, "on_init"):
                emitter.on_init(run_id, algorithm, config)

    def round_start(self, round_num: int):
        self._round_start_times[round_num] = time.time()
        self._pending_clients[round_num] = []

    def log_client_update(self, client_id: str, **kwargs):
        with self._access_lock:
            client = ClientUpdate.from_dict({"client_id": client_id, **kwargs})
            if client.round is not None:
                self._pending_clients.setdefault(client.round, []).append(client)
            for emitter in self.emitters:
                if hasattr(emitter, "on_client_update"):
                    try:
                        emitter.on_client_update(client)
                    except Exception as exc:
                        logger.warning("[hivewatch-s3] emitter client update failed: %s", exc)

    def log_round(
        self,
        round_num: int,
        *,
        global_accuracy: Optional[float] = None,
        global_loss: Optional[float] = None,
        num_selected: Optional[int] = None,
        num_stragglers: int = 0,
        num_failures: int = 0,
        total_bytes_up: int = 0,
        total_bytes_down: int = 0,
        gradient_divergence: Optional[float] = None,
        aggregation_time_sec: Optional[float] = None,
        algorithm_metadata: Optional[dict] = None,
    ):
        clients = self._pending_clients.pop(round_num, [])

        if total_bytes_up == 0:
            reported = [client.bytes_sent for client in clients if client.bytes_sent is not None]
            if reported:
                total_bytes_up = sum(reported)
        if total_bytes_down == 0:
            reported = [client.bytes_received for client in clients if client.bytes_received is not None]
            if reported:
                total_bytes_down = sum(reported)
        if gradient_divergence is None:
            norms = [client.gradient_norm for client in clients if client.gradient_norm is not None]
            if len(norms) > 1:
                gradient_divergence = statistics.stdev(norms)

        round_duration = None
        if round_num in self._round_start_times:
            round_duration = time.time() - self._round_start_times.pop(round_num)

        summary = RoundSummary(
            round=round_num,
            global_accuracy=global_accuracy,
            global_loss=global_loss,
            num_selected=num_selected if num_selected is not None else len(clients),
            num_completed=len([client for client in clients if client.status == "active"]),
            num_stragglers=num_stragglers,
            num_failures=num_failures,
            total_bytes_up=total_bytes_up,
            total_bytes_down=total_bytes_down,
            round_duration_sec=round_duration,
            gradient_divergence=gradient_divergence,
            aggregation_time_sec=aggregation_time_sec,
            algorithm_metadata=algorithm_metadata or {},
        )

        for emitter in self.emitters:
            if hasattr(emitter, "on_round"):
                try:
                    emitter.on_round(summary, clients)
                except Exception as exc:
                    logger.warning("[hivewatch-s3] emitter round failed: %s", exc)

    def log_dropout(self, round_num: int, client_id: str, reason: Optional[str] = None):
        for emitter in self.emitters:
            if hasattr(emitter, "on_dropout"):
                emitter.on_dropout(round_num, client_id, reason)

    def log_comm_failure(self, round_num: int, client_id: str, reason: Optional[str] = None):
        for emitter in self.emitters:
            if hasattr(emitter, "on_comm_failure"):
                emitter.on_comm_failure(round_num, client_id, reason)

    def log_checkpoint(self, round_num: int, path: str, **metadata):
        for emitter in self.emitters:
            if hasattr(emitter, "on_checkpoint"):
                emitter.on_checkpoint(round_num, path, metadata)

    def set_server_metadata(self, **metadata):
        for emitter in self.emitters:
            if hasattr(emitter, "on_server_metadata"):
                emitter.on_server_metadata(metadata)

    def finish(self):
        for emitter in self.emitters:
            if hasattr(emitter, "finish"):
                emitter.finish()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.finish()


def init(
    *,
    run_id: Optional[str] = None,
    algorithm: str = "FedAvg",
    config: Optional[dict] = None,
    emitters: Optional[list] = None,
    verbose: bool = True,
) -> HiveWatchRun:
    run = HiveWatchRun(
        run_id=run_id or new_run_id(),
        algorithm=algorithm,
        config=config or {},
        emitters=list(emitters or []),
        verbose=verbose,
    )
    if verbose:
        emitter_names = ", ".join(type(emitter).__name__ for emitter in run.emitters) or "no emitters"
        print(f"[hivewatch-s3] run={run.run_id} algorithm={algorithm} emitters=[{emitter_names}]")
    return run
