from __future__ import annotations

import time
import logging
import statistics
from typing import Dict, List, Optional

from . import _state
from .schema import ClientUpdate, RoundSummary, new_run_id

logger = logging.getLogger("fedviz")


class FedVizRun:
    """
    A single federated learning observability run.

    Tracks per-client updates and round summaries, auto-computes derived
    metrics (gradient divergence, byte totals, round wall time), and fans
    events out to one or more emitter backends.

    Prefer constructing via ``fedviz.init()`` rather than directly, which
    also registers the instance as the global singleton used by the
    module-level convenience functions.

    Can be used as a context manager — ``finish()`` is called automatically
    on exit::

        with fedviz.init(...) as run:
            for round in range(num_rounds):
                run.round_start(round)
                run.log_client_update(client_id, **metadata)
                run.log_round(round, global_accuracy=acc, global_loss=loss)
    """

    def __init__(
        self,
        run_id:    str,
        algorithm: str,
        config:    dict,
        emitters:  list,
        verbose:   bool,
    ):
        self.run_id    = run_id
        self.algorithm = algorithm
        self.config    = config
        self.emitters  = emitters
        self.verbose   = verbose

        self._round_start_times: Dict[int, float] = {}
        self._pending_clients:   Dict[int, List[ClientUpdate]] = {}

        for e in self.emitters:
            if hasattr(e, "on_init"):
                e.on_init(run_id, algorithm, config)

    # ── Public API ────────────────────────────────────────────────────────────

    def round_start(self, round: int):
        self._round_start_times[round] = time.time()
        self._pending_clients[round]   = []

    def log_client_update(self, client_id: str, **kwargs):
        client    = ClientUpdate.from_dict({"client_id": client_id, **kwargs})
        round_num = client.round

        if round_num is not None:
            self._pending_clients.setdefault(round_num, []).append(client)

        for e in self.emitters:
            if hasattr(e, "on_client_update"):
                try:
                    e.on_client_update(client)
                except Exception as ex:
                    logger.warning(f"[fedviz] emitter {type(e).__name__}.on_client_update failed: {ex}")

    def log_round(
        self,
        round:                int,
        *,
        global_accuracy:      Optional[float] = None,
        global_loss:          Optional[float] = None,
        num_selected:         Optional[int]   = None,
        num_stragglers:       int             = 0,
        num_failures:         int             = 0,
        total_bytes_up:       int             = 0,
        total_bytes_down:     int             = 0,
        gradient_divergence:  Optional[float] = None,
        aggregation_time_sec: Optional[float] = None,
        algorithm_metadata:   Optional[dict]  = None,
    ):
        clients = self._pending_clients.pop(round, [])

        if total_bytes_up == 0:
            reported = [c.bytes_sent for c in clients if c.bytes_sent is not None]
            if reported:
                total_bytes_up = sum(reported)
        if total_bytes_down == 0:
            reported = [c.bytes_received for c in clients if c.bytes_received is not None]
            if reported:
                total_bytes_down = sum(reported)

        if gradient_divergence is None:
            norms = [c.gradient_norm for c in clients if c.gradient_norm is not None]
            if len(norms) > 1:
                gradient_divergence = statistics.stdev(norms)

        round_duration = None
        if round in self._round_start_times:
            round_duration = time.time() - self._round_start_times.pop(round)

        num_completed = len([c for c in clients if c.status == "active"])

        summary = RoundSummary(
            round                = round,
            global_accuracy      = global_accuracy,
            global_loss          = global_loss,
            num_selected         = num_selected if num_selected is not None else len(clients),
            num_completed        = num_completed,
            num_stragglers       = num_stragglers,
            num_failures         = num_failures,
            total_bytes_up       = total_bytes_up,
            total_bytes_down     = total_bytes_down,
            round_duration_sec   = round_duration,
            gradient_divergence  = gradient_divergence,
            aggregation_time_sec = aggregation_time_sec,
            algorithm_metadata   = algorithm_metadata or {},
        )

        for e in self.emitters:
            if hasattr(e, "on_round"):
                try:
                    e.on_round(summary, clients)
                except Exception as ex:
                    logger.warning(f"[fedviz] emitter {type(e).__name__}.on_round failed: {ex}")

        if self.verbose:
            acc_str  = f"{global_accuracy:.4f}" if global_accuracy is not None else "n/a"
            loss_str = f"{global_loss:.4f}"     if global_loss     is not None else "n/a"
            strag    = f"  stragglers={num_stragglers}" if num_stragglers else ""
            print(f"[fedviz] round {round:>3}  acc={acc_str}  loss={loss_str}  clients={len(clients)}{strag}")

    def log_dropout(self, round: int, client_id: str, reason: Optional[str] = None):
        for e in self.emitters:
            if hasattr(e, "on_dropout"):
                try:
                    e.on_dropout(round, client_id, reason)
                except Exception as ex:
                    logger.warning(f"[fedviz] emitter on_dropout failed: {ex}")

    def log_comm_failure(self, round: int, client_id: str, reason: Optional[str] = None):
        for e in self.emitters:
            if hasattr(e, "on_comm_failure"):
                try:
                    e.on_comm_failure(round, client_id, reason)
                except Exception as ex:
                    logger.warning(f"[fedviz] emitter on_comm_failure failed: {ex}")

    def log_checkpoint(self, round: int, path: str, **metadata):
        for e in self.emitters:
            if hasattr(e, "on_checkpoint"):
                try:
                    e.on_checkpoint(round, path, metadata)
                except Exception as ex:
                    logger.warning(f"[fedviz] emitter on_checkpoint failed: {ex}")

    def finish(self):
        for e in self.emitters:
            if hasattr(e, "finish"):
                try:
                    e.finish()
                except Exception as ex:
                    logger.warning(f"[fedviz] emitter finish failed: {ex}")
        if self.verbose:
            print(f"[fedviz] run {self.run_id} finished ✓")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.finish()


def init(
    *,
    run_id:    Optional[str]  = None,
    algorithm: str            = "FedAvg",
    config:    Optional[dict] = None,
    emitters:  Optional[list] = None,
    verbose:   bool           = True,
) -> FedVizRun:
    """
    Initialize a `fedviz` run. Call once via `with fedviz.init()` before your training loop.

    ``init()`` is now fully backend-agnostic. Construct emitter instances
    explicitly and pass them in. This keeps `init()` stable as new backends
    are added — it never needs to change.

    Examples
    --------
    
    **Weights & Biases:**

    ```python
    from fedviz.emitters import WandbEmitter

    fedviz.init(
        algorithm = "FedAvg",
        emitters  = [WandbEmitter(project="my-fl-project")],
    )
    ```

    **MLflow:**

    ```python
    from fedviz.emitters import MLflowEmitter

    fedviz.init(
        algorithm = "FedAvg",
        emitters  = [MLflowEmitter(tracking_uri="http://localhost:5000", experiment="fedviz")],
    )
    ```

    **Both at once:**

    ```python
    from fedviz.emitters import WandbEmitter, MLflowEmitter

    fedviz.init(
        algorithm = "FedAvg",
        emitters  = [
            WandbEmitter(project="my-fl-project"),
            MLflowEmitter(tracking_uri="http://localhost:5000", experiment="fedviz"),
        ],
    )
    ```

    **Auto-adopt an existing wandb run:**

    ```python
    import wandb
    from fedviz.emitters import WandbEmitter

    wandb.init(project="my-project")
    fedviz.init(emitters=[WandbEmitter()])  # adopts the active run
    ```
    """
    _state._run = FedVizRun(
        run_id    = run_id or new_run_id(),
        algorithm = algorithm,
        config    = config or {},
        emitters  = list(emitters or []),
        verbose   = verbose,
    )

    if verbose and _state._run.emitters:
        names = ", ".join(type(e).__name__ for e in _state._run.emitters)
        print(f"[fedviz] run={_state._run.run_id}  algorithm={algorithm}  emitters=[{names}]")
    elif verbose:
        print(f"[fedviz] run={_state._run.run_id}  algorithm={algorithm}  no emitters")

    return _state._run