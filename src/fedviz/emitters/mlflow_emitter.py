from __future__ import annotations

import os
import time
import logging
from typing import Dict, List, Optional
from fedviz import ClientUpdate, RoundSummary

logger = logging.getLogger("fedviz.emitters.mlflow")


class MLflowEmitter:
    """
    Emitter that logs `fedviz` events to MLflow Tracking.

    **Usage**:

        import fedviz
        from fedviz.emitters import MLflowEmitter

        fedviz.init(
            algorithm = "FedAvg",
            emitters  = [
                MLflowEmitter(
                    tracking_uri = "http://localhost:5000",
                    experiment = "my-fl-experiment",
                    mlflow_system_metrics = True,  # enable MLflow native server-side system metrics
                    system_metrics_sampling_interval = 10,  # sample every 10 s
                )
            ],
        )
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None, # None = local ./mlruns
        experiment: str = "fedviz",
        run_name: Optional[str] = None, # auto-set to run_id on init if None
        tags: Optional[Dict[str, str]] = None,
        log_system: bool = True,
        log_per_client: bool = True,
        mlflow_system_metrics: bool = False, # enable MLflow native server-side system metrics
        system_metrics_sampling_interval: Optional[int] = None,  # seconds; None = MLflow default
    ):
        try:
            import mlflow
            self._mlflow = mlflow
        except ImportError:
            raise ImportError("mlflow not installed. Run: pip install mlflow")

        self.tracking_uri   = tracking_uri
        self.experiment     = experiment
        self.run_name       = run_name
        self.tags           = tags or {}
        self.log_system                       = log_system
        self.log_per_client                   = log_per_client
        self.mlflow_system_metrics            = mlflow_system_metrics
        self.system_metrics_sampling_interval = system_metrics_sampling_interval
        self._run = None
        self._run_id: Optional[str] = None
        self._client = None   # MlflowClient — set in on_init, used for thread-safe logging

    # ── Called by FedVizRun ───────────────────────────────────────────────────

    def on_init(self, run_id: str, algorithm: str, config: dict):
        if self._run is not None:
            return

        if self.tracking_uri:
            self._mlflow.set_tracking_uri(self.tracking_uri)

        if self.mlflow_system_metrics:
            self._mlflow.enable_system_metrics_logging()
            if self.system_metrics_sampling_interval is not None:
                os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = str(
                    self.system_metrics_sampling_interval
                )

        self._mlflow.set_experiment(self.experiment)

        self._run = self._mlflow.start_run(
            run_name = self.run_name or run_id,
            tags     = {
                "fedviz/run_id":   run_id,
                "fedviz/algorithm": algorithm,
                **self.tags,
            },
        )

        self._run_id = self._run.info.run_id
        self._client = self._mlflow.tracking.MlflowClient()

        # Log hyperparameters as MLflow params (logged once, not per step)
        flat_config = self._flatten(config)
        if flat_config:
            self._mlflow.log_params(flat_config)

        logger.info(f"[fedviz/mlflow] run started: {self._run_id}")

    def on_round(self, summary: RoundSummary, clients: List[ClientUpdate]):
        step = summary.round
        metrics: Dict[str, float] = {}

        # Global
        if summary.global_accuracy is not None:
            metrics["round/accuracy"] = summary.global_accuracy
        if summary.global_loss is not None:
            metrics["round/loss"] = summary.global_loss
        if summary.num_selected > 0:
            metrics["round/participation_rate"] = summary.num_completed / summary.num_selected
        metrics["round/num_stragglers"] = summary.num_stragglers
        metrics["round/num_failures"]   = summary.num_failures
        if summary.round_duration_sec is not None:
            metrics["round/duration_sec"] = summary.round_duration_sec

        # Communication
        up_mb   = summary.total_bytes_up   / 1e6
        down_mb = summary.total_bytes_down / 1e6
        metrics["comm/total_bytes_up_mb"]   = up_mb
        metrics["comm/total_bytes_down_mb"] = down_mb
        metrics["comm/total_bytes_mb"]      = up_mb + down_mb
        if summary.num_completed > 0:
            metrics["comm/bytes_per_client_mb"] = up_mb / summary.num_completed

        # Aggregation
        if summary.gradient_divergence is not None:
            metrics["agg/gradient_divergence"] = summary.gradient_divergence
        if summary.aggregation_time_sec is not None:
            metrics["agg/aggregation_time_sec"] = summary.aggregation_time_sec
        for k, v in summary.algorithm_metadata.items():
            if isinstance(v, (int, float)):
                metrics[f"agg/algo/{k}"] = float(v)

        # Per-client
        for c in clients:
            if self.log_per_client:
                metrics.update(self._client_metrics(c))
            if self.log_system:
                metrics.update(self._sys_metrics(c))

        self._log_metrics(metrics, step=step)

    def on_client_update(self, client: ClientUpdate):
        """Async FL: log a single client update immediately."""
        if client.round is None:
            return
        metrics: Dict[str, float] = {}
        if self.log_per_client:
            metrics.update(self._client_metrics(client))
        if self.log_system:
            metrics.update(self._sys_metrics(client))
        if metrics:
            self._log_metrics(metrics, step=client.round)

    def on_dropout(self, round: int, client_id: str, reason: Optional[str]):
        self._log_metrics({"event/client_dropout": 1}, step=round)
        self._set_tag(f"dropout.round_{round}.{client_id}", reason or "unknown")

    def on_comm_failure(self, round: int, client_id: str, reason: Optional[str]):
        self._log_metrics({"event/comm_failure": 1}, step=round)
        self._set_tag(f"comm_failure.round_{round}.{client_id}", reason or "unknown")

    def on_checkpoint(self, round: int, path: str, metadata: dict):
        """Log model checkpoint as an MLflow artifact."""
        try:
            self._client.log_artifact(self._run_id, path, artifact_path=f"checkpoints/round_{round}")
            self._log_metrics({"event/checkpoint": 1}, step=round)
            logger.info(f"[fedviz/mlflow] checkpoint logged: {path}")
        except Exception as e:
            logger.warning(f"[fedviz/mlflow] artifact upload failed: {e}")

    def finish(self):
        if self._run is not None:
            self._mlflow.end_run()
            self._run    = None
            self._run_id = None
            self._client = None

    # ── Thread-safe logging helpers (use MlflowClient, not fluent API) ────────
    # The fluent API (mlflow.log_metrics etc.) resolves the active run from a
    # thread-local stack. If a new thread is spawned (e.g. a grpc worker thread) 
    # and the fluent API is called there, MLflow will auto-create a new run in 
    # that thread instead of using the existing run created in the main thread.
    # By contrast, MlflowClient with an explicit run_id works across threads and 
    # always targets the correct run.

    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if not metrics or self._run_id is None:
            return
        ts = int(time.time() * 1000)
        self._client.log_batch(
            run_id  = self._run_id,
            metrics = [
                self._mlflow.entities.Metric(k, v, ts, step)
                for k, v in metrics.items()
            ],
        )

    def _set_tag(self, key: str, value: str) -> None:
        if self._run_id is None:
            return
        self._client.set_tag(self._run_id, key, value)

    # ── Metric builders ───────────────────────────────────────────────────────

    # MLflow key constraint: no "/" in metric names for per-client metrics
    # (MLflow UI groups by "." separators in the metric name)
    # Round-level metrics use "/" which MLflow handles fine.

    def _client_metrics(self, c: ClientUpdate) -> Dict[str, float]:
        # Use dot notation for per-client: client.<id>.<metric>
        p = f"client.{c.client_id}."
        d: Dict[str, float] = {}

        def s(k, v):
            if v is not None:
                d[p + k] = float(v)

        s("accuracy",           c.local_accuracy)
        s("loss",               c.local_loss)
        s("num_samples",        c.num_samples or None)
        s("gradient_norm",      c.gradient_norm)
        s("gradient_magnitude", c.gradient_magnitude)
        s("sparsity",           c.sparsity)
        s("compression_ratio",  c.compression_ratio)
        s("network_latency_ms", c.network_latency_ms)
        s("train_time_sec",     c.train_time_sec)
        s("staleness",          c.staleness)
        if c.bytes_sent:     d[p + "bytes_sent_mb"]     = c.bytes_sent / 1e6
        if c.bytes_received: d[p + "bytes_received_mb"] = c.bytes_received / 1e6

        _status = {"active": 0, "idle": 1, "dropped": 2, "failed": 3}
        d[p + "status"] = float(_status.get(c.status, 0))
        return d

    def _sys_metrics(self, c: ClientUpdate) -> Dict[str, float]:
        p = f"sys.{c.client_id}."
        d: Dict[str, float] = {}

        def s(k, v):
            if v is not None:
                d[p + k] = float(v)

        s("cpu_pct",      c.cpu_pct)
        s("ram_mb",       c.ram_mb)
        s("gpu_util_pct", c.gpu_util_pct)
        s("gpu_vram_mb",  c.gpu_vram_mb)
        return d

    @staticmethod
    def _flatten(d: dict, prefix: str = "") -> Dict[str, str]:
        """Flatten nested config dict for mlflow.log_params (values must be str)."""
        out: Dict[str, str] = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(MLflowEmitter._flatten(v, prefix=key))
            else:
                out[key] = str(v)
        return out