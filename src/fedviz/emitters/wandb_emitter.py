
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..schema import ClientUpdate, RoundSummary

logger = logging.getLogger("fedviz.emitters.wandb")

_STATUS_INT = {"active": 0, "idle": 1, "dropped": 2, "failed": 3}


class WandbEmitter:
    def __init__(
        self,
        project:        str  = "fedviz",
        entity:         Optional[str] = None,
        group:          Optional[str] = None,
        job_type:       str  = "federated-training",
        tags:           Optional[List[str]] = None,
        config:         Optional[Dict[str, Any]] = None,
        log_system:     bool = True,
        log_per_client: bool = True,
        log_geo:        bool = False,  
        mode:           str  = "online",
    ):
        try:
            import wandb
            self._w = wandb
        except ImportError:
            raise ImportError("wandb not installed. Run: pip install wandb")

        self.project        = project
        self.entity         = entity
        self.group          = group
        self.job_type       = job_type
        self.tags           = tags or []
        self.extra_config   = config or {}
        self.log_system     = log_system
        self.log_per_client = log_per_client
        self.log_geo        = log_geo
        self.mode           = mode

        self._run             = None
        self._metrics_defined = False
        self._geo_table       = None

    # ── Called by FedVizRun ───────────────────────────────────────────────────

    def on_init(self, run_id: str, algorithm: str, config: dict):
        if self._run is not None:
            return
        # Adopt existing wandb run if user already called wandb.init()
        if self._w.run is not None:
            self._run = self._w.run
            logger.info("[fedviz/wandb] adopted existing wandb run")
        else:
            self._run = self._w.init(
                project  = self.project,
                entity   = self.entity,
                group    = self.group or algorithm,
                job_type = self.job_type,
                name     = run_id,
                tags     = self.tags,
                config   = {"fedviz/run_id": run_id, "fedviz/algorithm": algorithm, **config, **self.extra_config},
                mode     = self.mode,
            )
        self._define_metrics()

    def on_round(self, summary: RoundSummary, clients: List[ClientUpdate]):
        log: Dict[str, Any] = {"round": summary.round}

        # Global metrics
        log["round/accuracy"] = summary.global_accuracy
        log["round/loss"]     = summary.global_loss
        if summary.num_selected > 0:
            log["round/participation_rate"] = summary.num_completed / summary.num_selected
        log["round/num_stragglers"] = summary.num_stragglers
        log["round/num_failures"]   = summary.num_failures
        if summary.round_duration_sec is not None:
            log["round/duration_sec"] = summary.round_duration_sec

        # Communication
        up_mb   = summary.total_bytes_up   / 1e6
        down_mb = summary.total_bytes_down / 1e6
        log["comm/total_bytes_up_mb"]   = up_mb
        log["comm/total_bytes_down_mb"] = down_mb
        log["comm/total_bytes_mb"]      = up_mb + down_mb
        if summary.num_completed > 0:
            log["comm/bytes_per_client_mb"] = up_mb / summary.num_completed

        # Aggregation health
        if summary.gradient_divergence is not None:
            log["agg/gradient_divergence"] = summary.gradient_divergence
        if summary.aggregation_time_sec is not None:
            log["agg/aggregation_time_sec"] = summary.aggregation_time_sec
        for k, v in summary.algorithm_metadata.items():
            log[f"agg/algo/{k}"] = v

        # Per-client
        for c in clients:
            if self.log_per_client:
                log.update(self._client_metrics(c))
            if self.log_system:
                log.update(self._sys_metrics(c))
            if self.log_geo and c.lat is not None:
                self._record_geo(c, summary.round)

        self._w.log(log, step=summary.round)

    def on_client_update(self, client: ClientUpdate):
        """Async FL: log a single client update immediately."""
        if client.round is None:
            return
        log = {"round": client.round}
        if self.log_per_client:
            log.update(self._client_metrics(client))
        if self.log_system:
            log.update(self._sys_metrics(client))
        self._w.log(log, step=client.round)

    def on_dropout(self, round: int, client_id: str, reason: Optional[str]):
        self._w.log({"round": round, "event/client_dropout": 1}, step=round)
        self._w.alert(
            title=f"Client dropout: {client_id}",
            text=f"Round {round} — {reason or 'unknown'}",
            level=self._w.AlertLevel.WARN,
        )

    def on_comm_failure(self, round: int, client_id: str, reason: Optional[str]):
        self._w.log({"round": round, "event/comm_failure": 1}, step=round)
        self._w.alert(
            title=f"Comm failure: {client_id}",
            text=f"Round {round} — {reason or 'unknown'}",
            level=self._w.AlertLevel.ERROR,
        )

    def on_checkpoint(self, round: int, path: str, metadata: dict):
        artifact = self._w.Artifact(
            name=f"model-round-{round}", type="model", metadata=metadata
        )
        try:
            artifact.add_file(path)
            self._run.log_artifact(artifact)
        except Exception as e:
            logger.warning(f"[fedviz/wandb] artifact upload failed: {e}")
        self._w.log({"round": round, "event/checkpoint": 1}, step=round)

    def finish(self):
        if self._geo_table is not None:
            self._w.log({"geo/client_locations": self._geo_table})
        if self._run:
            self._run.finish()

    # ── Metric builders ───────────────────────────────────────────────────────

    def _client_metrics(self, c: ClientUpdate) -> dict:
        p = f"client/{c.client_id}/"
        d = {}
        def s(k, v):
            if v is not None: d[p + k] = v

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
        d[p + "status"] = _STATUS_INT.get(c.status, 0)
        return d

    def _sys_metrics(self, c: ClientUpdate) -> dict:
        p = f"sys/{c.client_id}/"
        d = {}
        def s(k, v):
            if v is not None: d[p + k] = v
        s("cpu_pct",      c.cpu_pct)
        s("ram_mb",       c.ram_mb)
        s("gpu_util_pct", c.gpu_util_pct)
        s("gpu_vram_mb",  c.gpu_vram_mb)
        return d

    def _record_geo(self, c: ClientUpdate, round_num: int):
        if self._geo_table is None:
            self._geo_table = self._w.Table(
                columns=["round", "client_id", "lat", "lng",
                         "city", "country", "accuracy", "loss", "status"]
            )
        self._geo_table.add_data(
            round_num, c.client_id, c.lat, c.lng,
            c.city or "", c.country or "",
            c.local_accuracy, c.local_loss, c.status,
        )

    def _define_metrics(self):
        if self._metrics_defined:
            return
        self._metrics_defined = True
        self._w.define_metric("round")
        for ns in ["round/*", "comm/*", "agg/*", "client/*", "sys/*", "event/*"]:
            self._w.define_metric(ns, step_metric="round")
