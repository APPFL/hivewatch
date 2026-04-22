
from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def new_run_id() -> str:
    prefix = "".join(random.choices(string.ascii_lowercase, k=4))
    suffix = "".join(random.choices(string.digits, k=4))
    return f"run-{prefix}-{suffix}"


@dataclass
class ClientUpdate:
    """
    Required:
        `client_id`   str     unique identifier for the client

    Core distributed/federated training metrics:
        `round`               int     current global round number
        `local_loss`          float   loss after local training
        `local_accuracy`      float   accuracy after local training
        `num_samples`         int     number of local training samples

    Federated-specific metrics:
        `gradient_norm`       float   L2 norm of local gradients
        `gradient_magnitude`  float   mean absolute gradient value
        `sparsity`            float   fraction of near-zero parameters (0-1)
        `compression_ratio`   float   fraction of parameters actually transmitted
        `bytes_sent`          int     bytes uploaded to server (None if not reported)
        `bytes_received`      int     bytes downloaded from server (None if not reported)
        `network_latency_ms`  float   round-trip latency to server
        `train_time_sec`      float   local training wall time
        `base_round`          int     which global round this update is based on
                                    (async FL only — staleness = round - base_round)

    System resources (logged under sys/<client_id>/ in wandb):
        `cpu_pct`       float   CPU utilization %
        `ram_mb`        float   peak RSS memory in MB
        `gpu_util_pct`  float   GPU utilization %
        `gpu_vram_mb`   float   GPU VRAM used in MB

    Geo (for client map visualization):
        `lat`     float   latitude
        `lng`     float   longitude
        `city`    str     city name
        `country` str     country code e.g. "US"

    Status:
        `status`  str     "active" | "idle" | "dropped" | "failed"
    """

    # Required
    client_id: str

    # Core FL metrics
    round:          Optional[int]   = None
    local_accuracy: Optional[float] = None
    local_loss:     Optional[float] = None
    num_samples:    int             = 0

    # Async FL staleness
    base_round: Optional[int] = None   # staleness = round - base_round

    # Gradient forensics
    gradient_norm:      Optional[float] = None
    gradient_magnitude: Optional[float] = None
    sparsity:           Optional[float] = None
    compression_ratio:  Optional[float] = None

    # Communication
    bytes_sent:         Optional[int]  = None
    bytes_received:     Optional[int]  = None
    network_latency_ms: Optional[float] = None
    train_time_sec:     Optional[float] = None

    # System resources
    cpu_pct:     Optional[float] = None
    ram_mb:      Optional[float] = None
    gpu_util_pct:Optional[float] = None
    gpu_vram_mb: Optional[float] = None

    # Geo
    lat:     Optional[float] = None
    lng:     Optional[float] = None
    city:    Optional[str]   = None
    country: Optional[str]   = None

    # Health
    status: str = "active"

    # Any extra fields from your framework — nothing is dropped
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def staleness(self) -> Optional[int]:
        if self.round is not None and self.base_round is not None:
            return self.round - self.base_round
        return None

    def to_dict(self) -> dict:
        return {
            "client_id":         self.client_id,
            "round":             self.round,
            "local_accuracy":    self.local_accuracy,
            "local_loss":        self.local_loss,
            "num_samples":       self.num_samples,
            "base_round":        self.base_round,
            "gradient_norm":     self.gradient_norm,
            "gradient_magnitude":self.gradient_magnitude,
            "sparsity":          self.sparsity,
            "compression_ratio": self.compression_ratio,
            "bytes_sent":        self.bytes_sent,
            "bytes_received":    self.bytes_received,
            "network_latency_ms":self.network_latency_ms,
            "train_time_sec":    self.train_time_sec,
            "cpu_pct":           self.cpu_pct,
            "ram_mb":            self.ram_mb,
            "gpu_util_pct":      self.gpu_util_pct,
            "gpu_vram_mb":       self.gpu_vram_mb,
            "lat":               self.lat,
            "lng":               self.lng,
            "city":              self.city,
            "country":           self.country,
            "status":            self.status,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClientUpdate":
        """
        Build a ClientUpdate from any dict.
        Known keys are mapped; everything else goes into extra.
        This is the bridge between 'whatever your framework returns'
        and 'what hivewatch understands'.
        """
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in known and k != "extra"}
        extra  = {k: v for k, v in d.items() if k not in known}
        return cls(**kwargs, extra=extra)


@dataclass
class RoundSummary:
    """
    Server-side summary for one completed round.
    Produced by the user after aggregation, passed to hivewatch.log_round().

    Required:
        `round`           int

    Optional:
        `global_accuracy` float
        `global_loss`     float
        `num_selected`        int     clients invited this round
        `num_completed`       int     clients that finished
        `num_stragglers`      int     clients that timed out
        `num_failures`        int     clients that errored
        `total_bytes_up`      int     total bytes uploaded across all clients
        `total_bytes_down`    int     total bytes downloaded across all clients
        `round_duration_sec`  float   wall time for the round
        `gradient_divergence` float   std dev of per-client gradient norms
                                    — high value = non-IID data distribution
        `aggregation_time_sec` float  pure aggregation compute time
        `algorithm_metadata`  dict    algorithm-specific params
                                    e.g. {"mu": 0.01} for FedProx
    """
    round:           int
    global_accuracy: Optional[float] = None
    global_loss:     Optional[float] = None

    num_selected:         int            = 0
    num_completed:        int            = 0
    num_stragglers:       int            = 0
    num_failures:         int            = 0
    total_bytes_up:       int            = 0
    total_bytes_down:     int            = 0
    round_duration_sec:   Optional[float] = None
    gradient_divergence:  Optional[float] = None
    aggregation_time_sec: Optional[float] = None
    algorithm_metadata:   Dict[str, Any]  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "round":                self.round,
            "global_accuracy":      self.global_accuracy,
            "global_loss":          self.global_loss,
            "num_selected":         self.num_selected,
            "num_completed":        self.num_completed,
            "num_stragglers":       self.num_stragglers,
            "num_failures":         self.num_failures,
            "total_bytes_up":       self.total_bytes_up,
            "total_bytes_down":     self.total_bytes_down,
            "round_duration_sec":   self.round_duration_sec,
            "gradient_divergence":  self.gradient_divergence,
            "aggregation_time_sec": self.aggregation_time_sec,
            "algorithm_metadata":   self.algorithm_metadata,
        }
