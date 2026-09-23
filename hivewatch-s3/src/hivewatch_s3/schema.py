from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def new_run_id() -> str:
    prefix = "".join(random.choices(string.ascii_lowercase, k=4))
    suffix = "".join(random.choices(string.digits, k=4))
    return f"run-{prefix}-{suffix}"


@dataclass
class ClientUpdate:
    client_id: str
    round: Optional[int] = None
    local_accuracy: Optional[float] = None
    local_loss: Optional[float] = None
    num_samples: int = 0
    base_round: Optional[int] = None
    gradient_norm: Optional[float] = None
    gradient_magnitude: Optional[float] = None
    sparsity: Optional[float] = None
    compression_ratio: Optional[float] = None
    bytes_sent: Optional[int] = None
    bytes_received: Optional[int] = None
    network_latency_ms: Optional[float] = None
    train_time_sec: Optional[float] = None
    cpu_pct: Optional[float] = None
    ram_mb: Optional[float] = None
    gpu_util_pct: Optional[float] = None
    gpu_vram_mb: Optional[float] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    city: Optional[str] = None
    country: Optional[str] = None
    status: str = "active"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "round": self.round,
            "local_accuracy": self.local_accuracy,
            "local_loss": self.local_loss,
            "num_samples": self.num_samples,
            "base_round": self.base_round,
            "gradient_norm": self.gradient_norm,
            "gradient_magnitude": self.gradient_magnitude,
            "sparsity": self.sparsity,
            "compression_ratio": self.compression_ratio,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "network_latency_ms": self.network_latency_ms,
            "train_time_sec": self.train_time_sec,
            "cpu_pct": self.cpu_pct,
            "ram_mb": self.ram_mb,
            "gpu_util_pct": self.gpu_util_pct,
            "gpu_vram_mb": self.gpu_vram_mb,
            "lat": self.lat,
            "lng": self.lng,
            "city": self.city,
            "country": self.country,
            "status": self.status,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "ClientUpdate":
        known = {field_name for field_name in cls.__dataclass_fields__}
        kwargs = {key: value for key, value in raw.items() if key in known and key != "extra"}
        extra = {key: value for key, value in raw.items() if key not in known}
        return cls(**kwargs, extra=extra)


@dataclass
class RoundSummary:
    round: int
    global_accuracy: Optional[float] = None
    global_loss: Optional[float] = None
    num_selected: int = 0
    num_completed: int = 0
    num_stragglers: int = 0
    num_failures: int = 0
    total_bytes_up: int = 0
    total_bytes_down: int = 0
    round_duration_sec: Optional[float] = None
    gradient_divergence: Optional[float] = None
    aggregation_time_sec: Optional[float] = None
    algorithm_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "global_accuracy": self.global_accuracy,
            "global_loss": self.global_loss,
            "num_selected": self.num_selected,
            "num_completed": self.num_completed,
            "num_stragglers": self.num_stragglers,
            "num_failures": self.num_failures,
            "total_bytes_up": self.total_bytes_up,
            "total_bytes_down": self.total_bytes_down,
            "round_duration_sec": self.round_duration_sec,
            "gradient_divergence": self.gradient_divergence,
            "aggregation_time_sec": self.aggregation_time_sec,
            "algorithm_metadata": self.algorithm_metadata,
        }
