from __future__ import annotations

from typing import Any, Dict, Optional

from .run import init
from .storage import S3ArtifactEmitter


def _to_plain_dict(config_obj) -> dict:
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(config_obj, resolve=True)
    except Exception:
        return {}


def _pick_first(raw: dict, *keys):
    for key in keys:
        value = raw.get(key)
        if value is not None:
            return value
    return None


def normalize_client_metadata(client_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    geo = metadata.get("geo", {}) if isinstance(metadata.get("geo"), dict) else {}
    normalized = {
        "client_id": client_id,
        "round": _pick_first(metadata, "round", "global_round", "epoch"),
        "local_accuracy": _pick_first(metadata, "local_accuracy", "val_accuracy", "accuracy"),
        "local_loss": _pick_first(metadata, "local_loss", "val_loss", "loss"),
        "num_samples": _pick_first(metadata, "num_samples", "sample_size", "train_samples") or 0,
        "gradient_norm": _pick_first(metadata, "gradient_norm", "grad_norm"),
        "gradient_magnitude": _pick_first(metadata, "gradient_magnitude", "grad_magnitude"),
        "bytes_sent": _pick_first(metadata, "bytes_sent", "upload_bytes"),
        "bytes_received": _pick_first(metadata, "bytes_received", "download_bytes"),
        "train_time_sec": _pick_first(metadata, "train_time_sec", "train_time"),
        "network_latency_ms": _pick_first(metadata, "network_latency_ms", "latency_ms"),
        "lat": _pick_first(metadata, "lat", "latitude") if _pick_first(metadata, "lat", "latitude") is not None else geo.get("lat"),
        "lng": _pick_first(metadata, "lng", "longitude") if _pick_first(metadata, "lng", "longitude") is not None else geo.get("lng"),
        "city": _pick_first(metadata, "city") or geo.get("city"),
        "country": _pick_first(metadata, "country") or geo.get("country"),
        "cpu_pct": _pick_first(metadata, "cpu_pct"),
        "ram_mb": _pick_first(metadata, "ram_mb"),
        "gpu_util_pct": _pick_first(metadata, "gpu_util_pct"),
        "gpu_vram_mb": _pick_first(metadata, "gpu_vram_mb"),
        "status": _pick_first(metadata, "status") or "active",
    }

    known_keys = set(normalized.keys()) | {"geo", "latitude", "longitude", "val_accuracy", "val_loss", "accuracy", "loss", "sample_size", "train_samples", "upload_bytes", "download_bytes", "train_time", "latency_ms", "grad_norm", "grad_magnitude"}
    extras = {key: value for key, value in metadata.items() if key not in known_keys}
    normalized.update(extras)
    return normalized


class APPFLxTracker:
    """
    Thin helper around HiveWatchRun for APPFLx entry points.

     It only emits run artifacts from inside
    the already-running ECS task.
    """

    def __init__(self, run):
        self.run = run

    @classmethod
    def from_server_agent(
        cls,
        *,
        server_agent,
        base_dir: str,
        bucket_name: str = "appflx-bucket",
        task_id: Optional[str] = None,
        artifact_dirname: str = "hivewatch",
        run_id: Optional[str] = None,
        algorithm: Optional[str] = None,
        upload_interval_sec: float = 1.0,
        verbose: bool = True,
    ) -> "APPFLxTracker":
        config = _to_plain_dict(getattr(server_agent, "server_agent_config", None))
        algorithm_name = algorithm or config.get("server_configs", {}).get("aggregator", "FedAvg")
        emitter = S3ArtifactEmitter(
            base_dir=base_dir,
            bucket_name=bucket_name,
            task_id=task_id,
            artifact_dirname=artifact_dirname,
            upload_interval_sec=upload_interval_sec,
        )
        run = init(
            run_id=run_id,
            algorithm=algorithm_name,
            config=config,
            emitters=[emitter],
            verbose=verbose,
        )
        return cls(run)

    def round_start(self, round_num: int):
        self.run.round_start(round_num)

    def set_server_metadata(self, **metadata):
        self.run.set_server_metadata(**metadata)

    def log_client_result(self, client_id: str, client_metadata: Dict[str, Any]):
        self.run.log_client_update(client_id, **normalize_client_metadata(client_id, client_metadata))

    def log_round(
        self,
        round_num: int,
        *,
        global_accuracy: Optional[float] = None,
        global_loss: Optional[float] = None,
        num_selected: Optional[int] = None,
        num_stragglers: int = 0,
        num_failures: int = 0,
        aggregation_time_sec: Optional[float] = None,
        algorithm_metadata: Optional[dict] = None,
    ):
        self.run.log_round(
            round_num,
            global_accuracy=global_accuracy,
            global_loss=global_loss,
            num_selected=num_selected,
            num_stragglers=num_stragglers,
            num_failures=num_failures,
            aggregation_time_sec=aggregation_time_sec,
            algorithm_metadata=algorithm_metadata,
        )

    def log_dropout(self, round_num: int, client_id: str, reason: Optional[str] = None):
        self.run.log_dropout(round_num, client_id, reason)

    def log_comm_failure(self, round_num: int, client_id: str, reason: Optional[str] = None):
        self.run.log_comm_failure(round_num, client_id, reason)

    def finish(self):
        self.run.finish()
