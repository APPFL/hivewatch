from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import boto3
import requests

from .metadata import build_manifest, build_map_metadata_from_events

logger = logging.getLogger("hivewatch_s3.storage")


def detect_ecs_task_id(default: str = "local-task") -> str:
    try:
        response = requests.get(
            "http://169.254.170.2/v2/metadata",
            headers={"Metadata": "true"},
            timeout=1,
        )
        response.raise_for_status()
        task_arn = response.json()["TaskARN"]
        return task_arn.split("/")[-1]
    except Exception:
        return default


class S3ArtifactEmitter:
    """
    Direct-upload artifact emitter for ECS/APPFLx jobs.

    The emitter keeps a local runs directory, similar to the local-first
    HiveWatch/FedViz flow:

        runs/<run_id>.jsonl
        runs/<run_id>.map.json

    and uploads those artifacts to:

        <base_dir>/<task_id>/<artifact_dirname>/

    throughout the run.
    """

    def __init__(
        self,
        *,
        base_dir: str,
        bucket_name: str = "appflx-bucket",
        task_id: Optional[str] = None,
        artifact_dirname: str = "hivewatch",
        aws_region: str = "us-east-1",
        local_cache_dir: Optional[str] = None,
        runs_dir: Optional[str] = None,
        upload_interval_sec: float = 1.0,
        s3_client=None,
        upload_events: bool = True,
        upload_stable_aliases: bool = True,
    ):
        self.base_dir = base_dir.strip("/")
        self.bucket_name = bucket_name
        self.task_id = task_id or detect_ecs_task_id()
        self.artifact_dirname = artifact_dirname.strip("/")
        self.artifact_prefix = f"{self.base_dir}/{self.task_id}/{self.artifact_dirname}"
        self.upload_interval_sec = max(0.0, upload_interval_sec)
        self.upload_events = upload_events
        self.s3 = s3_client or boto3.client("s3", region_name=aws_region)
        self._lock = threading.Lock()
        self._events = []
        self._run_id = None
        self._algorithm = None
        self._status = "initializing"
        self._started_at = None
        self._finished_at = None
        self._last_upload_ts = 0.0
        self._runs_dir_arg = runs_dir
        self.upload_stable_aliases = upload_stable_aliases

        cache_root = (
            Path(local_cache_dir).expanduser()
            if local_cache_dir is not None
            else Path.home() / ".hivewatch_s3" / self.task_id
        )
        self.cache_dir = cache_root
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = (
            Path(runs_dir).expanduser()
            if runs_dir is not None
            else self.cache_dir / "runs"
        )
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.events_path: Optional[Path] = None
        self.map_path: Optional[Path] = None
        self.manifest_path = self.cache_dir / "manifest.json"
        if self.manifest_path.exists():
            self.manifest_path.unlink()

    @property
    def runs_prefix(self) -> str:
        return f"{self.artifact_prefix}/runs"

    def on_init(self, run_id: str, algorithm: str, config: dict):
        payload = {
            "event_type": "init",
            "run_id": run_id,
            "algorithm": algorithm,
            "config": config,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with self._lock:
            self._run_id = run_id
            self._algorithm = algorithm
            self._status = "running"
            self._started_at = payload["started_at"]
            self.events_path = self.runs_dir / f"{run_id}.jsonl"
            self.map_path = self.runs_dir / f"{run_id}.map.json"
            for artifact_path in (self.events_path, self.map_path):
                if artifact_path.exists():
                    artifact_path.unlink()
            self._events.append(payload)
            self._append_event(payload)
            self._sync_locked(force=True)

    def on_client_update(self, client):
        payload = {
            "event_type": "client_update",
            "run_id": self._run_id,
            "round": client.round,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "clients": [client.to_dict()],
        }
        with self._lock:
            self._events.append(payload)
            self._append_event(payload)
            self._sync_locked()

    def on_round(self, summary, clients):
        payload = {
            "event_type": "round_end",
            "run_id": self._run_id,
            "round": summary.round,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "round_metrics": {
                "global_accuracy": summary.global_accuracy,
                "global_loss": summary.global_loss,
                "num_selected": summary.num_selected,
                "num_completed": summary.num_completed,
                "num_stragglers": summary.num_stragglers,
                "num_failures": summary.num_failures,
                "total_bytes_up": summary.total_bytes_up,
                "total_bytes_down": summary.total_bytes_down,
                "round_duration_sec": summary.round_duration_sec,
                "gradient_divergence": summary.gradient_divergence,
                "aggregation_time_sec": summary.aggregation_time_sec,
                "algorithm_metadata": summary.algorithm_metadata,
            },
            "clients": [client.to_dict() for client in clients],
        }
        with self._lock:
            self._events.append(payload)
            self._append_event(payload)
            self._sync_locked(force=True)

    def on_dropout(self, round_num: int, client_id: str, reason: Optional[str]):
        payload = {
            "event_type": "client_dropout",
            "run_id": self._run_id,
            "round": round_num,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": reason or "unknown",
            "clients": [{"client_id": client_id, "status": "dropped"}],
        }
        with self._lock:
            self._events.append(payload)
            self._append_event(payload)
            self._sync_locked()

    def on_comm_failure(self, round_num: int, client_id: str, reason: Optional[str]):
        payload = {
            "event_type": "comm_failure",
            "run_id": self._run_id,
            "round": round_num,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "client_id": client_id,
            "reason": reason or "unknown",
        }
        with self._lock:
            self._events.append(payload)
            self._append_event(payload)
            self._sync_locked()

    def on_checkpoint(self, round_num: int, path: str, metadata: dict):
        payload = {
            "event_type": "checkpoint",
            "run_id": self._run_id,
            "round": round_num,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "path": path,
            "metadata": metadata,
        }
        with self._lock:
            self._events.append(payload)
            self._append_event(payload)
            self._sync_locked()

    def on_server_metadata(self, metadata: dict):
        payload = {
            "event_type": "server_metadata",
            "run_id": self._run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "server": metadata,
        }
        with self._lock:
            self._events.append(payload)
            self._append_event(payload)
            self._sync_locked(force=True)

    def finish(self):
        payload = {
            "event_type": "finished",
            "run_id": self._run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with self._lock:
            self._status = "finished"
            self._finished_at = payload["timestamp"]
            self._events.append(payload)
            self._append_event(payload)
            self._sync_locked(force=True)

    def _append_event(self, payload: dict):
        if self.events_path is None:
            raise RuntimeError("events_path is not initialized; call on_init first")
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _sync_locked(self, force: bool = False):
        now = time.monotonic()
        if not force and (now - self._last_upload_ts) < self.upload_interval_sec:
            return

        map_metadata = build_map_metadata_from_events(self._events)
        manifest = build_manifest(
            run_id=self._run_id or "unknown-run",
            algorithm=self._algorithm or "unknown",
            base_dir=self.base_dir,
            task_id=self.task_id,
            artifact_prefix=self.artifact_prefix,
            bucket_name=self.bucket_name,
            started_at=self._started_at,
            finished_at=self._finished_at,
            status=self._status,
            event_count=len(self._events),
        )

        if self.map_path is None or self.events_path is None:
            raise RuntimeError("artifact paths are not initialized; call on_init first")

        self.map_path.write_text(json.dumps(map_metadata, indent=2) + "\n", encoding="utf-8")
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        try:
            self._upload_file(self.map_path, f"{self.runs_prefix}/{self.map_path.name}")
            self._upload_file(self.manifest_path, f"{self.artifact_prefix}/manifest.json")
            if self.upload_events:
                self._upload_file(self.events_path, f"{self.runs_prefix}/{self.events_path.name}")
            if self.upload_stable_aliases:
                self._upload_file(self.map_path, f"{self.artifact_prefix}/map.json")
                if self.upload_events:
                    self._upload_file(self.events_path, f"{self.artifact_prefix}/events.jsonl")
            self._last_upload_ts = now
        except Exception as exc:
            logger.warning("[hivewatch-s3] failed to upload artifacts: %s", exc)

    def _upload_file(self, local_path: Path, key: str):
        self.s3.upload_file(str(local_path), self.bucket_name, key)
