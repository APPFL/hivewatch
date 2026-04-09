"""
fedviz.emitters.sse_emitter
────────────────────────────
Starts a lightweight HTTP server that:
  1. Streams live fedviz events to the map dashboard via SSE (GET /stream)
  2. Persists every event to runs/<run_id>.jsonl for history replay
  3. Persists map-ready metadata to runs/<run_id>.map.json for deferred reloads
  4. Serves run history endpoints:
       GET /runs                    → list of all past runs
       GET /runs/<run_id>/events    → all events for a run (for replay)
       GET /runs/<run_id>/metadata  → map metadata for a run
       GET /                        → serves fedviz_map.html

Directory structure:
    runs/
      run-abcd-1234.jsonl     ← one JSON event per line, appended live
      run-abcd-1234.map.json  ← self-contained map metadata
      run-efgh-5678.jsonl
      ...
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from ..map import MapServer, merge_client_state
from ..schema import ClientUpdate, RoundSummary

logger = logging.getLogger("fedviz.emitters.sse")


class SSEEmitter:
    """
    Emitter that broadcasts fedviz events over SSE and persists to JSONL files.
    """

    def __init__(
        self,
        host:     str  = "0.0.0.0",
        port:     int  = 7070,
        runs_dir: str  = "runs",
        map_path: Optional[str] = None,   # path to fedviz_map.html
        serve_map: bool = True,
    ):
        self.host     = host
        self.port     = port
        self.runs_dir = Path(runs_dir)
        self.map_path = map_path          # if None, serves a placeholder
        self.serve_map = serve_map
        self.run_id:  Optional[str] = None
        self._jsonl:  Optional[object]   = None   # open file handle
        self._metadata_path: Optional[Path] = None
        self._map_metadata: Optional[dict] = None
        self._lock    = threading.Lock()
        self._server: Optional[MapServer] = None

        self.runs_dir.mkdir(parents=True, exist_ok=True)

    # ── Emitter hooks ─────────────────────────────────────────────────────────

    def on_init(self, run_id: str, algorithm: str, config: dict):
        self.run_id = run_id
        # Open JSONL file for this run
        jsonl_path  = self.runs_dir / f"{run_id}.jsonl"
        self._metadata_path = self.runs_dir / f"{run_id}.map.json"
        self._jsonl = open(jsonl_path, "a", encoding="utf-8")

        # Write run header as first line
        
        header = {
            "event_type": "init",
            "run_id":     run_id,
            "algorithm":  algorithm,
            "config":     config,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self._map_metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "algorithm": algorithm,
            "config": config,
            "started_at": header["started_at"],
            "finished_at": None,
            "rounds": [],
        }
        self._append(header)
        self._write_metadata()
        if self.serve_map:
            self._start_server()
        self._broadcast(header)

        print(f"[fedviz/sse] run={run_id}")
        print(f"[fedviz/sse] history → {jsonl_path}")
        if self.serve_map:
            print(f"[fedviz/sse] dashboard → http://localhost:{self.port}")
        else:
            print(f"[fedviz/sse] dashboard disabled; run `hivewatch map run --runs-dir {self.runs_dir}` to serve the map")

    def on_round(self, summary: RoundSummary, clients: List[ClientUpdate]):
        payload = {
            "event_type":    "round_end",
            "run_id":        self.run_id,
            "round":         summary.round,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "round_metrics": {
                "global_accuracy":     summary.global_accuracy,
                "global_loss":         summary.global_loss,
                "num_selected":        summary.num_selected,
                "num_completed":       summary.num_completed,
                "num_stragglers":      summary.num_stragglers,
                "round_duration_sec":  summary.round_duration_sec,
                "gradient_divergence": summary.gradient_divergence,
            },
            "clients": [self._client_dict(c) for c in clients],
        }
        round_state = self._ensure_round_state(summary.round)
        round_state["globalAcc"] = summary.global_accuracy
        round_state["globalLoss"] = summary.global_loss
        round_state["duration"] = summary.round_duration_sec
        round_state["divergence"] = summary.gradient_divergence
        for client in clients:
            self._upsert_client(summary.round, self._client_dict(client))
        self._append(payload)
        self._write_metadata()
        self._broadcast(payload)

    def on_client_update(self, client: ClientUpdate):
        payload = {
            "event_type": "client_update",
            "run_id":     self.run_id,
            "round":      client.round,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "clients":    [self._client_dict(client)],
        }
        if client.round is not None:
            self._upsert_client(client.round, self._client_dict(client))
        self._append(payload)
        self._write_metadata()
        self._broadcast(payload)

    def on_dropout(self, round: int, client_id: str, reason: Optional[str]):
        payload = {
            "event_type": "client_dropout",
            "run_id":     self.run_id,
            "round":      round,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "clients":    [{"client_id": client_id, "status": "dropped"}],
            "reason":     reason or "unknown",
        }
        if round is not None:
            self._upsert_client(round, {"client_id": client_id, "status": "dropped"})
        self._append(payload)
        self._write_metadata()
        self._broadcast(payload)

    def on_comm_failure(self, round: int, client_id: str, reason: Optional[str]):
        payload = {
            "event_type": "comm_failure",
            "run_id":     self.run_id,
            "round":      round,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "client_id":  client_id,
            "reason":     reason or "unknown",
        }
        if round is not None:
            self._upsert_client(round, {"client_id": client_id, "status": "failed"})
        self._append(payload)
        self._write_metadata()
        self._broadcast(payload)

    def on_checkpoint(self, round: int, path: str, metadata: dict):
        payload = {
            "event_type": "checkpoint",
            "run_id":     self.run_id,
            "round":      round,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "path":       path,
        }
        self._append(payload)
        self._write_metadata()
        self._broadcast(payload)

    def finish(self):
        payload = {
            "event_type":  "finished",
            "run_id":      self.run_id,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
        }
        if self._map_metadata is not None:
            self._map_metadata["finished_at"] = payload["timestamp"]
        self._append(payload)
        self._write_metadata()
        self._broadcast(payload)

        if self._jsonl:
            self._jsonl.close()
            self._jsonl = None

        if self._server:
            # Keep server alive so dashboard can still load history
            # It runs as a daemon thread so it dies with the process
            pass

    # ── Persistence ───────────────────────────────────────────────────────────

    def _append(self, payload: dict):
        if self._jsonl is None:
            return
        with self._lock:
            self._jsonl.write(json.dumps(payload) + "\n")
            self._jsonl.flush()

    def _ensure_round_state(self, round_num: int) -> dict:
        if self._map_metadata is None:
            raise RuntimeError("map metadata not initialized")

        rounds = self._map_metadata["rounds"]
        for round_state in rounds:
            if round_state["round"] == round_num:
                return round_state

        round_state = {
            "round": round_num,
            "globalAcc": None,
            "globalLoss": None,
            "duration": None,
            "divergence": None,
            "clients": [],
        }
        rounds.append(round_state)
        rounds.sort(key=lambda item: item["round"])
        return round_state

    def _upsert_client(self, round_num: int, client: dict):
        round_state = self._ensure_round_state(round_num)
        clients = {item["client_id"]: item for item in round_state["clients"] if item.get("client_id")}
        client_id = client.get("client_id")
        if not client_id:
            return
        clients[client_id] = merge_client_state(clients.get(client_id), client)
        round_state["clients"] = list(clients.values())

    def _write_metadata(self):
        if self._metadata_path is None or self._map_metadata is None:
            return
        with self._lock:
            self._metadata_path.write_text(
                json.dumps(self._map_metadata, indent=2) + "\n",
                encoding="utf-8",
            )

    # ── SSE broadcast ─────────────────────────────────────────────────────────

    def _broadcast(self, payload: dict):
        if self._server is not None:
            self._server.publish(payload)

    # ── HTTP server ───────────────────────────────────────────────────────────

    def _start_server(self):
        if self._server is not None:
            return

        self._server = MapServer(
            host=self.host,
            port=self.port,
            runs_dir=str(self.runs_dir),
            map_path=self.map_path,
            watch=False,
        )
        self._server.start()

    # ── Client dict ───────────────────────────────────────────────────────────

    @staticmethod
    def _client_dict(c: ClientUpdate) -> dict:
        return {
            "client_id":      c.client_id,
            "round":          c.round,
            "lat":            c.lat,
            "lng":            c.lng,
            "city":           c.city,
            "country":        c.country,
            "local_accuracy": c.local_accuracy,
            "local_loss":     c.local_loss,
            "num_samples":    c.num_samples,
            "gradient_norm":  c.gradient_norm,
            "bytes_sent":     c.bytes_sent,
            "train_time_sec": c.train_time_sec,
            "cpu_pct":        c.cpu_pct,
            "ram_mb":         c.ram_mb,
            "status":         c.status,
        }
