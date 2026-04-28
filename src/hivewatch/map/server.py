from __future__ import annotations

import json
import logging
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Dict, List, Optional

from .metadata import build_map_metadata_from_events

logger = logging.getLogger("hivewatch.map_server")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class MapServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 7070,
        runs_dir: str = "runs",
        run_id: Optional[str] = None,
        map_path: Optional[str] = None,
        watch: bool = False,
        poll_interval: float = 1.0,
    ):
        self.host = host
        self.port = port
        self.runs_dir = Path(runs_dir)
        self.map_path = map_path
        self.watch = watch
        self.poll_interval = poll_interval

        self._lock = threading.Lock()
        self._subscribers: List[queue.Queue] = []
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._watch_thread: Optional[threading.Thread] = None
        self._seen_offsets: Dict[Path, int] = {}
        self._live_run_id: Optional[str] = run_id
        self._fixed_run_id = run_id is not None

        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        if self._server is not None:
            return

        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_OPTIONS(self):
                self.send_response(200)
                self._cors()
                self.end_headers()

            def do_GET(self):
                path = self.path.rstrip("/")

                if path == "" or path == "/map":
                    self._serve_map()
                elif path == "/stream":
                    self._serve_sse()
                elif path == "/health":
                    self._serve_text("ok")
                elif path == "/runs":
                    self._serve_runs()
                elif path.startswith("/runs/") and path.endswith("/metadata"):
                    run_id = path[len("/runs/") : -len("/metadata")]
                    self._serve_run_metadata(run_id)
                elif path.startswith("/runs/") and path.endswith("/events"):
                    run_id = path[len("/runs/") : -len("/events")]
                    self._serve_run_events(run_id)
                else:
                    self.send_response(404)
                    self.end_headers()

            def _serve_map(self):
                candidates = []
                if server.map_path:
                    candidates.append(Path(server.map_path))
                candidates.append(Path(__file__).parent / "hivewatch_map.html")
                for candidate in candidates:
                    if candidate.exists():
                        content = candidate.read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self._cors()
                        self.end_headers()
                        self.wfile.write(content)
                        return
                self._serve_text("hivewatch_map.html not found", 404)

            def _serve_sse(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self._cors()
                self.end_headers()

                self.wfile.write(b"data: {\"event_type\": \"connected\"}\n\n")
                self.wfile.flush()

                if server._live_run_id:
                    mode = "static" if server._fixed_run_id else "live"
                    msg = json.dumps({"event_type": "init", "run_id": server._live_run_id, "mode": mode})
                    self.wfile.write(f"data: {msg}\n\n".encode("utf-8"))
                    self.wfile.flush()

                    if server._fixed_run_id:
                        jsonl_path = server.runs_dir / f"{server._live_run_id}.jsonl"
                        if jsonl_path.exists():
                            try:
                                with jsonl_path.open("r", encoding="utf-8") as handle:
                                    first = True
                                    for line in handle:
                                        line = line.strip()
                                        if not line:
                                            continue
                                        if first:
                                            try:
                                                obj = json.loads(line)
                                                if obj.get("event_type") == "init":
                                                    first = False
                                                    continue
                                            except Exception:
                                                pass
                                            first = False
                                        self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
                                        self.wfile.flush()
                            except Exception as exc:
                                logger.warning("[hivewatch] error reading events: %s", exc)
                        self.wfile.write(b"data: {\"event_type\": \"finished\"}\n\n")
                        self.wfile.flush()
                        return

                q = server._subscribe()
                try:
                    while True:
                        try:
                            msg = q.get(timeout=25)
                            self.wfile.write(msg.encode("utf-8"))
                            self.wfile.flush()
                        except queue.Empty:
                            self.wfile.write(b"data: {\"event_type\": \"ping\"}\n\n")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    server._unsubscribe(q)

            def _serve_runs(self):
                runs = []

                if server._live_run_id:
                    jsonl_path = server.runs_dir / f"{server._live_run_id}.jsonl"
                    if not jsonl_path.exists():
                        self._serve_text(f"run {server._live_run_id} not found", 404)
                        return

                    try:
                        with jsonl_path.open("r", encoding="utf-8") as handle:
                            first_line = handle.readline().strip()
                            if first_line:
                                header = json.loads(first_line)
                                with jsonl_path.open("r", encoding="utf-8") as handle:
                                    num_events = sum(1 for _ in handle)

                                runs.append(
                                    {
                                        "run_id": header.get("run_id", jsonl_path.stem),
                                        "algorithm": header.get("algorithm", "unknown"),
                                        "started_at": header.get("started_at", ""),
                                        "num_events": num_events,
                                        "file": jsonl_path.name,
                                        "metadata_file": f"{jsonl_path.stem}.map.json",
                                        "has_metadata": (server.runs_dir / f"{jsonl_path.stem}.map.json").exists(),
                                    }
                                )
                    except Exception as exc:
                        logger.warning("[hivewatch/map] could not read %s: %s", jsonl_path, exc)
                else:
                    for jsonl_path in sorted(
                        server.runs_dir.glob("*.jsonl"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    ):
                        try:
                            with jsonl_path.open("r", encoding="utf-8") as handle:
                                first_line = handle.readline().strip()
                                if not first_line:
                                    continue
                                header = json.loads(first_line)

                            with jsonl_path.open("r", encoding="utf-8") as handle:
                                num_events = sum(1 for _ in handle)

                            runs.append(
                                {
                                    "run_id": header.get("run_id", jsonl_path.stem),
                                    "algorithm": header.get("algorithm", "unknown"),
                                    "started_at": header.get("started_at", ""),
                                    "num_events": num_events,
                                    "file": jsonl_path.name,
                                    "metadata_file": f"{jsonl_path.stem}.map.json",
                                    "has_metadata": (server.runs_dir / f"{jsonl_path.stem}.map.json").exists(),
                                }
                            )
                        except Exception as exc:
                            logger.warning("[hivewatch/map] could not read %s: %s", jsonl_path, exc)

                self._serve_json(runs)

            def _serve_run_events(self, run_id: str):
                jsonl_path = server.runs_dir / f"{run_id}.jsonl"
                if not jsonl_path.exists():
                    self._serve_text(f"run {run_id} not found", 404)
                    return
                self._serve_json(server.read_events(jsonl_path))

            def _serve_run_metadata(self, run_id: str):
                metadata_path = server.runs_dir / f"{run_id}.map.json"
                if metadata_path.exists():
                    try:
                        with metadata_path.open("r", encoding="utf-8") as handle:
                            self._serve_json(json.load(handle))
                            return
                    except Exception as exc:
                        logger.warning("[hivewatch/map] could not read %s: %s", metadata_path, exc)

                jsonl_path = server.runs_dir / f"{run_id}.jsonl"
                if not jsonl_path.exists():
                    self._serve_text(f"run {run_id} not found", 404)
                    return

                self._serve_json(build_map_metadata_from_events(server.read_events(jsonl_path)))

            def _serve_json(self, data):
                body = json.dumps(data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self._cors()
                self.end_headers()
                self.wfile.write(body)

            def _serve_text(self, text: str, code: int = 200):
                body = text.encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self._cors()
                self.end_headers()
                self.wfile.write(body)

            def _cors(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")

        self._server = ThreadedHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        if self.watch:
            self._prime_watch_offsets()
            self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
            self._watch_thread.start()

    def serve_forever(self):
        self.start()
        if self._thread is not None:
            self._thread.join()

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def publish(self, payload: dict):
        if payload.get("run_id"):
            self._live_run_id = payload["run_id"]
        msg = f"data: {json.dumps(payload)}\n\n"
        with self._lock:
            dead = []
            for subscriber in self._subscribers:
                try:
                    subscriber.put_nowait(msg)
                except queue.Full:
                    dead.append(subscriber)
            for subscriber in dead:
                self._subscribers.remove(subscriber)

    def _subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._lock:
            self._subscribers.append(q)
        return q

    def _unsubscribe(self, q: queue.Queue):
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def read_events(self, jsonl_path: Path) -> List[dict]:
        events = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("[hivewatch/map] could not parse %s: %s", jsonl_path, exc)
        return events
