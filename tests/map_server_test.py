from __future__ import annotations

import json
import queue
import urllib.error
import urllib.request

import pytest

from hivewatch.map.server import MapServer


def write_jsonl(path, events):
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def get_json(base_url: str, path: str):
    with urllib.request.urlopen(base_url + path, timeout=2) as response:
        return json.loads(response.read().decode("utf-8"))


def get_text_response(base_url: str, path: str):
    response = urllib.request.urlopen(base_url + path, timeout=2)
    try:
        return response, response.read().decode("utf-8")
    except Exception:
        response.close()
        raise


@pytest.fixture
def running_server(tmp_path):
    servers = []

    def start(**kwargs):
        server = MapServer(
            host="127.0.0.1",
            port=0,
            runs_dir=str(tmp_path),
            **kwargs,
        )
        server.start()
        servers.append(server)
        port = server._server.server_address[1]
        return server, f"http://127.0.0.1:{port}"

    yield start

    for server in servers:
        server.stop()


def test_map_server_serves_runs_events_metadata_and_map_file(tmp_path, running_server):
    events = [
        {
            "event_type": "init",
            "run_id": "run-abc",
            "algorithm": "FedAvg",
            "config": {"epochs": 1},
            "started_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "event_type": "round_end",
            "round": 1,
            "round_metrics": {"global_accuracy": 0.9},
            "clients": [{"client_id": "client-1", "local_accuracy": 0.8}],
        },
    ]
    write_jsonl(tmp_path / "run-abc.jsonl", events)
    map_file = tmp_path / "viewer.html"
    map_file.write_text("<html>custom map</html>", encoding="utf-8")

    _, base_url = running_server(map_path=str(map_file))

    runs = get_json(base_url, "/runs")
    assert runs == [
        {
            "run_id": "run-abc",
            "algorithm": "FedAvg",
            "started_at": "2026-01-01T00:00:00+00:00",
            "num_events": 2,
            "file": "run-abc.jsonl",
            "metadata_file": "run-abc.map.json",
            "has_metadata": False,
        }
    ]
    assert get_json(base_url, "/runs/run-abc/events") == events
    metadata = get_json(base_url, "/runs/run-abc/metadata")
    assert metadata["run_id"] == "run-abc"
    assert metadata["rounds"][0]["clients"][0]["client_id"] == "client-1"

    response, body = get_text_response(base_url, "/map")
    assert body == "<html>custom map</html>"
    assert response.headers["Cache-Control"].startswith("no-store")
    assert response.headers["Access-Control-Allow-Origin"] == "*"


def test_map_server_prefers_prebuilt_metadata_file(tmp_path, running_server):
    write_jsonl(
        tmp_path / "run-abc.jsonl",
        [{"event_type": "init", "run_id": "run-abc", "algorithm": "FedAvg"}],
    )
    metadata = {"schema_version": 99, "rounds": [{"round": 7}]}
    (tmp_path / "run-abc.map.json").write_text(json.dumps(metadata), encoding="utf-8")

    _, base_url = running_server()

    assert get_json(base_url, "/runs/run-abc/metadata") == metadata


def test_map_server_returns_404_for_missing_run(tmp_path, running_server):
    _, base_url = running_server()

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(base_url + "/runs/missing/events", timeout=2)

    assert excinfo.value.code == 404


def test_map_server_watch_mode_publishes_new_jsonl_events(tmp_path, running_server):
    server, _ = running_server(watch=True, poll_interval=0.01)
    subscriber = server._subscribe()
    event = {"event_type": "init", "run_id": "run-live", "algorithm": "FedAvg"}

    write_jsonl(tmp_path / "run-live.jsonl", [event])

    message = subscriber.get(timeout=2)
    assert message == f"data: {json.dumps(event)}\n\n"
    assert server._live_run_id == "run-live"
