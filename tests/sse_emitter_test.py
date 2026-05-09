from __future__ import annotations

import json

from hivewatch.emitters.sse_emitter import SSEEmitter
from hivewatch.schema import ClientUpdate, RoundSummary


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_sse_emitter_persists_jsonl_and_map_metadata_with_custom_client_fields(tmp_path):
    emitter = SSEEmitter(runs_dir=str(tmp_path), serve_map=False)
    emitter.on_init("run-custom", "FedAvg", {"epochs": 3})
    emitter.on_server_metadata({"lat": 41.7, "lng": -87.9, "city": "Chicago"})

    client = ClientUpdate.from_dict(
        {
            "client_id": "client-1",
            "round": 1,
            "local_accuracy": 0.71,
            "local_loss": 0.33,
            "num_samples": 64,
            "lat": 1.3,
            "lng": 103.8,
            "city": "Singapore",
            "country": "SG",
            "current_local_steps": 200,
            "blocking": True,
            "_hidden_acc": 0.99,
        }
    )
    emitter.on_client_update(client)
    emitter.on_round(
        RoundSummary(
            round=1,
            global_accuracy=0.8,
            global_loss=0.2,
            num_selected=1,
            num_completed=1,
            round_duration_sec=2.5,
            gradient_divergence=0.0,
        ),
        [client],
    )
    emitter.finish()

    events = read_jsonl(tmp_path / "run-custom.jsonl")
    assert [event["event_type"] for event in events] == [
        "init",
        "server_metadata",
        "client_update",
        "round_end",
        "finished",
    ]
    client_update = events[2]["clients"][0]
    assert client_update["current_local_steps"] == 200
    assert client_update["blocking"] is True
    assert client_update["_hidden_acc"] == 0.99

    metadata = json.loads((tmp_path / "run-custom.map.json").read_text(encoding="utf-8"))
    assert metadata["server"]["city"] == "Chicago"
    assert metadata["finished_at"] == events[-1]["timestamp"]
    round_state = metadata["rounds"][0]
    assert round_state["globalAcc"] == 0.8
    assert round_state["duration"] == 2.5
    map_client = round_state["clients"][0]
    assert map_client["current_local_steps"] == 200
    assert map_client["blocking"] is True
    assert map_client["_hidden_acc"] == 0.99


def test_sse_emitter_merges_dropout_and_comm_failure_into_map_metadata(tmp_path):
    emitter = SSEEmitter(runs_dir=str(tmp_path), serve_map=False)
    emitter.on_init("run-status", "FedAvg", {})

    emitter.on_dropout(2, "client-drop", "timeout")
    emitter.on_comm_failure(2, "client-fail", "socket closed")
    emitter.finish()

    metadata = json.loads((tmp_path / "run-status.map.json").read_text(encoding="utf-8"))
    clients = {
        client["client_id"]: client
        for client in metadata["rounds"][0]["clients"]
    }
    assert clients["client-drop"]["status"] == "dropped"
    assert clients["client-fail"]["status"] == "failed"
