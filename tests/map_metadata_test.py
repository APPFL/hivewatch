from __future__ import annotations

from hivewatch.map.metadata import build_map_metadata_from_events, build_rounds_from_events


def test_build_rounds_merges_client_updates_and_status_events():
    events = [
        {
            "event_type": "client_update",
            "round": 2,
            "clients": [
                {
                    "client_id": "client-a",
                    "local_accuracy": 0.6,
                    "current_local_steps": 100,
                    "_hidden_acc": 0.1,
                }
            ],
        },
        {
            "event_type": "round_end",
            "round": 2,
            "round_metrics": {
                "global_accuracy": 0.75,
                "global_loss": 0.4,
                "round_duration_sec": 3.2,
                "gradient_divergence": 0.12,
            },
            "clients": [
                {
                    "client_id": "client-a",
                    "local_loss": 0.25,
                    "current_local_steps": None,
                },
                {"client_id": "client-b", "local_accuracy": 0.7},
            ],
        },
        {"event_type": "comm_failure", "round": 3, "client_id": "client-b"},
    ]

    rounds = build_rounds_from_events(events)

    assert [round_state["round"] for round_state in rounds] == [2, 3]
    round_two = rounds[0]
    assert round_two["globalAcc"] == 0.75
    assert round_two["globalLoss"] == 0.4
    assert round_two["duration"] == 3.2
    assert round_two["divergence"] == 0.12

    clients = {client["client_id"]: client for client in round_two["clients"]}
    assert clients["client-a"]["local_accuracy"] == 0.6
    assert clients["client-a"]["local_loss"] == 0.25
    assert clients["client-a"]["current_local_steps"] == 100
    assert clients["client-a"]["_hidden_acc"] == 0.1
    assert clients["client-b"]["local_accuracy"] == 0.7

    assert rounds[1]["clients"] == [{"client_id": "client-b", "status": "failed"}]


def test_build_map_metadata_includes_run_server_rounds_and_finish_time():
    events = [
        {
            "event_type": "init",
            "run_id": "run-1234",
            "algorithm": "FedAvg",
            "config": {"epochs": 2},
            "started_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "event_type": "server_metadata",
            "server": {"lat": 41.7, "lng": -87.9, "city": "Chicago"},
        },
        {"event_type": "round_end", "round": 1, "round_metrics": {}, "clients": []},
        {"event_type": "finished", "timestamp": "2026-01-01T00:01:00+00:00"},
    ]

    metadata = build_map_metadata_from_events(events)

    assert metadata["schema_version"] == 1
    assert metadata["run_id"] == "run-1234"
    assert metadata["algorithm"] == "FedAvg"
    assert metadata["config"] == {"epochs": 2}
    assert metadata["server"] == {"lat": 41.7, "lng": -87.9, "city": "Chicago"}
    assert metadata["finished_at"] == "2026-01-01T00:01:00+00:00"
    assert metadata["rounds"][0]["round"] == 1
