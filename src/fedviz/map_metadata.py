from __future__ import annotations

from typing import Any, Dict, List, Optional


def merge_client_state(current: Optional[dict], update: dict) -> dict:
    merged = dict(current or {})
    for key, value in update.items():
        if value is not None:
            merged[key] = value
    return merged


def build_rounds_from_events(events: List[dict]) -> List[dict]:
    by_round: Dict[int, Dict[str, Any]] = {}

    for event in events:
        round_num = event.get("round")
        if round_num is None:
            continue

        round_state = by_round.setdefault(
            round_num,
            {
                "round": round_num,
                "globalAcc": None,
                "globalLoss": None,
                "duration": None,
                "divergence": None,
                "clients": {},
            },
        )

        event_type = event.get("event_type")
        if event_type == "round_end":
            metrics = event.get("round_metrics", {})
            if metrics.get("global_accuracy") is not None:
                round_state["globalAcc"] = metrics.get("global_accuracy")
            if metrics.get("global_loss") is not None:
                round_state["globalLoss"] = metrics.get("global_loss")
            if metrics.get("round_duration_sec") is not None:
                round_state["duration"] = metrics.get("round_duration_sec")
            if metrics.get("gradient_divergence") is not None:
                round_state["divergence"] = metrics.get("gradient_divergence")

        for client in event.get("clients", []):
            client_id = client.get("client_id")
            if client_id:
                round_state["clients"][client_id] = merge_client_state(
                    round_state["clients"].get(client_id),
                    client,
                )

        if event_type == "comm_failure":
            client_id = event.get("client_id")
            if client_id:
                round_state["clients"][client_id] = merge_client_state(
                    round_state["clients"].get(client_id),
                    {"client_id": client_id, "status": "failed"},
                )

    return [
        {
            "round": round_state["round"],
            "globalAcc": round_state["globalAcc"],
            "globalLoss": round_state["globalLoss"],
            "duration": round_state["duration"],
            "divergence": round_state["divergence"],
            "clients": list(round_state["clients"].values()),
        }
        for _, round_state in sorted(by_round.items())
    ]


def build_map_metadata_from_events(events: List[dict]) -> dict:
    init_event = next((event for event in events if event.get("event_type") == "init"), {})
    finished_event = next(
        (event for event in reversed(events) if event.get("event_type") == "finished"),
        {},
    )
    return {
        "schema_version": 1,
        "run_id": init_event.get("run_id"),
        "algorithm": init_event.get("algorithm"),
        "config": init_event.get("config", {}),
        "started_at": init_event.get("started_at"),
        "finished_at": finished_event.get("timestamp"),
        "rounds": build_rounds_from_events(events),
    }
