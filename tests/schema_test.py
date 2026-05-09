from __future__ import annotations

from hivewatch.schema import ClientUpdate


def test_client_update_preserves_unknown_metadata_fields():
    client = ClientUpdate.from_dict(
        {
            "client_id": "client-1",
            "round": 4,
            "local_accuracy": 0.8,
            "current_local_steps": 200,
            "blocking": True,
            "_hidden_acc": 0.99,
        }
    )

    assert client.extra == {
        "current_local_steps": 200,
        "blocking": True,
        "_hidden_acc": 0.99,
    }
    assert client.to_dict()["current_local_steps"] == 200
    assert client.to_dict()["blocking"] is True
    assert client.to_dict()["_hidden_acc"] == 0.99


def test_client_update_staleness_is_derived_when_base_round_is_known():
    client = ClientUpdate(client_id="client-1", round=5, base_round=2)

    assert client.staleness == 3
