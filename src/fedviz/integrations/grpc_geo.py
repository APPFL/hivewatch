from __future__ import annotations

from typing import Any

from ..geo import extract_client_id


def patch_communicator_for_geo(communicator: Any, server_agent: Any) -> None:
    """Patch APPFL communicator RPC methods to record peer-to-client mappings."""

    def record_peer(client_id, peer):
        if client_id and str(client_id) not in server_agent._peer_by_client:
            server_agent._peer_by_client[str(client_id)] = peer
            print(f"[fedviz/patch] Mapped client_id={client_id!r} -> peer={peer!r}")

    def wrap_method(original):
        def wrapped(request_or_iter, context):
            peer = context.peer()
            print(f"[fedviz/patch] Incoming connection | peer={peer!r}")

            if hasattr(request_or_iter, "__iter__") and hasattr(request_or_iter, "__next__"):
                def capturing_iterator():
                    for request in request_or_iter:
                        record_peer(extract_client_id(request), peer)
                        yield request

                return original(capturing_iterator(), context)

            record_peer(extract_client_id(request_or_iter), peer)
            return original(request_or_iter, context)

        return wrapped

    for method_name in ["GetConfiguration", "GetGlobalModel", "UpdateGlobalModel", "InvokeCustomAction"]:
        if hasattr(communicator, method_name):
            setattr(communicator, method_name, wrap_method(getattr(communicator, method_name)))
            print(f"[fedviz] Patched: {method_name}")

    print("[fedviz] All RPC methods patched for geo IP resolution")
