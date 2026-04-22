from __future__ import annotations

from typing import Optional

from ._state import _r
from .run import HivewatchRun, init
from .schema import ClientUpdate, RoundSummary

__version__ = "0.1.0"

__all__ = [
    "HivewatchRun",
    "init",
    "round_start",
    "log_client_update",
    "log_round",
    "log_dropout",
    "log_comm_failure",
    "log_checkpoint",
    "set_server_metadata",
    "finish",
    "ClientUpdate",
    "RoundSummary",
]


# ── Global convenience functions ──────────────────────────────────────────────

def round_start(round: int):
    """Call at the start of each round for wall-time tracking."""
    _r().round_start(round)

def log_client_update(client_id: str, **kwargs):
    """Log one client's update. Pass metadata as keyword args."""
    _r().log_client_update(client_id, **kwargs)

def log_round(round: int, *, global_accuracy: Optional[float] = None, global_loss: Optional[float] = None, **kwargs):
    """Log the round summary after aggregation."""
    _r().log_round(round, global_accuracy=global_accuracy, global_loss=global_loss, **kwargs)

def log_dropout(round: int, client_id: str, reason: Optional[str] = None):
    _r().log_dropout(round, client_id, reason)

def log_comm_failure(round: int, client_id: str, reason: Optional[str] = None):
    _r().log_comm_failure(round, client_id, reason)

def log_checkpoint(round: int, path: str, **metadata):
    _r().log_checkpoint(round, path, **metadata)

def set_server_metadata(**metadata):
    _r().set_server_metadata(**metadata)

def finish():
    _r().finish()
