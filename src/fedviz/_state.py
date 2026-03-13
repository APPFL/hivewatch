from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .run import FedVizRun

_run: Optional["FedVizRun"] = None


def _r() -> "FedVizRun":
    if _run is None:
        raise RuntimeError(
            "Call fedviz.init() before logging.\n"
            "Example: fedviz.init(wandb_project='my-fl-project')"
        )
    return _run
