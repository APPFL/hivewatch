from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .run import HivewatchRun

_run: Optional["HivewatchRun"] = None


def _r() -> "HivewatchRun":
    if _run is None:
        raise RuntimeError(
            "Call hivewatch.init() before logging.\n"
            "Example: hivewatch.init(wandb_project='my-fl-project')"
        )
    return _run
