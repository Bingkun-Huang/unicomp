"""pushlib.io_and_logging

Small I/O + logging helpers shared by entrypoints.

- Save run outputs to NPZ (same key names as returned by state_machine.run_push_minimal).

Keeping this separate lets the core logic stay focused on simulation/state machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def save_npz(out_path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Save payload to an .npz file.

    Parameters
    ----------
    out_path:
        Output path. If it doesn't end with '.npz', '.npz' will be appended.
    payload:
        Dict-like mapping of arrays/scalars.

    Returns
    -------
    Path to the saved file.
    """
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".npz":
        out_path = out_path.with_suffix(out_path.suffix + ".npz") if out_path.suffix else out_path.with_suffix('.npz')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # np.savez accepts scalars, lists, and numpy arrays.
    np.savez(str(out_path), **dict(payload))
    return out_path
