"""compsim: complementarity-based contact simulation kernel.
CPU-only JAX
------------
"""

from __future__ import annotations

import os as _os

# Set env var before any JAX import happens anywhere.
_os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# Apply JAX config (imports jax).
from .config import apply_jax_cpu as _apply_jax_cpu

_apply_jax_cpu()

# Public API
from .state import init_from_xml, ensure_initialized
from .sim import TBlockSimulator_Step_NoBounce

__all__ = [
    "init_from_xml",
    "ensure_initialized",
    "TBlockSimulator_Step_NoBounce",
]
