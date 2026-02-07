from __future__ import annotations

import os

# -----------------------------
# JAX config (CPU recommended)
# -----------------------------
JAX_PLATFORM_NAME: str = os.environ.get("JAX_PLATFORM_NAME", "cpu")
JAX_ENABLE_X64: bool = True

# -----------------------------
# ECP freeze-by-velocity (NEW)
# -----------------------------
ECP_FREEZE_ENABLE = True
ECP_FREEZE_DWELL_N = 1000
ECP_FREEZE_L_REF = 0.6
ECP_FREEZE_V_SLEEP = 1.0
ECP_FREEZE_V_WAKE = 1.5
ECP_FREEZE_F_SLEEP_FACTOR = 0.01
ECP_FREEZE_F_WAKE_FACTOR = 0.03
ECP_FREEZE_TAU_SLEEP_FACTOR = 0.01
ECP_FREEZE_TAU_WAKE_FACTOR = 0.03
ECP_FREEZE_Z_PLANE = 0.0
ECP_FREEZE_DEBUG = False


def apply_jax_cpu() -> None:
    """Force JAX to use CPU and x64.

    Must run before importing JAX-heavy submodules if you want to be maximally
    safe about backend selection.
    """
    # Also set env var here (harmless if already set)
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    import jax

    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", bool(JAX_ENABLE_X64))
