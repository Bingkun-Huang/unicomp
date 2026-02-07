from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import mujoco

from . import state
from .math3d import rotate_vector_by_quaternion_np

# Lazily built on first access
_local_points_ref: jnp.ndarray | None = None


def generate_corners_from_model(model_: mujoco.MjModel) -> jnp.ndarray:
    """Generate corner samples (in COM frame) from all box geoms in body."""
    state.ensure_initialized()
    if state.body_id < 0:
        raise RuntimeError("compsim.state not initialized")

    pts = []
    gstart = int(model_.body_geomadr[state.body_id])
    gnum = int(model_.body_geomnum[state.body_id])
    for i in range(gstart, gstart + gnum):
        if model_.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        gp = np.array(model_.geom_pos[i], dtype=np.float64)
        gq = np.array(model_.geom_quat[i], dtype=np.float64)  # wxyz
        gs = np.array(model_.geom_size[i], dtype=np.float64)  # half extents
        for dx in (-1, 1):
            for dy in (-1, 1):
                for dz in (-1, 1):
                    corner_local = np.array([dx * gs[0], dy * gs[1], dz * gs[2]], dtype=np.float64)
                    corner_body = rotate_vector_by_quaternion_np(corner_local, gq) + gp
                    corner_com = corner_body - np.asarray(state.T_IPOS_BODY, dtype=np.float64)
                    pts.append(corner_com)
    return jnp.array(np.asarray(pts, dtype=np.float64), dtype=jnp.float64)


def get_local_points_ref(verbose: bool = True) -> jnp.ndarray:
    global _local_points_ref
    if _local_points_ref is None:
        state.ensure_initialized()
        _local_points_ref = generate_corners_from_model(state.model)
        if verbose:
            print(f"[LS-PROX] Total sampled points = {int(_local_points_ref.shape[0])}")
    return _local_points_ref
