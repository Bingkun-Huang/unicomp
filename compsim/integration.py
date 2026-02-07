from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .math3d import get_world_points


def project_to_ground_and_damp(
    q_np: np.ndarray,
    v_np: np.ndarray,
    dt: float,
    local_pts_all,
    support_z: float = 0.0,         
    contact_tol: float = 5e-5,
    lin_damp_contact: float = 0.02,
    ang_damp_contact: float = 0.02,
    vz_sleep: float = 1e-3,
    pn_ground: float = 0.0,
    pn_support_thresh: float = 0.0,
):
    q_np = np.array(q_np, dtype=np.float64, copy=True)
    v_np = np.array(v_np, dtype=np.float64, copy=True)
    qj = jnp.array(q_np, dtype=jnp.float64)
    Vw_all = np.asarray(get_world_points(qj, local_pts_all))
    min_z = float(np.min(Vw_all[:, 2]))


    supported = (min_z <= float(support_z) + float(contact_tol)) and \
                (float(pn_ground) > float(pn_support_thresh))


    if min_z < float(support_z):
        dz = float(support_z) - min_z
        q_np[2] += dz
        if v_np[2] < 0.0:
            v_np[2] = 0.0

    if supported:
        v_np[0:2] *= (1.0 - float(lin_damp_contact))
        v_np[5] *= (1.0 - float(ang_damp_contact))
        if abs(v_np[2]) < float(vz_sleep):
            v_np[2] = 0.0

    return q_np, v_np, min_z, supported
