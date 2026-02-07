from __future__ import annotations

import jax
import jax.numpy as jnp

from .math3d import build_jacobian_single, project_ellipsoid3


@jax.jit
def mcp_residual_step3B_prox_hull(
    z,
    A_w,
    b_w,
    com,
    v_free,
    M_inv,
    dt,
    restitution,
    contact_eps,
    mu_fric,
    e_t,
    e_o,
    e_r,
    ecp_xy_reg,
    a0_xy,
):
    """Full MCP residual (v081): dynamics + friction prox + ECP-in-hull KKT + normal complementarity."""
    v_next = z[0:6]
    p_tx = z[6]
    p_ty = z[7]
    p_r = z[8]
    a = z[9:12]
    p_n = z[12]
    l = z[13:]

    r = a - com
    J = build_jacobian_single(r)

    J_tx = J[0:1, :]
    J_ty = J[1:2, :]
    J_n = J[2:3, :]

    p_contact = (J_n.T * p_n).reshape((6,)) + (J_tx.T * p_tx).reshape((6,)) + (J_ty.T * p_ty).reshape((6,))
    p_contact = p_contact + jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, p_r], dtype=jnp.float64)

    G_dyn = v_next - (v_free + (M_inv @ p_contact))

    vn = (J_n @ v_next)[0]
    vn_minus = (J_n @ v_free)[0]
    vn_minus_neg = jnp.minimum(vn_minus, 0.0)

    vtx = (J_tx @ v_next)[0]
    vty = (J_ty @ v_next)[0]
    v_r = v_next[5]

    gamma = 0.05
    reg = 1e-6
    limit = mu_fric * p_n

    pt0 = p_tx - gamma * vtx
    po0 = p_ty - gamma * vty
    pr0 = p_r - gamma * v_r

    ppt, ppo, ppr = project_ellipsoid3(pt0, po0, pr0, limit, e_t, e_o, e_r)

    G_fric_x = (p_tx - ppt) + reg * p_tx
    G_fric_y = (p_ty - ppo) + reg * p_ty
    G_fric_r = (p_r - ppr) + reg * p_r

    a0_xy = jnp.asarray(a0_xy, dtype=jnp.float64)
    grad = jnp.array([ecp_xy_reg * (a[0] - a0_xy[0]), ecp_xy_reg * (a[1] - a0_xy[1]), 1.0], dtype=jnp.float64)
    G_kkt = grad + (A_w.T @ l)

    slack = b_w - (A_w @ a)

    gap_curr = a[2]
    G_gap = gap_curr + contact_eps * p_n + dt * (vn + restitution * vn_minus_neg)

    free = jnp.concatenate([G_dyn, jnp.array([G_fric_x, G_fric_y, G_fric_r], dtype=jnp.float64), G_kkt])
    comp = jnp.concatenate([jnp.array([G_gap], dtype=jnp.float64), slack])
    return jnp.concatenate([free, comp])


mcp_jacobian_step3B_hull = jax.jit(jax.jacfwd(mcp_residual_step3B_prox_hull, argnums=0))


@jax.jit
def mcp_residual_step3B_prox_fixed_a(
    z,
    com,
    v_free,
    M_inv,
    dt,
    restitution,
    contact_eps,
    mu_fric,
    e_t,
    e_o,
    e_r,
    a_fixed_world,
    gap_curr,
):
    """Frozen MCP residual (v081): a fixed, no hull KKT, solve dynamics+friction+normal."""
    v_next = z[0:6]
    p_tx = z[6]
    p_ty = z[7]
    p_r = z[8]
    p_n = z[9]

    r = a_fixed_world - com
    J = build_jacobian_single(r)

    J_tx = J[0:1, :]
    J_ty = J[1:2, :]
    J_n = J[2:3, :]

    p_contact = (J_n.T * p_n).reshape((6,)) + (J_tx.T * p_tx).reshape((6,)) + (J_ty.T * p_ty).reshape((6,))
    p_contact = p_contact + jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, p_r], dtype=jnp.float64)

    G_dyn = v_next - (v_free + (M_inv @ p_contact))

    vn = (J_n @ v_next)[0]
    vn_minus = (J_n @ v_free)[0]
    vn_minus_neg = jnp.minimum(vn_minus, 0.0)

    vtx = (J_tx @ v_next)[0]
    vty = (J_ty @ v_next)[0]
    v_r = v_next[5]

    gamma = 0.05
    reg = 1e-6
    limit = mu_fric * p_n

    pt0 = p_tx - gamma * vtx
    po0 = p_ty - gamma * vty
    pr0 = p_r - gamma * v_r

    ppt, ppo, ppr = project_ellipsoid3(pt0, po0, pr0, limit, e_t, e_o, e_r)

    G_fric_x = (p_tx - ppt) + reg * p_tx
    G_fric_y = (p_ty - ppo) + reg * p_ty
    G_fric_r = (p_r - ppr) + reg * p_r

    G_gap = gap_curr + contact_eps * p_n + dt * (vn + restitution * vn_minus_neg)

    free = jnp.concatenate([G_dyn, jnp.array([G_fric_x, G_fric_y, G_fric_r], dtype=jnp.float64)])
    comp = jnp.array([G_gap], dtype=jnp.float64)
    return jnp.concatenate([free, comp])


mcp_jacobian_step3B_fixed_a = jax.jit(jax.jacfwd(mcp_residual_step3B_prox_fixed_a, argnums=0))
