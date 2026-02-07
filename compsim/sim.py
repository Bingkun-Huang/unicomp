from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import siconos.numerics as sn
from siconos.numerics import MCP, mcp_newton_FB_FBLSA

try:
    from scipy.spatial import ConvexHull
except Exception:  # pragma: no cover
    ConvexHull = None

from .config import (
    ECP_FREEZE_ENABLE, ECP_FREEZE_DWELL_N, ECP_FREEZE_L_REF,
    ECP_FREEZE_V_SLEEP, ECP_FREEZE_V_WAKE,
    ECP_FREEZE_F_SLEEP_FACTOR, ECP_FREEZE_F_WAKE_FACTOR,
    ECP_FREEZE_TAU_SLEEP_FACTOR, ECP_FREEZE_TAU_WAKE_FACTOR,
    ECP_FREEZE_Z_PLANE, ECP_FREEZE_DEBUG,
)

from .halfspace import normalize_halfspaces, reduce_coplanar_halfspaces
from .math3d import mass_matrix_inv_6x6, quat_to_R_wxyz, get_world_points
from .ground_mcp import (
    mcp_residual_step3B_prox_hull, mcp_jacobian_step3B_hull,
    mcp_residual_step3B_prox_fixed_a, mcp_jacobian_step3B_fixed_a,
)

class TBlockSimulator_Step_NoBounce:
    def __init__(
        self,
        dt,
        mass,
        inertia_diag_body,
        local_pts_all,
        Kmax=12,
        support_eps=1.5e-3,
        alpha_sigma=0.06,
        alpha_rho=1e-3,
        alpha_com_blend=0.25,
        mu_fric=0.5,
        e_t=1.0,
        e_o=1.0,
        e_r=None,
        e_r_factor=0.10,
        restitution=0.10,
        contact_eps=1e-6,
        ground_enable_margin=2e-3,
        proj_tol=1e-6,
        support_z: float = 0.0,
    ):
        self.dt = float(dt)
        self.restitution = float(restitution)
        self.contact_eps = float(contact_eps)
        self.proj_tol = float(proj_tol)
        self.ground_enable_margin = float(ground_enable_margin)

        # Support plane height (world z). Internally we shift the state so that
        # the support plane is at z=0 for the ground MCP.
        self.support_z = float(support_z)

        self.mass = jnp.array(float(mass), dtype=jnp.float64)
        self.mass_scalar = float(mass)
        self.inertia_body_diag = jnp.array(np.asarray(inertia_diag_body, dtype=np.float64), dtype=jnp.float64)

        self.local_pts_all = local_pts_all
        self.N_all = int(local_pts_all.shape[0])

        self.K = int(min(Kmax, self.N_all))
        self.support_eps = float(support_eps)
        self.alpha_sigma = float(alpha_sigma)

        self.mu_fric = float(mu_fric)
        self.e_t = float(e_t)
        self.e_o = float(e_o)

        # ---- convex hull halfspaces in BODY frame (for full MCP only)
        pts_hull = np.asarray(np.array(local_pts_all), dtype=np.float64)
        try:
            hull = ConvexHull(pts_hull)
            eq = hull.equations
            self.A_body = eq[:, :3].astype(np.float64)
            self.b_body = (-eq[:, 3]).astype(np.float64)
        except Exception:
            mn = pts_hull.min(axis=0)
            mx = pts_hull.max(axis=0)
            self.A_body = np.array(
                [
                    [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
                ],
                dtype=np.float64,
            )
            self.b_body = np.array([mx[0], -mn[0], mx[1], -mn[1], mx[2], -mn[2]], dtype=np.float64)

        self.A_body, self.b_body = normalize_halfspaces(self.A_body, self.b_body)
        self.A_body, self.b_body = reduce_coplanar_halfspaces(self.A_body, self.b_body)
        self.m_hull = int(self.A_body.shape[0])

        # tie-break in full MCP
        self.ecp_xy_reg = 1e-2

        # ellipsoid spin radius
        if e_r is None:
            bbox = pts_hull.max(axis=0) - pts_hull.min(axis=0)
            L = float(np.linalg.norm(bbox[0:2])) + 1e-12
            self.e_r = float(e_r_factor * L)
        else:
            self.e_r = float(e_r)

        self.alpha_rho = float(alpha_rho)
        self.alpha_com_blend = float(alpha_com_blend)

        # ---- full MCP sizes
        self.n1 = 12
        self.n2 = 1 + self.m_hull
        self.z_dim = self.n1 + self.n2

        self.options = sn.SolverOptions(sn.SICONOS_MCP_NEWTON_FB_FBLSA)
        self.options.iparam[sn.SICONOS_IPARAM_MAX_ITER] = 240
        self.options.dparam[sn.SICONOS_DPARAM_TOL] = 1e-6

        self.z_guess = np.zeros(self.z_dim, dtype=np.float64)
        self.z_guess[12] = 1e-6
        self.w_sol = np.zeros(self.z_dim, dtype=np.float64)

        self.jac_reg = 1e-8
        self.ecp_prev = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # ============================================================
        # ECP FREEZE (your final logic)
        #   - Freeze condition: (low speed) AND (no external wrench except gravity)
        #   - Wake condition: (speed high) OR (external wrench present) OR (tool impulse)
        # ============================================================
        self.ecp_freeze_enable = bool(ECP_FREEZE_ENABLE)
        self.ecp_freeze_dwell_n = int(ECP_FREEZE_DWELL_N)
        self.ecp_freeze_L = float(ECP_FREEZE_L_REF)

        self.ecp_freeze_v_sleep = float(ECP_FREEZE_V_SLEEP)
        self.ecp_freeze_v_wake  = float(ECP_FREEZE_V_WAKE)

        self.ecp_freeze_f_sleep = float(ECP_FREEZE_F_SLEEP_FACTOR  * self.mass_scalar * 9.81)
        self.ecp_freeze_f_wake  = float(ECP_FREEZE_F_WAKE_FACTOR   * self.mass_scalar * 9.81)
        self.ecp_freeze_tau_sleep = float(ECP_FREEZE_TAU_SLEEP_FACTOR * self.mass_scalar * 9.81 * self.ecp_freeze_L)
        self.ecp_freeze_tau_wake  = float(ECP_FREEZE_TAU_WAKE_FACTOR  * self.mass_scalar * 9.81 * self.ecp_freeze_L)

        self.ecp_freeze_z_plane = float(ECP_FREEZE_Z_PLANE)
        self.ecp_freeze_debug = bool(ECP_FREEZE_DEBUG)

        self._freeze_count = 0
        self._frozen = False

        # ---- frozen MCP sizes
        self.n1_frozen = 9
        self.n2_frozen = 1
        self.z_dim_frozen = self.n1_frozen + self.n2_frozen

        self.z_guess_frozen = np.zeros(self.z_dim_frozen, dtype=np.float64)
        self.z_guess_frozen[9] = 1e-6  # p_n
        self.w_sol_frozen = np.zeros(self.z_dim_frozen, dtype=np.float64)

        self._warmup()

    # -----------------------------
    # support-plane utilities
    # -----------------------------
    def _shift_q_to_support_frame(self, q_curr_np: np.ndarray) -> np.ndarray:
        """Return a copy of q with z translated so that the support plane is at z=0."""
        q_shift = np.asarray(q_curr_np, dtype=np.float64).copy()
        q_shift[2] -= self.support_z
        return q_shift

    def _ecp_to_world(self, ecp_shift_np: np.ndarray) -> np.ndarray:
        """Convert an ECP expressed in the support frame back to world coordinates."""
        e = np.asarray(ecp_shift_np, dtype=np.float64).copy()
        e[2] += self.support_z
        return e

    # -----------------------------
    # helpers
    # -----------------------------
    def _wrench_without_gravity(self, f_applied_np: np.ndarray) -> np.ndarray:
        """
        Your f_applied includes gravity already (f[2] += -m*g).
        Remove gravity so we can detect "external" wrench.
        """
        f = np.asarray(f_applied_np, dtype=np.float64).copy()
        f[2] += self.mass_scalar * 9.81
        return f

    def _v_eff(self, v_curr_np: np.ndarray) -> float:
        v_curr_np = np.asarray(v_curr_np, dtype=np.float64).reshape(6,)
        v_lin = float(np.linalg.norm(v_curr_np[0:3]))
        w = float(np.linalg.norm(v_curr_np[3:6]))
        return v_lin + self.ecp_freeze_L * w

    def _compute_a_frozen(self, q_curr_np: np.ndarray, Vw_all: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Frozen a (support frame): COM projection in XY + current zmin.
        Since the support plane is z=0 in this frame, zmin is also a gap surrogate.
        """
        com = np.asarray(q_curr_np[0:3], dtype=np.float64)
        zmin = float(np.min(np.asarray(Vw_all[:, 2], dtype=np.float64)))

        a = np.array([com[0], com[1], zmin], dtype=np.float64)

        # If you want to hard snap to a plane once very close:
        if abs(a[2] - self.ecp_freeze_z_plane) <= (5.0 * self.proj_tol):
            a[2] = self.ecp_freeze_z_plane

        return a, zmin

    def _update_freeze_state(
        self,
        v_curr_np: np.ndarray,
        f_applied_np: np.ndarray,
        zmin: float,
        z_pred_min: float,
        step_idx: int,
        tool_impulse_norm: float = 0.0,
    ) -> bool:
        """
        Returns current frozen flag.
        Only consider freezing when contact is possible (otherwise do not freeze).
        """
        if not self.ecp_freeze_enable:
            self._frozen = False
            self._freeze_count = 0
            return False

        # If clearly airborne -> do not freeze
        airborne = (zmin > self.ground_enable_margin) and (z_pred_min > 0.0)
        if airborne:
            self._frozen = False
            self._freeze_count = 0
            return False

        # External wrench (remove gravity)
        wng = self._wrench_without_gravity(f_applied_np)
        fn = float(np.linalg.norm(wng[0:3]))
        taun = float(np.linalg.norm(wng[3:6]))

        v_eff = self._v_eff(v_curr_np)
        tool_hit = float(tool_impulse_norm) > 1e-12

        if self._frozen:
            # Wake hysteresis: any strong motion OR any external wrench OR tool impulse
            if (v_eff > self.ecp_freeze_v_wake) or (fn > self.ecp_freeze_f_wake) or (taun > self.ecp_freeze_tau_wake) or tool_hit:
                if self.ecp_freeze_debug:
                    print(f"[ECP-FREEZE] step {step_idx}: WAKE  v_eff={v_eff:.3e}, fn={fn:.3e}, tau={taun:.3e}, tool={tool_impulse_norm:.3e}")
                self._frozen = False
                self._freeze_count = 0
        else:
            # Sleep dwell: must be slow AND no external wrench AND no tool impulse
            ok_sleep = (v_eff < self.ecp_freeze_v_sleep) and (fn < self.ecp_freeze_f_sleep) and (taun < self.ecp_freeze_tau_sleep) and (not tool_hit)
            if ok_sleep:
                self._freeze_count += 1
                if self._freeze_count >= self.ecp_freeze_dwell_n:
                    self._frozen = True
                    if self.ecp_freeze_debug:
                        print(f"[ECP-FREEZE] step {step_idx}: ENTER frozen  v_eff={v_eff:.3e}, fn={fn:.3e}, tau={taun:.3e}")
            else:
                self._freeze_count = 0

        return self._frozen

    # -----------------------------
    # warmup
    # -----------------------------
    def _warmup(self):
        # Warmup in the *support frame* (support plane at z=0).
        q0_world = jnp.array([0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
        q0 = q0_world.at[2].set(q0_world[2] - self.support_z)
        v0 = jnp.zeros((6,), dtype=jnp.float64)
        f0 = jnp.zeros((6,), dtype=jnp.float64)

        M_inv0 = mass_matrix_inv_6x6(self.mass, self.inertia_body_diag, q0[3:7])
        v_free0 = v0 + (M_inv0 @ (self.dt * f0))

        R0 = np.asarray(quat_to_R_wxyz(q0[3:7]))
        A_w0 = jnp.array(self.A_body @ R0.T, dtype=jnp.float64)
        b_w0 = jnp.array(self.b_body + (self.A_body @ R0.T) @ np.asarray(q0[0:3]), dtype=jnp.float64)

        Vw_all0 = np.asarray(get_world_points(q0, self.local_pts_all))
        a0 = Vw_all0[int(np.argmin(Vw_all0[:, 2]))].astype(np.float64)
        a0_xy = jnp.array(a0[0:2], dtype=jnp.float64)

        self.ecp_prev = a0.copy()

        # full guess init
        self.z_guess[:] = 0.0
        self.z_guess[0:6] = np.asarray(v0)
        self.z_guess[9:12] = a0
        self.z_guess[12] = 1e-6

        # frozen guess init
        self.z_guess_frozen[:] = 0.0
        self.z_guess_frozen[0:6] = np.asarray(v0)
        self.z_guess_frozen[9] = 1e-6

        # jit warmup full residual/jac
        mcp_residual_step3B_prox_hull(
            self.z_guess,
            A_w0,
            b_w0,
            q0[0:3],
            v_free0,
            M_inv0,
            self.dt,
            self.restitution,
            self.contact_eps,
            self.mu_fric,
            self.e_t,
            self.e_o,
            self.e_r,
            self.ecp_xy_reg,
            a0_xy,
        ).block_until_ready()

        mcp_jacobian_step3B_hull(
            self.z_guess,
            A_w0,
            b_w0,
            q0[0:3],
            v_free0,
            M_inv0,
            self.dt,
            self.restitution,
            self.contact_eps,
            self.mu_fric,
            self.e_t,
            self.e_o,
            self.e_r,
            self.ecp_xy_reg,
            a0_xy,
        ).block_until_ready()

        # jit warmup frozen residual/jac
        a_fix0 = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64)
        gap0 = jnp.array(0.0, dtype=jnp.float64)

        mcp_residual_step3B_prox_fixed_a(
            self.z_guess_frozen,
            q0[0:3],
            v_free0,
            M_inv0,
            self.dt,
            self.restitution,
            self.contact_eps,
            self.mu_fric,
            self.e_t,
            self.e_o,
            self.e_r,
            a_fix0,
            gap0,
        ).block_until_ready()

        mcp_jacobian_step3B_fixed_a(
            self.z_guess_frozen,
            q0[0:3],
            v_free0,
            M_inv0,
            self.dt,
            self.restitution,
            self.contact_eps,
            self.mu_fric,
            self.e_t,
            self.e_o,
            self.e_r,
            a_fix0,
            gap0,
        ).block_until_ready()

    # -----------------------------
    # main solve
    # -----------------------------
    def solve_step(self, q_curr_np, v_curr_np, f_applied_np, step_idx, return_ecp=False, tool_impulse_norm: float = 0.0):
        """
        tool_impulse_norm:
          - offline (no tool): keep default 0.0
          - realtime tool mode: pass np.linalg.norm(p_lin) from previous tool solve step to wake immediately
        """
        # Work in a translated frame where the support plane is z=0.
        # This avoids touching the JAX residuals: the ground MCP still assumes "ground at z=0",
        # while you can place the support plane at any world height via self.support_z.
        q_shift_np = self._shift_q_to_support_frame(q_curr_np)

        q = jnp.array(q_shift_np, dtype=jnp.float64)
        v = jnp.array(v_curr_np, dtype=jnp.float64)
        f = jnp.array(f_applied_np, dtype=jnp.float64)

        M_inv = mass_matrix_inv_6x6(self.mass, self.inertia_body_diag, q[3:7])
        v_free = v + (M_inv @ (self.dt * f))

        R = np.asarray(quat_to_R_wxyz(q[3:7]))
        A_w_np = (self.A_body @ R.T).astype(np.float64)
        b_w_np = (self.b_body + A_w_np @ np.asarray(q_shift_np[0:3], dtype=np.float64)).astype(np.float64)

        A_w = jnp.array(A_w_np, dtype=jnp.float64)
        b_w = jnp.array(b_w_np, dtype=jnp.float64)
        com = q[0:3]

        Vw_all = np.asarray(get_world_points(q, self.local_pts_all))
        zmin = float(Vw_all[:, 2].min())

        v_free_np = np.asarray(v_free, dtype=np.float64)
        v_lin = v_free_np[0:3]
        w_np = v_free_np[3:6]
        com_np = np.asarray(q_shift_np[0:3], dtype=np.float64)

        r_all = Vw_all - com_np[None, :]
        v_all = v_lin[None, :] + np.cross(w_np[None, :], r_all)
        z_pred_min = float(np.min(Vw_all[:, 2] + self.dt * v_all[:, 2]))

        # Early exit: clearly no contact
        if (zmin > self.ground_enable_margin) and (z_pred_min > 0.0):
            v_next = np.asarray(v_free, dtype=np.float64)
            self.z_guess[6:9] = 0.0
            self.z_guess[12] = 0.0
            self.z_guess[13:] = 0.0
            a_guess = Vw_all[int(np.argmin(Vw_all[:, 2]))].astype(np.float64)
            self.z_guess[9:12] = a_guess
            self.ecp_prev = a_guess.copy()

            # airborne -> definitely not frozen
            self._frozen = False
            self._freeze_count = 0

            info = 0
            if return_ecp:
                return v_next, self._ecp_to_world(self.ecp_prev), info
            return v_next, None, info

        # decide freeze state (your final logic)
        frozen_now = self._update_freeze_state(
            v_curr_np=v_curr_np,
            f_applied_np=f_applied_np,
            zmin=zmin,
            z_pred_min=z_pred_min,
            step_idx=step_idx,
            tool_impulse_norm=float(tool_impulse_norm),
        )

        # ============================================================
        # FROZEN branch: solve reduced MCP with fixed a
        # ============================================================
        if frozen_now:
            a_fix_np, gap_curr = self._compute_a_frozen(q_shift_np, Vw_all)
            a_fix = jnp.array(a_fix_np, dtype=jnp.float64)
            gap_j = jnp.array(float(gap_curr), dtype=jnp.float64)

            def call_F_frozen(n, z, w_out):
                w_out[:] = np.asarray(
                    mcp_residual_step3B_prox_fixed_a(
                        z,
                        com,
                        v_free,
                        M_inv,
                        self.dt,
                        self.restitution,
                        self.contact_eps,
                        self.mu_fric,
                        self.e_t,
                        self.e_o,
                        self.e_r,
                        a_fix,
                        gap_j,
                    )
                )

            def call_Jac_frozen(n, z, J_out):
                J_out[:] = np.asarray(
                    mcp_jacobian_step3B_fixed_a(
                        z,
                        com,
                        v_free,
                        M_inv,
                        self.dt,
                        self.restitution,
                        self.contact_eps,
                        self.mu_fric,
                        self.e_t,
                        self.e_o,
                        self.e_r,
                        a_fix,
                        gap_j,
                    )
                )
                if self.jac_reg > 0.0:
                    np.fill_diagonal(J_out, np.diag(J_out) + self.jac_reg)

            problem_frozen = MCP(self.n1_frozen, self.n2_frozen, call_F_frozen, call_Jac_frozen)

            # warm-start frozen
            self.z_guess_frozen[0:6] = np.asarray(v_curr_np, dtype=np.float64)
            self.z_guess_frozen[6:9] *= 0.9
            self.z_guess_frozen[9] = max(float(self.z_guess_frozen[9]), 1e-6)

            info = mcp_newton_FB_FBLSA(problem_frozen, self.z_guess_frozen, self.w_sol_frozen, self.options)

            if info != 0:
                # fallback: if frozen solve fails, wake and do full solve next step
                if self.ecp_freeze_debug:
                    print(f"[ECP-FREEZE] step {step_idx}: frozen solve FAIL -> wake")
                self._frozen = False
                self._freeze_count = 0
                # simple fallback velocity
                v_next = np.asarray(v_free, dtype=np.float64)
                self.ecp_prev = a_fix_np.copy()
                if return_ecp:
                    return v_next, self._ecp_to_world(self.ecp_prev), info
                return v_next, None, info

            v_next = self.z_guess_frozen[0:6].copy()
            if frozen_now and (self._v_eff(v_next) < self.ecp_freeze_v_sleep):
                v_next[3] *= 0.0
                v_next[4] *= 0.0
            # pn_ground is at index 9 in frozen vector
            # pn_ground = float(self.z_guess_frozen[9])

            # ECP output in frozen mode = fixed a (COM proj)
            ecp_used = a_fix_np.copy()
            self.ecp_prev = ecp_used.copy()

            if return_ecp:
                return v_next, self._ecp_to_world(self.ecp_prev), 0
            return v_next, None, 0

        # ============================================================
        # ACTIVE branch: solve full MCP (original)
        # ============================================================
        a_guess = Vw_all[int(np.argmin(Vw_all[:, 2]))].astype(np.float64)
        a0_xy = a_guess[0:2].copy()

        # keep your warm-start blending
        if self.ecp_prev is not None:
            a_guess[0:2] = 0.85 * a_guess[0:2] + 0.15 * self.ecp_prev[0:2]

        a0_xy_j = jnp.array(a0_xy, dtype=jnp.float64)

        def call_F(n, z, w_out):
            w_out[:] = np.asarray(
                mcp_residual_step3B_prox_hull(
                    z,
                    A_w,
                    b_w,
                    com,
                    v_free,
                    M_inv,
                    self.dt,
                    self.restitution,
                    self.contact_eps,
                    self.mu_fric,
                    self.e_t,
                    self.e_o,
                    self.e_r,
                    self.ecp_xy_reg,
                    a0_xy_j,
                )
            )

        def call_Jac(n, z, J_out):
            J_out[:] = np.asarray(
                mcp_jacobian_step3B_hull(
                    z,
                    A_w,
                    b_w,
                    com,
                    v_free,
                    M_inv,
                    self.dt,
                    self.restitution,
                    self.contact_eps,
                    self.mu_fric,
                    self.e_t,
                    self.e_o,
                    self.e_r,
                    self.ecp_xy_reg,
                    a0_xy_j,
                )
            )
            if self.jac_reg > 0.0:
                np.fill_diagonal(J_out, np.diag(J_out) + self.jac_reg)

        problem = MCP(self.n1, self.n2, call_F, call_Jac)

        # warm-start full
        self.z_guess[0:6] = np.asarray(v_curr_np, dtype=np.float64)
        self.z_guess[6:9] *= 0.9
        self.z_guess[9:12] = a_guess
        self.z_guess[12] = max(float(self.z_guess[12]), 1e-6)
        self.z_guess[13:] *= 0.5

        info = mcp_newton_FB_FBLSA(problem, self.z_guess, self.w_sol, self.options)

        if info != 0:
            v_next = np.array(v_curr_np, dtype=np.float64)
            v_next[0:3] = np.array(v_free[0:3])
            if v_next[2] < 0.0:
                v_next[2] = 0.0
            if return_ecp:
                return v_next, self._ecp_to_world(self.ecp_prev), info
            return v_next, None, info

        v_next = self.z_guess[0:6].copy()
        ecp_raw = self.z_guess[9:12].copy()

        self.ecp_prev = np.asarray(ecp_raw, dtype=np.float64).copy()

        if return_ecp:
            return v_next, self._ecp_to_world(self.ecp_prev), 0
        return v_next, None, 0



# ===========================
# Anti-bounce projection / damping (FIXED)
# ===========================