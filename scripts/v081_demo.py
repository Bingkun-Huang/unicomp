#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v081_demo.py (compsim)

Legacy interactive demo: visualize the T-block + green tool sphere and push it.

This script intentionally preserves the original monolithic behavior (mouse tool,
optional mocap target, etc.) by calling into `compsim.legacy_v081_monolithic`.

Run (from the folder that contains both `compsim/` and `scripts/`):
  export JAX_PLATFORM_NAME=cpu
  python3 scripts/v081_demo.py --view --use_tool --tool_mouse

Tips:
  - If you don't see the tool move with the mouse, press 'M' in the viewer.
  - Use --tool_pos0/--tool_des0 to change the initial tool position.
"""

from __future__ import annotations

# Make CPU the default BEFORE importing JAX via any module.
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import argparse

from compsim.legacy_v081_monolithic import run_simulation, _parse_vec3


def main() -> None:
    ap = argparse.ArgumentParser()

    # --- core sim ---
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=500000)
    ap.add_argument("--view", action="store_true")

    # --- ground contact ---
    ap.add_argument("--restitution", type=float, default=0.10)
    ap.add_argument("--proj_tol", type=float, default=1e-6)
    ap.add_argument("--ground_enable_margin", type=float, default=2e-3)
    ap.add_argument("--contact_eps", type=float, default=1e-6)
    ap.add_argument("--ecp_xy_reg", type=float, default=1e-2)
    ap.add_argument("--jac_reg", type=float, default=1e-8)

    # --- tool (green sphere) ---
    ap.add_argument("--use_tool", action="store_true")
    ap.add_argument("--tool_mass", type=float, default=1.0)
    ap.add_argument("--tool_radius", type=float, default=0.05)
    ap.add_argument("--tool_mu", type=float, default=0.6)
    ap.add_argument("--tool_k", type=float, default=800.0)
    ap.add_argument("--tool_d", type=float, default=80.0)
    ap.add_argument("--tool_fmax", type=float, default=10.0)
    ap.add_argument("--tool_pos0", type=str, default="0.08,-0.35,0.08")
    ap.add_argument("--tool_des0", type=str, default="0.08,-0.35,0.08")
    ap.add_argument("--tool_des_vel", type=str, default="0.0,0.25,0.0")
    ap.add_argument("--tool_tstart", type=float, default=1.2)
    ap.add_argument("--tool_restitution", type=float, default=0.0)
    ap.add_argument("--tool_contact_eps", type=float, default=1e-6)
    ap.add_argument("--tool_enable_margin", type=float, default=2e-4)

    # --- mouse control ---
    ap.add_argument("--tool_mouse", action="store_true")
    ap.add_argument("--mouse_sensitivity", type=float, default=0.002)
    ap.add_argument("--mouse_z_step", type=float, default=0.01)

    # --- optional mocap target (MuJoCo) ---
    ap.add_argument("--tool_mocap", action="store_true")
    ap.add_argument("--mocap_body", type=str, default="target_mocap")

    args = ap.parse_args()

    run_simulation(
        dt=args.dt,
        steps=args.steps,
        view=args.view,
        restitution=args.restitution,
        proj_tol=args.proj_tol,
        ground_enable_margin=args.ground_enable_margin,
        contact_eps=args.contact_eps,
        ecp_xy_reg=args.ecp_xy_reg,
        jac_reg=args.jac_reg,
        use_tool=args.use_tool,
        tool_mass=args.tool_mass,
        tool_radius=args.tool_radius,
        tool_mu=args.tool_mu,
        tool_k=args.tool_k,
        tool_d=args.tool_d,
        tool_fmax=args.tool_fmax,
        tool_pos0=_parse_vec3(args.tool_pos0),
        tool_des0=_parse_vec3(args.tool_des0),
        tool_des_vel=_parse_vec3(args.tool_des_vel),
        tool_tstart=args.tool_tstart,
        tool_restitution=args.tool_restitution,
        tool_contact_eps=args.tool_contact_eps,
        tool_enable_margin=args.tool_enable_margin,
        tool_mouse=args.tool_mouse,
        mouse_sensitivity=args.mouse_sensitivity,
        mouse_z_step=args.mouse_z_step,
        tool_mocap=args.tool_mocap,
        mocap_body=args.mocap_body,
    )


if __name__ == "__main__":
    main()
