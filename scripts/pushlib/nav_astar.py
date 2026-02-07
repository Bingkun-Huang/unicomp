# Auto-split from push_waypoints_compsim_live.py
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# =================== NAV: A* GRID (NO SDF) ===================
# ============================================================

import heapq
from dataclasses import field

# ---- A* params (NAV only) ----
ASTAR_RES = 0.006
ASTAR_MARGIN = 0.18
ASTAR_DIAGONAL = True
ASTAR_MAX_EXPAND = 160000
ASTAR_REPLAN_DIST = 0.03
ASTAR_REPLAN_EVERY = 0.60
ASTAR_GOAL_NUDGE_MAX = 0.08
ASTAR_PROJECT_EPS = 1e-4


@dataclass
class AStarNavState:
    path_xy: Optional[np.ndarray] = None   # (N,2)
    wp_i: int = 0
    goal_xy: np.ndarray = field(default_factory=lambda: np.array([np.nan, np.nan], dtype=np.float64))
    last_plan_step: int = -10**9
    replans: int = 0
    failed_plans: int = 0


def _grid_from_bounds(xmin, xmax, ymin, ymax, res):
    nx = int(np.ceil((xmax - xmin) / res)) + 1
    ny = int(np.ceil((ymax - ymin) / res)) + 1
    return nx, ny


def _xy_to_ij(xy, xmin, ymin, res):
    ix = int(np.round((float(xy[0]) - xmin) / res))
    iy = int(np.round((float(xy[1]) - ymin) / res))
    return ix, iy


def _ij_to_xy(ix, iy, xmin, ymin, res):
    x = xmin + float(ix) * res
    y = ymin + float(iy) * res
    return np.array([x, y], dtype=np.float64)


def _clip_ij(ix, iy, nx, ny):
    ix = int(np.clip(ix, 0, nx - 1))
    iy = int(np.clip(iy, 0, ny - 1))
    return ix, iy


def _build_disk_offsets(r_cells: int):
    offs = []
    rr2 = int(r_cells) * int(r_cells)
    for dx in range(-r_cells, r_cells + 1):
        for dy in range(-r_cells, r_cells + 1):
            if dx*dx + dy*dy <= rr2:
                offs.append((dx, dy))
    return offs


def build_occupancy_from_points(obs_xy: np.ndarray, xmin, ymin, nx, ny, res, inflate_radius: float):
    occ = np.zeros((nx, ny), dtype=np.bool_)
    if obs_xy is None:
        return occ
    obs_xy = np.asarray(obs_xy, dtype=np.float64).reshape(-1, 2)
    if obs_xy.shape[0] == 0:
        return occ

    r_cells = int(np.ceil(float(inflate_radius) / float(res)))
    r_cells = max(1, r_cells)
    disk = _build_disk_offsets(r_cells)

    for p in obs_xy:
        ix0, iy0 = _xy_to_ij(p, xmin, ymin, res)
        for dx, dy in disk:
            ix = ix0 + dx
            iy = iy0 + dy
            if 0 <= ix < nx and 0 <= iy < ny:
                occ[ix, iy] = True
    return occ


def _nearest_free_cell(occ: np.ndarray, ix, iy, max_r=25):
    nx, ny = occ.shape
    ix, iy = _clip_ij(ix, iy, nx, ny)
    if not occ[ix, iy]:
        return ix, iy
    for r in range(1, int(max_r) + 1):
        for dx in range(-r, r + 1):
            for dy in (-r, r):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < nx and 0 <= jy < ny and (not occ[jx, jy]):
                    return jx, jy
        for dy in range(-r + 1, r):
            for dx in (-r, r):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < nx and 0 <= jy < ny and (not occ[jx, jy]):
                    return jx, jy
    return None


def astar_plan(start_xy: np.ndarray, goal_xy: np.ndarray, obs_xy: np.ndarray, inflate_radius: float,
               res: float, margin: float, max_expand: int = ASTAR_MAX_EXPAND, diagonal: bool = True):
    start_xy = np.asarray(start_xy, dtype=np.float64).reshape(2,)
    goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(2,)
    obs_xy = np.asarray(obs_xy, dtype=np.float64).reshape(-1, 2)

    xmin = float(min(start_xy[0], goal_xy[0], float(np.min(obs_xy[:, 0])) if obs_xy.size else start_xy[0])) - float(margin) - float(inflate_radius)
    xmax = float(max(start_xy[0], goal_xy[0], float(np.max(obs_xy[:, 0])) if obs_xy.size else start_xy[0])) + float(margin) + float(inflate_radius)
    ymin = float(min(start_xy[1], goal_xy[1], float(np.min(obs_xy[:, 1])) if obs_xy.size else start_xy[1])) - float(margin) - float(inflate_radius)
    ymax = float(max(start_xy[1], goal_xy[1], float(np.max(obs_xy[:, 1])) if obs_xy.size else start_xy[1])) + float(margin) + float(inflate_radius)

    nx, ny = _grid_from_bounds(xmin, xmax, ymin, ymax, float(res))
    if nx * ny > 450000:
        scale = np.sqrt((nx * ny) / 450000.0)
        res2 = float(res) * float(scale)
        nx, ny = _grid_from_bounds(xmin, xmax, ymin, ymax, res2)
        res = res2

    occ = build_occupancy_from_points(obs_xy, xmin, ymin, nx, ny, float(res), float(inflate_radius))

    s_ix, s_iy = _xy_to_ij(start_xy, xmin, ymin, float(res))
    g_ix, g_iy = _xy_to_ij(goal_xy, xmin, ymin, float(res))
    s_ix, s_iy = _clip_ij(s_ix, s_iy, nx, ny)
    g_ix, g_iy = _clip_ij(g_ix, g_iy, nx, ny)

    s2 = _nearest_free_cell(occ, s_ix, s_iy, max_r=30)
    g2 = _nearest_free_cell(occ, g_ix, g_iy, max_r=30)
    if s2 is None or g2 is None:
        return None, {'ok': False, 'reason': 'start_or_goal_blocked', 'nx': nx, 'ny': ny, 'res': res}
    s_ix, s_iy = s2
    g_ix, g_iy = g2

    if diagonal:
        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    else:
        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def h(ix, iy):
        dx = (ix - g_ix)
        dy = (iy - g_iy)
        return np.hypot(dx, dy)

    inf = 1e18
    gscore = np.full((nx, ny), inf, dtype=np.float64)
    parent = np.full((nx, ny, 2), -1, dtype=np.int32)

    pq = []
    gscore[s_ix, s_iy] = 0.0
    heapq.heappush(pq, (h(s_ix, s_iy), 0.0, s_ix, s_iy))

    expanded = 0
    found = False

    while pq and expanded < int(max_expand):
        f, gcur, ix, iy = heapq.heappop(pq)
        if (ix == g_ix) and (iy == g_iy):
            found = True
            break
        if gcur > gscore[ix, iy] + 1e-12:
            continue
        expanded += 1

        for dx, dy in nbrs:
            jx, jy = ix + dx, iy + dy
            if not (0 <= jx < nx and 0 <= jy < ny):
                continue
            if occ[jx, jy]:
                continue
            step = np.hypot(dx, dy)
            ng = gcur + step
            if ng + 1e-12 < gscore[jx, jy]:
                gscore[jx, jy] = ng
                parent[jx, jy, 0] = ix
                parent[jx, jy, 1] = iy
                nf = ng + h(jx, jy)
                heapq.heappush(pq, (nf, ng, jx, jy))

    if not found:
        return None, {'ok': False, 'reason': 'no_path', 'expanded': expanded, 'nx': nx, 'ny': ny, 'res': res}

    path_ij = [(g_ix, g_iy)]
    ix, iy = g_ix, g_iy
    while not (ix == s_ix and iy == s_iy):
        pix, piy = int(parent[ix, iy, 0]), int(parent[ix, iy, 1])
        if pix < 0:
            break
        path_ij.append((pix, piy))
        ix, iy = pix, piy
    path_ij.reverse()

    path_xy = np.stack([_ij_to_xy(ix, iy, xmin, ymin, float(res)) for (ix, iy) in path_ij], axis=0)
    info = {'ok': True, 'expanded': expanded, 'nx': nx, 'ny': ny, 'res': res}
    return path_xy, info


def project_out_of_obstacles(xyz: np.ndarray, obs_xy: np.ndarray, inflate_radius: float, z_tool: float):
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3,)
    obs_xy = np.asarray(obs_xy, dtype=np.float64).reshape(-1, 2)
    if obs_xy.shape[0] == 0:
        xyz[2] = float(z_tool)
        return xyz

    dxy = obs_xy - xyz[0:2].reshape(1, 2)
    d2 = np.sum(dxy*dxy, axis=1)
    j = int(np.argmin(d2))
    d = float(np.sqrt(d2[j]))
    if d >= float(inflate_radius) + float(ASTAR_PROJECT_EPS):
        xyz[2] = float(z_tool)
        return xyz

    v = xyz[0:2] - obs_xy[j]
    nv = float(np.linalg.norm(v))
    if nv < 1e-12:
        v = np.array([1.0, 0.0], dtype=np.float64)
        nv = 1.0
    v = v / (nv + 1e-12)
    xyz[0:2] = obs_xy[j] + v * (float(inflate_radius) + float(ASTAR_PROJECT_EPS))
    xyz[2] = float(z_tool)
    return xyz


def nudge_goal_if_in_collision(goal_xyz: np.ndarray, n_out: np.ndarray, obs_xy: np.ndarray, inflate_radius: float, z_tool: float):
    goal_xyz = np.asarray(goal_xyz, dtype=np.float64).reshape(3,)
    goal_xyz[2] = float(z_tool)
    n_out = np.asarray(n_out, dtype=np.float64).reshape(3,)
    nxy = n_out[0:2]
    nn = float(np.linalg.norm(nxy))
    if nn < 1e-9:
        nxy = np.array([1.0, 0.0], dtype=np.float64)
        nn = 1.0
    nxy = nxy / (nn + 1e-12)

    obs_xy = np.asarray(obs_xy, dtype=np.float64).reshape(-1, 2)
    if obs_xy.shape[0] == 0:
        return goal_xyz

    dxy = obs_xy - goal_xyz[0:2].reshape(1, 2)
    d2 = np.sum(dxy*dxy, axis=1)
    if float(np.min(d2)) >= float(inflate_radius)**2:
        return goal_xyz

    step = float(min(0.004, float(ASTAR_RES)))
    max_steps = int(np.ceil(float(ASTAR_GOAL_NUDGE_MAX) / step))
    x = goal_xyz.copy()
    for _ in range(max_steps):
        x[0:2] += nxy * step
        dxy = obs_xy - x[0:2].reshape(1, 2)
        d2 = np.sum(dxy*dxy, axis=1)
        if float(np.min(d2)) >= float(inflate_radius)**2:
            return x
    return x


def nav_step_astar(
    tool_pos: np.ndarray,
    goal_pos: np.ndarray,
    obs_xy: np.ndarray,
    inflate_radius: float,
    dt: float,
    vxy_cap: float,
    nav: AStarNavState,
    z_tool: float,
    step_idx: int,
) -> Tuple[np.ndarray, np.ndarray, AStarNavState]:
    tool_pos = np.asarray(tool_pos, dtype=np.float64).reshape(3,)
    goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3,)
    obs_xy = np.asarray(obs_xy, dtype=np.float64).reshape(-1, 2)

    tool_xy = tool_pos[0:2].copy()
    goal_xy = goal_pos[0:2].copy()

    replan_every_steps = int(round(float(ASTAR_REPLAN_EVERY) / float(dt)))
    goal_moved = float(np.linalg.norm(goal_xy - nav.goal_xy))
    need_plan = (nav.path_xy is None) or (goal_moved > float(ASTAR_REPLAN_DIST)) or ((step_idx - nav.last_plan_step) >= replan_every_steps)

    if need_plan:
        path, info = astar_plan(tool_xy, goal_xy, obs_xy, float(inflate_radius), float(ASTAR_RES), float(ASTAR_MARGIN), diagonal=bool(ASTAR_DIAGONAL))
        nav.last_plan_step = int(step_idx)
        if path is None or (not info.get('ok', False)):
            nav.failed_plans += 1
            nav.path_xy = None
            nav.wp_i = 0
            tool_des = goal_pos.copy()
            tool_des[2] = float(z_tool)
            tool_next = kinematic_step_towards(tool_pos, tool_des, float(dt), float(vxy_cap))
            tool_next[2] = float(z_tool)
            tool_next = project_out_of_obstacles(tool_next, obs_xy, float(inflate_radius), float(z_tool))
            if DEBUG_NAV:
                print(f"[NAV-A*] plan FAIL reason={info.get('reason','?')} expanded={info.get('expanded',-1)} grid=({info.get('nx','?')}x{info.get('ny','?')}) res={info.get('res','?'):.4f}")
            return tool_next, tool_des, nav

        nav.path_xy = path
        nav.wp_i = 0
        nav.goal_xy = goal_xy.copy()
        nav.replans += 1
        if DEBUG_NAV:
            print(f"[NAV-A*] plan OK len={int(path.shape[0])} expanded={info.get('expanded',-1)} replans={nav.replans} grid=({info.get('nx','?')}x{info.get('ny','?')}) res={info.get('res','?'):.4f}")

    path = nav.path_xy
    if path is None or path.shape[0] == 0:
        tool_des = goal_pos.copy()
        tool_des[2] = float(z_tool)
        tool_next = kinematic_step_towards(tool_pos, tool_des, float(dt), float(vxy_cap))
        tool_next[2] = float(z_tool)
        tool_next = project_out_of_obstacles(tool_next, obs_xy, float(inflate_radius), float(z_tool))
        return tool_next, tool_des, nav

    while nav.wp_i < path.shape[0] - 1:
        wp = path[int(nav.wp_i)]
        if float(np.linalg.norm(wp - tool_xy)) <= float(max(ASTAR_RES, 0.5 * vxy_cap * dt)):
            nav.wp_i += 1
        else:
            break

    wp = path[int(nav.wp_i)]
    tool_des = tool_pos.copy()
    tool_des[0:2] = wp
    tool_des[2] = float(z_tool)

    tool_next = kinematic_step_towards(tool_pos, tool_des, float(dt), float(vxy_cap))
    tool_next[2] = float(z_tool)
    tool_next = project_out_of_obstacles(tool_next, obs_xy, float(inflate_radius), float(z_tool))
    return tool_next, tool_des, nav


def pick_reachable_contact_idx_for_nav(
    tool_pos: np.ndarray,
    q_block: np.ndarray,
    z_tool: float,
    d_nav: float,
    F_xy_des: np.ndarray,
    p_w_use: np.ndarray,
    n_w_use: np.ndarray,
    idx_map: np.ndarray,
    last_idx_use: Optional[int] = None,
    top_k: int = 40,
    contact_min_align: float = 0.15,
    contact_hysteresis_bonus: float = 0.20,
) -> Tuple[Optional[int], float, Optional[np.ndarray]]:
    tool_pos = np.asarray(tool_pos, dtype=np.float64).reshape(3,)
    F_xy_des = np.asarray(F_xy_des, dtype=np.float64).reshape(2,)
    p_w_use = np.asarray(p_w_use, dtype=np.float64)
    n_w_use = np.asarray(n_w_use, dtype=np.float64)
    idx_map = np.asarray(idx_map, dtype=np.int32)

    if p_w_use.shape[0] == 0:
        return None, -1e18, None

    if np.linalg.norm(F_xy_des) > 1e-9:
        d = F_xy_des / (np.linalg.norm(F_xy_des) + 1e-12)
    else:
        d = np.array([1.0, 0.0], dtype=np.float64)

    n_xy = n_w_use[:, 0:2]
    n_xy_n = np.linalg.norm(n_xy, axis=1)
    valid = n_xy_n > 1e-9

    n_xy_unit = np.zeros_like(n_xy)
    n_xy_unit[valid] = n_xy[valid] / (n_xy_n[valid].reshape(-1, 1) + 1e-12)
    push_dir = -n_xy_unit
    push_align = np.sum(push_dir * d.reshape(1, 2), axis=1)

    score = push_align.copy().astype(np.float64)
    score[~valid] = -1e18
    score[push_align < float(contact_min_align)] = -1e18
    if last_idx_use is not None and 0 <= int(last_idx_use) < score.shape[0]:
        score[int(last_idx_use)] += float(contact_hysteresis_bonus)

    order = np.argsort(-score)
    if int(top_k) > 0:
        order = order[: min(int(top_k), order.shape[0])]

    obs_xy = p_w_use[:, 0:2]

    for j in order:
        s = float(score[int(j)])
        if s <= -1e17:
            break

        p_c = p_w_use[int(j)]
        n_c = n_w_use[int(j)]
        n_out = n_c / (np.linalg.norm(n_c) + 1e-12)

        x_nav = (p_c + n_out * float(d_nav)).astype(np.float64)
        x_nav[2] = float(z_tool)
        x_goal = nudge_goal_if_in_collision(x_nav, n_out, obs_xy, float(d_nav), float(z_tool))

        path, info = astar_plan(tool_pos[0:2], x_goal[0:2], obs_xy, float(d_nav), float(ASTAR_RES), float(ASTAR_MARGIN), diagonal=bool(ASTAR_DIAGONAL))
        if path is not None and info.get('ok', False):
            return int(idx_map[int(j)]), s, x_goal.copy()

    return None, -1e18, None


# ============================================================
