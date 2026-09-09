"""Full-map occupancy + directed A* + navigation state machine. No ROS."""

from __future__ import annotations

import heapq
import math
import time as time_mod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from runtime.lidar_avoid import (
    scan_sectors,
    speed_for_clearance,
    stop_dist_for_speed,
    sweep_gate,
)
from runtime.nav_geometry import (
    GEOMETRY_VERSION,
    R_PLAN_REF_M,
    forbidden_intrusion_m,
    goal_region_ok,
    in_center_forbidden,
    in_nav_bounds,
    in_physical_solid,
    in_south_gap,
    preferred_corridor_line,
)
from runtime.nav_types import NavGoal, PlanResult, infer_goal_kind
from runtime.p4_params import (
    ARRIVE_HOLD_S,
    BASE_LIN,
    CORRIDOR_COST,
    EXPLORE_SPEED_SCALE,
    FACE_BLOCK_S,
    FAIL_BACKOFF_S,
    FREE_COST,
    GRID_RES,
    HIT_MAX,
    HIT_MIN,
    LETHAL_SCORE,
    OFF_TRACK_M,
    PATH_SKIP_M,
    PROGRESS_EPS_M,
    PROGRESS_WINDOW_S,
    REPLAN_MIN_S,
    ROUTE_HOLD_S,
    SCORE_MAX,
    SEARCH_BUDGET_MS,
    STILL_ANG_RAD_S,
    STILL_LIN_M_S,
    STOP_DIST,
    UNKNOWN_COST,
    UNKNOWN_SPEED_SCALE,
    WALL_ECHO_M,
    YELLOW_SOUTH_Y,
)
from runtime.scene_zones import (
    HUG_WEST_X,
    NAV_BOUNDS,
    WEST_LANE_Y,
    in_center_wall_band,
    in_north_racks,
)
from runtime.waypoint_nav import WaypointFollower, wrap_to_pi

Point = Tuple[float, float]
Cell = Tuple[int, int]
DIRS = (
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
)
DIR_YAW = tuple(math.atan2(dr, dc) for dc, dr in DIRS)
V_REF = 0.30
W_REF = 0.55
LARGE_TURN = math.radians(60.0)
STOP_TURN_S = 0.70


def _world_bounds() -> Tuple[float, float, float, float]:
    return (
        float(NAV_BOUNDS["x"][0]),
        float(NAV_BOUNDS["x"][1]),
        float(NAV_BOUNDS["y"][0]),
        float(NAV_BOUNDS["y"][1]),
    )


def _point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    length2 = vx * vx + vy * vy
    if length2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / length2))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def _polyline_distance(px: float, py: float, line: Sequence[Point]) -> float:
    if not line:
        return 0.0
    if len(line) == 1:
        return math.hypot(px - line[0][0], py - line[0][1])
    nearest = float("inf")
    for index in range(len(line) - 1):
        nearest = min(nearest, _point_to_segment(px, py, *line[index], *line[index + 1]))
    return nearest


def _circle_rect_dist(cx: float, cy: float, x0: float, y0: float, x1: float, y1: float) -> float:
    px = min(max(cx, x0), x1)
    py = min(max(cy, y0), y1)
    return math.hypot(cx - px, cy - py)


def preferred_corridor(goal_xy: Sequence[float], kind: Optional[str] = None) -> List[Point]:
    return preferred_corridor_line(kind or infer_goal_kind(goal_xy))


def illegal_shortcut(x0: float, y0: float, x1: float, y1: float) -> bool:
    """True if the segment crosses a solid or the south gap. East southbound is legal."""
    if in_center_wall_band(x0, y0) or in_center_wall_band(x1, y1):
        return True
    if in_south_gap(x0, y0) or in_south_gap(x1, y1):
        return True
    left, right = float(NAV_BOUNDS["x"][0]), float(NAV_BOUNDS["x"][1])
    _ = (left, right)
    samples = max(2, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / 0.04)))
    for step in range(samples + 1):
        t = step / samples
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        if in_physical_solid(x, y) or in_south_gap(x, y):
            return True
        if not in_nav_bounds(x, y):
            return True
    return False


def _dir_from_yaw(yaw: float) -> int:
    best = 0
    best_d = 10.0
    for index, heading in enumerate(DIR_YAW):
        delta = abs(wrap_to_pi(float(yaw) - heading))
        if delta < best_d:
            best = index
            best_d = delta
    return best


def _turn_cost(prev_dir: int, nxt_dir: int) -> float:
    if prev_dir < 0:
        return 0.0
    step = min((nxt_dir - prev_dir) % 8, (prev_dir - nxt_dir) % 8)
    yaw = step * (math.pi / 4.0)
    cost = yaw / W_REF
    if yaw >= LARGE_TURN - 1e-6:
        cost += STOP_TURN_S
    return cost


class OccupancyGrid:
    """0.10 m grid over NAV_BOUNDS. Physical solids, centre bans, lidar evidence."""

    def __init__(self, resolution: float = GRID_RES, inflate: float = R_PLAN_REF_M) -> None:
        self.res = float(resolution)
        self.inflate = float(inflate)
        self.geometry_version = GEOMETRY_VERSION
        x0, x1, y0, y1 = _world_bounds()
        self.origin_x = x0
        self.origin_y = y0
        self.bound_x1 = x1
        self.bound_y1 = y1
        self.cols = int(math.ceil((x1 - x0) / self.res))
        self.rows = int(math.ceil((y1 - y0) / self.res))
        n = self.cols * self.rows
        self.static_solid = bytearray(n)
        self.center_forbidden = bytearray(n)
        self.static = bytearray(n)
        self.observed = bytearray(n)
        self.score = [0] * n
        self.last_hit = [0.0] * n
        self.last_free = [0.0] * n
        self.free_streak = [0] * n
        self.hit_scans = [0] * n
        self.confirmed = bytearray(n)
        self.lethal = bytearray(n)
        self.dynamic = bytearray(n)
        self.temp_block = bytearray(n)
        self.map_version = 1
        self.occupancy_version = 1
        self._accepted_scan_id = -1
        self._dynamic_inflate = self.inflate
        self._paint_static()
        self.rebuild_masks()

    def _index(self, col: int, row: int) -> int:
        return row * self.cols + col

    def in_grid(self, col: int, row: int) -> bool:
        if not (0 <= col < self.cols and 0 <= row < self.rows):
            return False
        x, y = self.cell_center(col, row)
        return in_nav_bounds(x, y)

    def world_to_cell(self, x: float, y: float) -> Cell:
        return (
            int(math.floor((float(x) - self.origin_x) / self.res)),
            int(math.floor((float(y) - self.origin_y) / self.res)),
        )

    def cell_center(self, col: int, row: int) -> Point:
        return (
            self.origin_x + (col + 0.5) * self.res,
            self.origin_y + (row + 0.5) * self.res,
        )

    def cell_rect(self, col: int, row: int) -> Tuple[float, float, float, float]:
        x0 = self.origin_x + col * self.res
        y0 = self.origin_y + row * self.res
        return x0, y0, x0 + self.res, y0 + self.res

    def _paint_static(self) -> None:
        for row in range(self.rows):
            for col in range(self.cols):
                x, y = self.cell_center(col, row)
                idx = self._index(col, row)
                if in_physical_solid(x, y):
                    self.static_solid[idx] = 1
                if in_center_forbidden(x, y):
                    self.center_forbidden[idx] = 1
        radius = float(self.inflate)
        span = max(1, int(math.ceil((radius + self.res) / self.res)))
        solids = [
            (idx % self.cols, idx // self.cols)
            for idx, flag in enumerate(self.static_solid)
            if flag
        ]
        for scol, srow in solids:
            sx0, sy0, sx1, sy1 = self.cell_rect(scol, srow)
            for dr in range(-span, span + 1):
                for dc in range(-span, span + 1):
                    col, row = scol + dc, srow + dr
                    if not (0 <= col < self.cols and 0 <= row < self.rows):
                        continue
                    cx, cy = self.cell_center(col, row)
                    if not in_nav_bounds(cx, cy):
                        continue
                    if _circle_rect_dist(cx, cy, sx0, sy0, sx1, sy1) <= radius + 1e-6:
                        # Keep the yellow lane open. Divider inflate to y=2.20
                        # plus a 0.10 cell made (1.56, 2.23) look occupied.
                        if cy >= YELLOW_SOUTH_Y and sy1 <= WEST_LANE_Y + 1e-6:
                            continue
                        self.static[self._index(col, row)] = 1
        for idx, flag in enumerate(self.static_solid):
            if flag:
                self.static[idx] = 1

    def _bump_map(self) -> None:
        self.map_version += 1

    def _bump_occ(self) -> None:
        self.occupancy_version += 1
        self.map_version += 1

    def integrate_scan(
        self,
        x: float,
        y: float,
        yaw: float,
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
        now: float,
        scan_id: Optional[int] = None,
        range_min: float = HIT_MIN,
        range_max: float = HIT_MAX,
        laser_xy: Optional[Point] = None,
        laser_yaw: Optional[float] = None,
    ) -> bool:
        if scan_id is not None and int(scan_id) == self._accepted_scan_id:
            return False
        if scan_id is not None:
            self._accepted_scan_id = int(scan_id)
        ox, oy = (laser_xy if laser_xy is not None else (float(x), float(y)))
        heading = float(yaw if laser_yaw is None else laser_yaw)
        increment = float(angle_increment) if angle_increment else (2.0 * math.pi / max(1, len(ranges)))
        hit_cells: Dict[int, None] = {}
        free_cells: Dict[int, None] = {}
        temp = bytearray(len(self.temp_block))
        for index, raw in enumerate(ranges):
            distance = float(raw)
            if math.isnan(distance) or distance < float(range_min):
                continue
            angle = wrap_to_pi(float(angle_min) + index * increment + heading)
            at_max = (not math.isfinite(distance)) or distance >= float(range_max) - 0.01
            if at_max:
                continue
            hx = ox + distance * math.cos(angle)
            hy = oy + distance * math.sin(angle)
            self._trace_ray(ox, oy, hx, hy, hit_cells, free_cells)
            col, row = self.world_to_cell(hx, hy)
            if self.in_grid(col, row):
                temp[self._index(col, row)] = 1
        changed = False
        occ_changed = False
        for idx in free_cells:
            if idx in hit_cells or self.static_solid[idx]:
                continue
            prev = self.score[idx]
            self.score[idx] = max(0, self.score[idx] - 1)
            self.observed[idx] = 1
            self.last_free[idx] = float(now)
            if self.score[idx] == 0:
                self.free_streak[idx] += 1
            if self.confirmed[idx] and self.score[idx] == 0 and self.free_streak[idx] >= 3:
                self.confirmed[idx] = 0
                occ_changed = True
            if prev != self.score[idx]:
                changed = True
        for idx in hit_cells:
            if self.static_solid[idx]:
                self.observed[idx] = 1
                continue
            col, row = idx % self.cols, idx // self.cols
            if self._near_physical(col, row):
                self.observed[idx] = 1
                continue
            prev = self.score[idx]
            self.score[idx] = min(SCORE_MAX, self.score[idx] + 2)
            self.last_hit[idx] = float(now)
            self.observed[idx] = 1
            self.free_streak[idx] = 0
            self.hit_scans[idx] += 1
            if self.score[idx] >= LETHAL_SCORE and self.hit_scans[idx] >= 2 and not self.confirmed[idx]:
                self.confirmed[idx] = 1
                occ_changed = True
            if prev != self.score[idx]:
                changed = True
        self.temp_block = temp
        expired = self._expire(now)
        if occ_changed or expired:
            self._bump_occ()
            self.rebuild_masks()
        elif changed:
            self._bump_map()
        return True

    def _trace_ray(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        hit_cells: Dict[int, None],
        free_cells: Dict[int, None],
    ) -> None:
        cells = self._supercover(x0, y0, x1, y1)
        if not cells:
            return
        for col, row in cells[:-1]:
            if not self.in_grid(col, row):
                return
            idx = self._index(col, row)
            if self.static_solid[idx]:
                return
            free_cells[idx] = None
        col, row = cells[-1]
        if self.in_grid(col, row):
            idx = self._index(col, row)
            if not self.static_solid[idx]:
                hit_cells[idx] = None

    def _supercover(self, x0: float, y0: float, x1: float, y1: float) -> List[Cell]:
        c0, r0 = self.world_to_cell(x0, y0)
        c1, r1 = self.world_to_cell(x1, y1)
        cells: List[Cell] = []
        dc = abs(c1 - c0)
        dr = abs(r1 - r0)
        sc = 1 if c0 < c1 else -1
        sr = 1 if r0 < r1 else -1
        err = dc - dr
        col, row = c0, r0
        for _ in range(dc + dr + 2):
            cells.append((col, row))
            if col == c1 and row == r1:
                break
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                col += sc
            if e2 < dc:
                err += dc
                row += sr
        return cells

    def _near_physical(self, col: int, row: int) -> bool:
        span = max(1, int(math.ceil(WALL_ECHO_M / self.res)))
        for dr in range(-span, span + 1):
            for dc in range(-span, span + 1):
                ncol, nrow = col + dc, row + dr
                if not self.in_grid(ncol, nrow):
                    continue
                if self.static_solid[self._index(ncol, nrow)]:
                    if math.hypot(dc * self.res, dr * self.res) <= WALL_ECHO_M + 1e-6:
                        return True
        return False

    def rebuild_masks(self, dynamic_inflate: Optional[float] = None) -> None:
        if dynamic_inflate is not None:
            self._dynamic_inflate = max(self.inflate, float(dynamic_inflate))
        radius = max(self.inflate, self._dynamic_inflate)
        n = self.cols * self.rows
        self.lethal = bytearray(self.static)
        for idx, flag in enumerate(self.center_forbidden):
            if flag:
                self.lethal[idx] = 1
        self.dynamic = bytearray(n)
        confirmed = [
            (idx % self.cols, idx // self.cols)
            for idx in range(n)
            if self.confirmed[idx] or self.score[idx] >= LETHAL_SCORE
        ]
        span = max(1, int(math.ceil((radius + self.res) / self.res)))
        for scol, srow in confirmed:
            sx0, sy0, sx1, sy1 = self.cell_rect(scol, srow)
            # Wall-face echoes must not grow a 0.30 halo onto the yellow lane.
            echo = self._near_physical(scol, srow)
            halo = 0.0 if echo else radius
            grow = 0 if echo else span
            for dr in range(-grow, grow + 1):
                for dc in range(-grow, grow + 1):
                    col, row = scol + dc, srow + dr
                    if not self.in_grid(col, row):
                        continue
                    cx, cy = self.cell_center(col, row)
                    if _circle_rect_dist(cx, cy, sx0, sy0, sx1, sy1) <= halo + 1e-6:
                        nidx = self._index(col, row)
                        self.lethal[nidx] = 1
                        if not self.static_solid[nidx] and not self.center_forbidden[nidx]:
                            self.dynamic[nidx] = 1

    def _expire(self, now: float) -> bool:
        changed = False
        for idx, score in enumerate(self.score):
            if self.static_solid[idx] or score <= 0:
                continue
            age = float(now) - self.last_hit[idx]
            if not self.confirmed[idx] and age > 5.0:
                self.score[idx] = 0
                self.hit_scans[idx] = 0
                self.observed[idx] = 0
                changed = True
        return changed

    def mark_circle(self, x: float, y: float, radius: float, now: float) -> None:
        lo_c, lo_r = self.world_to_cell(x - radius, y - radius)
        hi_c, hi_r = self.world_to_cell(x + radius, y + radius)
        for row in range(lo_r, hi_r + 1):
            for col in range(lo_c, hi_c + 1):
                if not self.in_grid(col, row):
                    continue
                cx, cy = self.cell_center(col, row)
                if math.hypot(cx - x, cy - y) > radius:
                    continue
                idx = self._index(col, row)
                if self.static_solid[idx]:
                    continue
                self.score[idx] = SCORE_MAX
                self.confirmed[idx] = 1
                self.observed[idx] = 1
                self.hit_scans[idx] = 2
                self.last_hit[idx] = float(now)
        self._bump_occ()
        self.rebuild_masks()

    def is_static(self, x: float, y: float) -> bool:
        col, row = self.world_to_cell(x, y)
        if not self.in_grid(col, row):
            return True
        return bool(self.static[self._index(col, row)])

    def is_physical(self, x: float, y: float) -> bool:
        col, row = self.world_to_cell(x, y)
        if not self.in_grid(col, row):
            return True
        return bool(self.static_solid[self._index(col, row)])

    def is_center_ban(self, x: float, y: float) -> bool:
        return in_center_forbidden(x, y)

    def is_lethal(self, col: int, row: int, inflate_dynamic: bool = True) -> bool:
        if not self.in_grid(col, row):
            return True
        idx = self._index(col, row)
        if self.static[idx] or self.center_forbidden[idx]:
            return True
        if inflate_dynamic:
            return bool(self.lethal[idx])
        return bool(self.confirmed[idx] or self.score[idx] >= LETHAL_SCORE)

    def is_dynamic(self, col: int, row: int) -> bool:
        if not self.in_grid(col, row):
            return False
        return bool(self.dynamic[self._index(col, row)])

    def confirmed_count(self) -> int:
        return int(sum(self.confirmed))

    def is_observed_free(self, col: int, row: int) -> bool:
        if not self.in_grid(col, row):
            return False
        idx = self._index(col, row)
        return bool(self.observed[idx]) and not self.lethal[idx]

    def pose_collision(self, x: float, y: float, radius: Optional[float] = None) -> str:
        if not in_nav_bounds(x, y):
            return "START_OUTSIDE_MAP"
        if in_physical_solid(x, y):
            return "START_IN_COLLISION"
        rad = self.inflate if radius is None else float(radius)
        overlap = 0.12
        near_obs = False
        col0, row0 = self.world_to_cell(x, y)
        span = max(1, int(math.ceil((rad + self.res) / self.res)))
        for dr in range(-span, span + 1):
            for dc in range(-span, span + 1):
                col, row = col0 + dc, row0 + dr
                if not self.in_grid(col, row):
                    continue
                idx = self._index(col, row)
                x0, y0, x1, y1 = self.cell_rect(col, row)
                dist = _circle_rect_dist(x, y, x0, y0, x1, y1)
                if self.static_solid[idx]:
                    if dist <= overlap + 1e-6:
                        return "START_IN_COLLISION"
                    if dist <= rad + 1e-6:
                        near_obs = True
                    continue
                if not self.confirmed[idx]:
                    continue
                if self._near_physical(col, row):
                    if dist <= rad + 1e-6:
                        near_obs = True
                    continue
                if dist <= overlap + 1e-6:
                    return "START_IN_COLLISION"
                if dist <= rad + 1e-6:
                    near_obs = True
        if in_center_forbidden(x, y):
            return "START_IN_FORBIDDEN"
        if near_obs:
            return "START_NEAR_OBSTACLE"
        return ""

    def segment_blocked(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        use_temp: bool = False,
    ) -> bool:
        if illegal_shortcut(x0, y0, x1, y1):
            return True
        length = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(math.ceil(length / (0.5 * self.res))))
        for step in range(steps + 1):
            t = step / steps
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            if in_center_forbidden(x, y):
                start_in = forbidden_intrusion_m(x0, y0)
                here_in = forbidden_intrusion_m(x, y)
                if start_in <= 1e-9 or here_in > start_in + 1e-6:
                    return True
            col, row = self.world_to_cell(x, y)
            if not self.in_grid(col, row):
                return True
            idx = self._index(col, row)
            if self.static[idx] or self.dynamic[idx] or self.confirmed[idx]:
                return True
            if use_temp and self.temp_block[idx]:
                return True
        return False


class GlobalGridPlanner:
    def __init__(self, grid: Optional[OccupancyGrid] = None) -> None:
        self.grid = grid if grid is not None else OccupancyGrid()
        self.last_result = PlanResult(status="EMPTY")

    def integrate_scan(self, *args, **kwargs) -> bool:
        return bool(self.grid.integrate_scan(*args, **kwargs))

    def plan(
        self,
        x: float,
        y: float,
        goal_xy: Sequence[float],
        dynamic_inflate: Optional[float] = None,
        yaw: float = 0.0,
        kind: Optional[str] = None,
        recover: bool = False,
    ) -> Optional[List[Point]]:
        result = self.plan_to(
            float(x),
            float(y),
            float(yaw),
            goal_xy,
            kind=kind,
            recover=recover,
            dynamic_inflate=dynamic_inflate,
        )
        if result.status != "OK":
            return None
        return list(result.path_xy)

    def plan_to(
        self,
        x: float,
        y: float,
        yaw: float,
        goal_xy: Sequence[float],
        *,
        kind: Optional[str] = None,
        recover: bool = False,
        dynamic_inflate: Optional[float] = None,
        intent: str = "travel",
    ) -> PlanResult:
        t0 = time_mod.perf_counter()
        # Production never shrinks below the S0 reference circle.
        self.grid.rebuild_masks(self.grid.inflate if dynamic_inflate is None else max(self.grid.inflate, float(dynamic_inflate)))
        goal_kind = kind or infer_goal_kind(goal_xy)
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        start_reason = self.grid.pose_collision(x, y)
        if start_reason == "START_OUTSIDE_MAP":
            return self._fail(start_reason, t0)
        if start_reason == "START_IN_COLLISION":
            return self._fail(start_reason, t0)
        if start_reason == "START_IN_FORBIDDEN" and not recover:
            return self._fail(start_reason, t0)

        start = self._start_cell(x, y, recover=recover)
        if start is None:
            return self._fail("START_CONNECTOR_BLOCKED", t0)
        if self.grid.segment_blocked(x, y, *self.grid.cell_center(*start)):
            if not recover:
                return self._fail("START_CONNECTOR_BLOCKED", t0)

        goal = self._goal_cell(gx, gy, goal_kind)
        if goal is None:
            return self._fail("GOAL_BLOCKED", t0)

        corridor = preferred_corridor((gx, gy), goal_kind)
        costs = self._cost_field(corridor)
        cells, cost, expanded = self._astar(start, goal, yaw, costs, t0)
        search_ms = 1000.0 * (time_mod.perf_counter() - t0)
        if cells is None:
            reason = "SEARCH_BUDGET_EXCEEDED" if search_ms >= SEARCH_BUDGET_MS else "NO_CONNECTIVITY"
            return self._fail(reason, t0, expanded=expanded, search_ms=search_ms)

        points = [self.grid.cell_center(col, row) for col, row in cells]
        if math.hypot(points[0][0] - x, points[0][1] - y) > 0.04:
            points = [(float(x), float(y)), *points]
        if self.grid.segment_blocked(points[-1][0], points[-1][1], gx, gy):
            return self._fail("PATH_VALIDATION_FAILED", t0, expanded=expanded, search_ms=search_ms)
        points.append((gx, gy))
        waypoints = plan_to_waypoints(points)
        if self.path_hits_lethal(waypoints):
            waypoints = points
        if self.path_hits_lethal([(x, y), *waypoints]):
            return self._fail("PATH_VALIDATION_FAILED", t0, expanded=expanded, search_ms=search_ms)
        turns = _count_large_turns(waypoints, yaw)
        prefix, unknown = known_safe_prefix_of(self.grid, waypoints)
        result = PlanResult(
            status="OK",
            path_xy=waypoints,
            start_connector=((x, y), waypoints[0]) if waypoints else None,
            accepted_goal=(gx, gy),
            map_version=self.grid.map_version,
            occupancy_version=self.grid.occupancy_version,
            cost=cost,
            estimated_travel_s=cost,
            large_turn_count=turns,
            known_safe_prefix=prefix,
            first_unknown_region=unknown,
            expanded_nodes=expanded,
            search_ms=search_ms,
            min_clearance=self.grid.inflate,
        )
        self.last_result = result
        return result

    def _fail(self, reason: str, t0: float, expanded: int = 0, search_ms: float = 0.0) -> PlanResult:
        result = PlanResult(
            status="FAIL",
            failure_reason=reason,
            map_version=self.grid.map_version,
            occupancy_version=self.grid.occupancy_version,
            expanded_nodes=expanded,
            search_ms=search_ms or 1000.0 * (time_mod.perf_counter() - t0),
        )
        self.last_result = result
        return result

    def _start_cell(self, x: float, y: float, recover: bool) -> Optional[Cell]:
        col, row = self.grid.world_to_cell(x, y)
        if self.grid.in_grid(col, row) and not self.grid.is_lethal(col, row):
            return (col, row)
        if recover and in_center_forbidden(x, y) and self.grid.pose_collision(x, y) != "START_IN_COLLISION":
            best = None
            best_key = None
            span = max(1, int(math.ceil(0.60 / self.grid.res)))
            here = forbidden_intrusion_m(x, y)
            for dr in range(-span, span + 1):
                for dc in range(-span, span + 1):
                    ncol, nrow = col + dc, row + dr
                    if not self.grid.in_grid(ncol, nrow) or self.grid.is_lethal(ncol, nrow):
                        continue
                    cx, cy = self.grid.cell_center(ncol, nrow)
                    nxt = forbidden_intrusion_m(cx, cy)
                    if nxt >= here - 1e-6:
                        continue
                    if self.grid.segment_blocked(x, y, cx, cy):
                        continue
                    key = (nxt, math.hypot(cx - x, cy - y))
                    if best_key is None or key < best_key:
                        best = (ncol, nrow)
                        best_key = key
            return best
        best = None
        best_d = 0.40 + 1.0
        span = max(1, int(math.ceil(0.40 / self.grid.res)))
        for dr in range(-span, span + 1):
            for dc in range(-span, span + 1):
                ncol, nrow = col + dc, row + dr
                if not self.grid.in_grid(ncol, nrow) or self.grid.is_lethal(ncol, nrow):
                    continue
                cx, cy = self.grid.cell_center(ncol, nrow)
                if self.grid.segment_blocked(x, y, cx, cy):
                    continue
                dist = math.hypot(cx - x, cy - y)
                if dist < best_d:
                    best = (ncol, nrow)
                    best_d = dist
        return best

    def _goal_cell(self, gx: float, gy: float, kind: str) -> Optional[Cell]:
        if in_physical_solid(gx, gy) or not in_nav_bounds(gx, gy):
            return None
        col, row = self.grid.world_to_cell(gx, gy)
        if self.grid.in_grid(col, row) and not self.grid.is_lethal(col, row) and goal_region_ok(kind, gx, gy):
            return (col, row)
        if in_center_forbidden(gx, gy):
            return None
        best = None
        best_d = 0.40 + 1.0
        span = max(1, int(math.ceil(0.40 / self.grid.res)))
        for dr in range(-span, span + 1):
            for dc in range(-span, span + 1):
                ncol, nrow = col + dc, row + dr
                if not self.grid.in_grid(ncol, nrow) or self.grid.is_lethal(ncol, nrow):
                    continue
                cx, cy = self.grid.cell_center(ncol, nrow)
                if not goal_region_ok(kind, cx, cy) and not goal_region_ok(kind, gx, gy):
                    continue
                if self.grid.segment_blocked(cx, cy, gx, gy):
                    continue
                dist = math.hypot(cx - gx, cy - gy)
                if dist < best_d:
                    best = (ncol, nrow)
                    best_d = dist
        return best

    def _cost_field(self, corridor: Sequence[Point]) -> List[float]:
        n = self.grid.cols * self.grid.rows
        costs = [0.0] * n
        for idx in range(n):
            col, row = idx % self.grid.cols, idx // self.grid.cols
            if not self.grid.in_grid(col, row):
                costs[idx] = 1e6
                continue
            base = FREE_COST if self.grid.observed[idx] else UNKNOWN_COST
            if self.grid.dynamic[idx]:
                base += 0.15
            x, y = self.grid.cell_center(col, row)
            # Soft ribbon only. 0.50 m off the preferred line already pays a
            # full extra free-cell, so a local box does not dump the robot
            # onto the outer wall.
            base += max(CORRIDOR_COST, 1.2) * min(1.0, _polyline_distance(x, y, corridor) / 0.50)
            costs[idx] = max(0.05, base)
        return costs

    def _astar(
        self,
        start: Cell,
        goal: Cell,
        yaw: float,
        costs: Sequence[float],
        t0: float,
    ) -> Tuple[Optional[List[Cell]], float, int]:
        cols, rows = self.grid.cols, self.grid.rows
        lethal = self.grid.lethal
        start_dir = _dir_from_yaw(yaw)
        gx, gy = self.grid.cell_center(*goal)
        res = self.grid.res
        origin_x, origin_y = self.grid.origin_x, self.grid.origin_y

        def heuristic(col: int, row: int) -> float:
            return math.hypot(
                gx - (origin_x + (col + 0.5) * res),
                gy - (origin_y + (row + 0.5) * res),
            ) / V_REF

        def blocked(col: int, row: int) -> bool:
            if not (0 <= col < cols and 0 <= row < rows):
                return True
            return bool(lethal[row * cols + col])

        start_state = (start[0], start[1], start_dir)
        open_heap = [(heuristic(*start), 0.0, start_state)]
        best = {start_state: 0.0}
        came: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {start_state: None}
        seen = 0
        while open_heap:
            if seen and seen % 256 == 0 and 1000.0 * (time_mod.perf_counter() - t0) >= SEARCH_BUDGET_MS:
                break
            _f, cost, state = heapq.heappop(open_heap)
            if cost > best.get(state, 1e18) + 1e-9:
                continue
            seen += 1
            col, row, heading = state
            if (col, row) == goal:
                path = []
                node: Optional[Tuple[int, int, int]] = state
                while node is not None:
                    path.append((node[0], node[1]))
                    node = came.get(node)
                path.reverse()
                cleaned: List[Cell] = [path[0]]
                for cell in path[1:]:
                    if cell != cleaned[-1]:
                        cleaned.append(cell)
                return cleaned, cost, seen
            if seen > 40000:
                break
            for nxt_dir, (dc, dr) in enumerate(DIRS):
                ncol, nrow = col + dc, row + dr
                if blocked(ncol, nrow):
                    continue
                if dc != 0 and dr != 0 and (blocked(col + dc, row) or blocked(col, row + dr)):
                    continue
                step = 1.41421356 * res if dc != 0 and dr != 0 else res
                new_cost = cost + step / V_REF * costs[nrow * cols + ncol] + _turn_cost(heading, nxt_dir)
                nxt = (ncol, nrow, nxt_dir)
                if new_cost >= best.get(nxt, 1e18):
                    continue
                best[nxt] = new_cost
                came[nxt] = state
                heapq.heappush(open_heap, (new_cost + heuristic(ncol, nrow), new_cost, nxt))
        return None, 0.0, seen

    def path_blocked_ahead(
        self,
        x: float,
        y: float,
        waypoints: Sequence[Point],
        horizon: float = 1.50,
        skip: float = PATH_SKIP_M,
    ) -> bool:
        if not waypoints:
            return False
        remaining = [(float(x), float(y)), *[(float(px), float(py)) for px, py in waypoints]]
        walked = 0.0
        for index in range(len(remaining) - 1):
            ax, ay = remaining[index]
            bx, by = remaining[index + 1]
            length = math.hypot(bx - ax, by - ay)
            if length < 1e-6:
                continue
            steps = max(1, int(math.ceil(length / self.grid.res)))
            for step in range(1, steps + 1):
                t = step / steps
                px, py = ax + t * (bx - ax), ay + t * (by - ay)
                walked += length / steps
                if walked < skip:
                    continue
                col, row = self.grid.world_to_cell(px, py)
                if not self.grid.in_grid(col, row):
                    return True
                idx = self.grid._index(col, row)
                # Confirmed/solid on the path. Inflate (dynamic) or this-scan
                # temp hits beside the path are not "this branch is dead".
                if self.grid.static_solid[idx] or self.grid.confirmed[idx]:
                    return True
                if walked >= horizon:
                    return False
        return False

    def path_hits_lethal(
        self,
        points: Sequence[Point],
        skip_start: float = 0.0,
    ) -> bool:
        if len(points) < 2:
            return False
        walked = 0.0
        for a, b in zip(points, points[1:]):
            ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
            if self.grid.segment_blocked(ax, ay, bx, by):
                return True
            length = math.hypot(bx - ax, by - ay)
            walked += length
            if walked <= skip_start:
                continue
        return False


def _count_large_turns(points: Sequence[Point], yaw: float) -> int:
    if len(points) < 2:
        return 0
    count = 0
    heading = float(yaw)
    prev = points[0]
    for point in points[1:]:
        nxt = math.atan2(point[1] - prev[1], point[0] - prev[0])
        if abs(wrap_to_pi(nxt - heading)) >= LARGE_TURN:
            count += 1
        heading = nxt
        prev = point
    return count


def _nearest_index(points: Sequence[Point], x: float, y: float) -> int:
    best_i = 0
    best_d = float("inf")
    for index, point in enumerate(points):
        dist = math.hypot(point[0] - x, point[1] - y)
        if dist < best_d:
            best_i = index
            best_d = dist
    return best_i


def _east_aisle_then_west(points: Sequence[Point], x: float, y: float) -> List[Point]:
    """Reach the yellow lane on the east side before cutting west.

    Must apply from spawn too. Skipping ``y < 1.70`` left the A* diagonal
    ``(1.9, 1.9) → (1.6, 2.4)``, which drove northwest into the rack corner
    (shelf_20260909_060912).
    """
    if float(x) < 1.35 or float(y) >= WEST_LANE_Y - 0.06:
        return list(points)
    later_west = any(px < float(x) - 0.30 and py > 1.70 for px, py in points)
    if not later_west:
        return list(points)
    entry = (float(x), float(WEST_LANE_Y))
    cleaned: List[Point] = [entry]
    for px, py in points:
        if px > float(x) - 0.08 and py < WEST_LANE_Y - 0.04:
            continue
        if math.hypot(px - entry[0], py - entry[1]) < 0.12:
            continue
        cleaned.append((float(px), float(py)))
    return cleaned


def known_safe_prefix_of(
    grid: OccupancyGrid, waypoints: Sequence[Point]
) -> Tuple[List[Point], Optional[Point]]:
    """Observed-free prefix. Unobserved tail is the first unknown region."""
    prefix: List[Point] = []
    unknown: Optional[Point] = None
    for point in waypoints:
        px, py = float(point[0]), float(point[1])
        col, row = grid.world_to_cell(px, py)
        if not grid.in_grid(col, row) or grid.is_lethal(col, row):
            break
        if not grid.is_observed_free(col, row):
            unknown = (px, py)
            break
        prefix.append((px, py))
    if not prefix and waypoints:
        prefix = [(float(waypoints[0][0]), float(waypoints[0][1]))]
        if unknown is None and len(waypoints) > 1:
            unknown = (float(waypoints[1][0]), float(waypoints[1][1]))
    return prefix, unknown


def plan_to_waypoints(
    points: Sequence[Point],
    spacing: float = 0.45,
    min_spacing: float = 0.25,
) -> List[Point]:
    if not points:
        return []
    simplified = [points[0]]
    for index in range(1, len(points) - 1):
        ax, ay = simplified[-1]
        bx, by = points[index]
        cx, cy = points[index + 1]
        v1 = (bx - ax, by - ay)
        v2 = (cx - bx, cy - by)
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        if dot < 0.985 or illegal_shortcut(ax, ay, cx, cy):
            simplified.append(points[index])
    simplified.append(points[-1])
    sampled = [simplified[0]]
    for index in range(len(simplified) - 1):
        ax, ay = simplified[index]
        bx, by = simplified[index + 1]
        if illegal_shortcut(ax, ay, bx, by):
            i0 = _nearest_index(points, ax, ay)
            i1 = _nearest_index(points, bx, by)
            if i1 < i0:
                i0, i1 = i1, i0
            for pt in points[i0 + 1 : i1 + 1]:
                if math.hypot(pt[0] - sampled[-1][0], pt[1] - sampled[-1][1]) >= 0.05:
                    sampled.append((float(pt[0]), float(pt[1])))
            continue
        length = math.hypot(bx - ax, by - ay)
        if length < min_spacing and index < len(simplified) - 2:
            continue
        steps = max(1, int(math.floor(length / spacing)))
        for step in range(1, steps + 1):
            t = min(1.0, (step * spacing) / length)
            px, py = ax + t * (bx - ax), ay + t * (by - ay)
            if math.hypot(px - sampled[-1][0], py - sampled[-1][1]) >= min_spacing:
                sampled.append((px, py))
        if math.hypot(bx - sampled[-1][0], by - sampled[-1][1]) >= min_spacing:
            sampled.append((bx, by))
    if math.hypot(sampled[-1][0] - points[-1][0], sampled[-1][1] - points[-1][1]) > 0.08:
        sampled.append((float(points[-1][0]), float(points[-1][1])))
    cleaned: List[Point] = []
    for point in sampled:
        if cleaned and math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) < 0.08:
            continue
        cleaned.append(point)
    return cleaned


@dataclass
class GridNavController:
    """Single navigation session: state machine is the only route writer."""

    goal_xy: Point
    final_yaw: float
    max_lin: float
    max_ang: float
    planner: GlobalGridPlanner = field(default_factory=GlobalGridPlanner)
    follower: Optional[WaypointFollower] = None
    plan_id: int = 0
    plan_attempt_id: int = 0
    last_status: str = "WAIT_SENSORS"
    last_plan_s: float = -100.0
    last_eval_s: float = -100.0
    last_plan_reason: str = ""
    last_plan_ms: float = 0.0
    stop_dist: float = STOP_DIST
    cruise_lin: float = 0.0
    explore_lin: float = 0.0
    goal_kind: str = ""
    state: str = "WAIT_SENSORS"
    _goal: Optional[NavGoal] = None
    _observe_goal: Optional[Point] = None
    _last_xy: Optional[Point] = None
    _moved_t: float = 0.0
    _off_t: float = 0.0
    _arrive_t: float = 0.0
    _turn_t: float = 0.0
    _recover_left: float = 0.0
    _recover_budget: float = 0.0
    _fail_plans: int = 0
    _last_scan_id: int = -1
    _started: bool = False
    _event: str = ""
    _observe_scans: int = 0
    _branch_visits: Dict[Point, int] = field(default_factory=dict)
    _known_prefix: List[Point] = field(default_factory=list)
    _turn_yaw: float = 0.0
    _face_block_t: float = 0.0
    _face_blocked: bool = False

    def __post_init__(self) -> None:
        if self.cruise_lin <= 0.0:
            self.cruise_lin = min(abs(self.max_lin), BASE_LIN * UNKNOWN_SPEED_SCALE)
        else:
            self.cruise_lin = min(abs(self.max_lin), abs(self.cruise_lin))
        if self.explore_lin <= 0.0:
            self.explore_lin = min(abs(self.max_lin), BASE_LIN * EXPLORE_SPEED_SCALE)
        else:
            self.explore_lin = min(abs(self.max_lin), abs(self.explore_lin))
        if not self.goal_kind:
            self.goal_kind = infer_goal_kind(self.goal_xy)
        if self._goal is None:
            self._goal = NavGoal(
                goal_id=self.goal_kind,
                kind=self.goal_kind,
                xy=self.goal_xy,
                final_yaw=self.final_yaw,
            )

    def set_goal(self, goal: NavGoal) -> None:
        self._goal = goal
        self.goal_xy = (float(goal.xy[0]), float(goal.xy[1]))
        self.final_yaw = float(goal.final_yaw)
        self.goal_kind = goal.kind
        self.follower = None
        self._known_prefix = []
        self._observe_goal = None
        self._arrive_t = 0.0
        self._recover_left = 0.0
        self._recover_budget = 0.0
        self._fail_plans = 0
        self._moved_t = 0.0
        self._off_t = 0.0
        self._turn_t = 0.0
        self._turn_yaw = 0.0
        self._face_block_t = 0.0
        self._face_blocked = False
        self._event = ""
        self.state = "PLAN"
        self.last_status = "set_goal"
        self.last_plan_s = -100.0

    def cancel(self) -> None:
        self.follower = None
        self._observe_goal = None
        self.state = "BLOCKED"
        self.last_status = "cancelled"

    def integrate(
        self,
        x: float,
        y: float,
        yaw: float,
        ranges: Optional[Sequence[float]],
        angle_min: float,
        angle_increment: float,
        now: float,
        scan_id: Optional[int] = None,
    ) -> None:
        if ranges is None:
            return
        if scan_id is not None and int(scan_id) == self._last_scan_id:
            return
        accepted = self.planner.integrate_scan(
            x, y, yaw, ranges, angle_min, angle_increment, now, scan_id=scan_id
        )
        if accepted and scan_id is not None:
            self._last_scan_id = int(scan_id)

    def _commit_route(
        self,
        route: Sequence[Point],
        x: float,
        y: float,
        reason: str,
        known_prefix: Optional[Sequence[Point]] = None,
    ) -> None:
        points = [(float(px), float(py)) for px, py in route]
        if not points:
            return
        if math.hypot(points[0][0] - x, points[0][1] - y) > 0.08:
            points = [(float(x), float(y)), *points]
        points = _east_aisle_then_west(points, x, y)
        self.follower = WaypointFollower(
            list(points),
            final_yaw=self.final_yaw,
            max_lin=self.max_lin,
            max_ang=self.max_ang,
            pos_tol=0.12,
            global_mode=True,
        )
        # 8× tightens in-place turn to 0.50 rad; a yellow-lane diagonal is ~0.51.
        self.follower.in_place_yaw = max(self.follower.in_place_yaw, 0.85)
        self.follower.drive_yaw = max(self.follower.drive_yaw, 0.45)
        self.follower.mode = "drive"
        if known_prefix:
            self._known_prefix = [(float(px), float(py)) for px, py in known_prefix]
        else:
            prefix, _unknown = known_safe_prefix_of(self.planner.grid, points)
            self._known_prefix = prefix
        self.plan_id += 1
        self.last_plan_reason = reason
        if self.last_plan_s < 0.0:
            self.last_plan_s = 0.0
        self._off_t = 0.0
        self._moved_t = 0.0
        self._turn_t = 0.0
        self._turn_yaw = 0.0
        self._face_block_t = 0.0
        self._face_blocked = False
        self._last_xy = (float(x), float(y))
        self._started = True
        self._fail_plans = 0
        self.state = "FOLLOW"

    def _path_clear_ahead_m(self, x: float, y: float) -> float:
        if self.follower is None:
            return 0.0
        points = [(float(x), float(y)), *self.follower.waypoints[self.follower.idx :]]
        total = 0.0
        for index in range(len(points) - 1):
            ax, ay = points[index]
            bx, by = points[index + 1]
            if self.planner.grid.segment_blocked(ax, ay, bx, by):
                break
            col, row = self.planner.grid.world_to_cell(bx, by)
            if self.planner.grid.is_lethal(col, row):
                break
            total += math.hypot(bx - ax, by - ay)
        return total

    def _verified_ahead_m(
        self,
        x: float,
        y: float,
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
    ) -> float:
        path_m = self._path_clear_ahead_m(x, y)
        sectors = scan_sectors(ranges, angle_min, angle_increment)
        scan_m = sectors.ahead if math.isfinite(sectors.ahead) else path_m
        # Unknown cells on the planned path are not a reason to crawl at
        # 0.11 m/s when the live cone is still 1.1 m (delivery_20260906_092828).
        if math.isfinite(scan_m) and scan_m > self.stop_dist + 0.20:
            return max(path_m, min(scan_m, 2.0))
        if not math.isfinite(scan_m):
            return path_m
        return min(path_m, scan_m)

    def _follow_speed_cap(
        self,
        x: float,
        y: float,
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
    ) -> float:
        if self.state in {"RECOVER", "OBSERVE", "ALIGN_GOAL", "WAIT_SENSORS"}:
            return self.cruise_lin
        verified = self._verified_ahead_m(x, y, ranges, angle_min, angle_increment)
        cap = self.explore_lin if self._observe_goal is not None else self.max_lin
        path_m = self._path_clear_ahead_m(x, y)
        if path_m < 0.40 and self._observe_goal is None:
            cap = min(cap, self.explore_lin)
        return speed_for_clearance(verified, cap, self.cruise_lin)

    def _path_bearing(self, points: Sequence[Point], x: float, y: float) -> Optional[float]:
        for px, py in points:
            if math.hypot(float(px) - float(x), float(py) - float(y)) > 0.18:
                return math.atan2(float(py) - float(y), float(px) - float(x))
        return None

    def _rejects_reverse(self, new_points: Sequence[Point], x: float, y: float) -> bool:
        if self.follower is None or self._observe_goal is not None:
            return False
        if self._face_blocked:
            return False
        old = self.follower.waypoints[self.follower.idx :]
        heading = self._path_bearing(old, x, y)
        nxt = self._path_bearing(new_points, x, y)
        if heading is None or nxt is None:
            return False
        return abs(wrap_to_pi(nxt - heading)) > 2.00

    def _pick_retreat(self, x: float, y: float) -> Optional[Point]:
        """Leave a face-blocked pocket toward the hug rail / last open aisle."""
        candidates = (
            (HUG_WEST_X, float(y) + 0.20),
            (HUG_WEST_X, min(float(y) + 0.90, float(WEST_LANE_Y))),
            (float(x), float(y) + 0.90),
            (-0.85, 0.55),
            (HUG_WEST_X, 0.60),
        )
        best: Optional[Point] = None
        best_key = None
        for cx, cy in candidates:
            if math.hypot(cx - float(x), cy - float(y)) < 0.35:
                continue
            if not in_nav_bounds(cx, cy) or in_physical_solid(cx, cy) or in_center_forbidden(cx, cy):
                continue
            col, row = self.planner.grid.world_to_cell(cx, cy)
            if not self.planner.grid.in_grid(col, row) or self.planner.grid.is_lethal(col, row):
                continue
            key = (abs(cx - HUG_WEST_X), abs(cy - float(y)))
            if best_key is None or key < best_key:
                best = (float(cx), float(cy))
                best_key = key
        return best

    def _path_heads_into_forward(self, x: float, y: float, yaw: float) -> bool:
        """True if the remaining route still aims into the current forward cone."""
        if self.follower is None or self.follower.idx >= len(self.follower.waypoints):
            return True
        target = None
        for px, py in self.follower.waypoints[self.follower.idx :]:
            if math.hypot(px - float(x), py - float(y)) > 0.18:
                target = (float(px), float(py))
                break
        if target is None:
            tx, ty = self.follower.waypoints[-1]
            if math.hypot(tx - float(x), ty - float(y)) <= 0.12:
                return False
            target = (float(tx), float(ty))
        bearing = math.atan2(target[1] - float(y), target[0] - float(x))
        return abs(wrap_to_pi(bearing - float(yaw))) < 0.70

    def _active_goal_xy(self) -> Point:
        if self._observe_goal is not None:
            return self._observe_goal
        return self.goal_xy

    def _search(self, x: float, y: float, yaw: float, reason: str, recover: bool = False) -> bool:
        self.plan_attempt_id += 1
        t0 = time_mod.perf_counter()
        target = self._active_goal_xy()
        kind = "inspection_approach" if self._observe_goal is not None else self.goal_kind
        result = self.planner.plan_to(x, y, yaw, target, kind=kind, recover=recover, intent=kind)
        self.last_plan_ms = 1000.0 * (time_mod.perf_counter() - t0)
        if result.status == "OK" and result.path_xy:
            if self._face_blocked and reason not in {"retreat", "start"}:
                nxt = self._path_bearing(result.path_xy, x, y)
                if nxt is not None and abs(wrap_to_pi(nxt - float(yaw))) < 0.70:
                    # Same pocket. A south-to-south replan is not leaving.
                    self.last_plan_reason = "face_still_blocked"
                    self.last_status = "face_still_blocked"
                    return False
            if reason != "start" and self._rejects_reverse(result.path_xy, x, y):
                # 7.7: keep the current branch; a nearer other-side path is
                # not evidence the first one is dead.
                self.last_plan_reason = "hold_branch"
                self.last_status = "hold_branch"
                self.state = "FOLLOW"
                return True
            self._commit_route(
                result.path_xy, x, y, reason, known_prefix=result.known_safe_prefix
            )
            self.last_status = f"replan {reason}"
            return True
        self._fail_plans += 1
        self.last_plan_reason = result.failure_reason or reason
        self.last_status = f"plan_fail {result.failure_reason}"
        return False

    def _pick_observe(self, x: float, y: float) -> Optional[Point]:
        best = None
        best_key = None
        for row in range(self.planner.grid.rows):
            for col in range(self.planner.grid.cols):
                if not self.planner.grid.is_observed_free(col, row):
                    continue
                unknown = False
                for dc, dr in DIRS:
                    ncol, nrow = col + dc, row + dr
                    if not self.planner.grid.in_grid(ncol, nrow):
                        continue
                    idx = self.planner.grid._index(ncol, nrow)
                    if not self.planner.grid.observed[idx] and not self.planner.grid.lethal[idx]:
                        unknown = True
                        break
                if not unknown:
                    continue
                cx, cy = self.planner.grid.cell_center(col, row)
                if in_center_forbidden(cx, cy):
                    continue
                visits = self._branch_visits.get((round(cx, 2), round(cy, 2)), 0)
                if visits >= 2:
                    continue
                dist = math.hypot(cx - x, cy - y)
                remain = math.hypot(self.goal_xy[0] - cx, self.goal_xy[1] - cy)
                key = (visits, dist + 0.25 * remain)
                if best_key is None or key < best_key:
                    best = (cx, cy)
                    best_key = key
        return best

    def _note_progress(self, x: float, y: float, dt: float, moving: bool) -> None:
        here = (float(x), float(y))
        if self._last_xy is None:
            self._last_xy = here
            self._moved_t = 0.0
            return
        if not moving:
            self._last_xy = here
            return
        moved = math.hypot(here[0] - self._last_xy[0], here[1] - self._last_xy[1])
        if moved >= PROGRESS_EPS_M:
            self._moved_t = 0.0
            self._last_xy = here
        else:
            self._moved_t += float(dt)
        if self.follower is None or self.follower.idx >= len(self.follower.waypoints):
            self._off_t = 0.0
            return
        remain = list(self.follower.waypoints[self.follower.idx :])
        if self.follower.idx > 0:
            remain.insert(0, self.follower.waypoints[self.follower.idx - 1])
        else:
            remain.insert(0, here)
        nearest = float("inf")
        for index in range(len(remain) - 1):
            nearest = min(nearest, _point_to_segment(x, y, *remain[index], *remain[index + 1]))
        self._off_t = self._off_t + float(dt) if nearest > OFF_TRACK_M else 0.0

    def _arrived(self, x: float, y: float, yaw: float, lin: float, ang: float, dt: float, now: float) -> bool:
        if self._observe_goal is not None:
            if math.hypot(x - self._observe_goal[0], y - self._observe_goal[1]) <= 0.18:
                key = (round(self._observe_goal[0], 2), round(self._observe_goal[1], 2))
                self._branch_visits[key] = self._branch_visits.get(key, 0) + 1
                self._observe_goal = None
                self.state = "PLAN"
                self.last_status = "observe_done"
            return False
        pos_ok = math.hypot(x - self.goal_xy[0], y - self.goal_xy[1]) <= 0.12
        yaw_ok = abs(wrap_to_pi(self.final_yaw - yaw)) <= 0.05
        region_ok = goal_region_ok(self.goal_kind, x, y)
        still = abs(lin) <= STILL_LIN_M_S and abs(ang) <= STILL_ANG_RAD_S
        if pos_ok and yaw_ok and region_ok and still:
            self._arrive_t += float(dt)
        else:
            self._arrive_t = 0.0
        return self._arrive_t >= ARRIVE_HOLD_S

    def step(
        self,
        x: float,
        y: float,
        yaw: float,
        ranges: Optional[Sequence[float]],
        angle_min: float,
        angle_increment: float,
        dt: float,
        now: float,
        scan_id: Optional[int] = None,
    ) -> Tuple[float, float, bool, str]:
        self.integrate(x, y, yaw, ranges, angle_min, angle_increment, now, scan_id=scan_id)
        if ranges is None:
            self.state = "WAIT_SENSORS"
            self.last_status = "WAIT_SENSORS"
            return 0.0, 0.0, False, self.last_status

        ahead = float("inf")
        sectors = scan_sectors(ranges, angle_min, angle_increment)
        if math.isfinite(sectors.ahead):
            ahead = sectors.ahead
        nose_on_box = (
            math.isfinite(ahead)
            and ahead <= self.stop_dist
            and self.follower is not None
            and self._path_heads_into_forward(x, y, yaw)
        )
        if nose_on_box:
            self._face_block_t += float(dt)
        else:
            self._face_block_t = 0.0
        self._face_blocked = nose_on_box or self._face_block_t >= FACE_BLOCK_S

        start_reason = self.planner.grid.pose_collision(x, y)
        if start_reason == "START_IN_COLLISION" and in_physical_solid(x, y):
            self.state = "BLOCKED"
            self.last_status = "START_IN_COLLISION"
            return 0.0, 0.0, False, self.last_status
        # Divider-echo inflate used to hard-stop here (delivery_20260906_090209).
        # A legal pose with a live path keeps following; lidar still gates hits.
        if start_reason == "START_IN_FORBIDDEN" and self.state not in {"RECOVER", "FOLLOW"}:
            self.state = "RECOVER"
            self._event = "CENTER_GUARD_VIOLATION"
            if self._recover_budget <= 0.0:
                self._recover_budget = 15.0

        if not self._started and self.state not in {"RECOVER", "BLOCKED", "FAILED"}:
            self.state = "PLAN"

        if self.state == "RECOVER":
            self._recover_budget = max(0.0, self._recover_budget - float(dt))
            if self._search(x, y, yaw, "recover_exit", recover=True):
                self.last_plan_s = now
                self._turn_t = 0.0
                self._turn_yaw = float(yaw)
                # Keep this tick: a zero publish here is what froze the yellow-lane turn.
            elif self._recover_budget <= 0.0:
                self.state = "BLOCKED"
                self.last_status = "BLOCKED recover"
                return 0.0, 0.0, False, self.last_status
            elif in_north_racks(x, y):
                south_err = wrap_to_pi(-math.pi / 2.0 - float(yaw))
                back = -min(0.45, 0.20 * abs(self.max_lin)) if math.sin(float(yaw)) > 0.12 else 0.0
                ang = max(-self.max_ang, min(self.max_ang, 1.8 * south_err))
                self.last_status = "RECOVER reverse"
                return back, ang, False, self.last_status
            else:
                self.last_status = "RECOVER CENTER_GUARD_VIOLATION"
                return 0.0, 0.0, False, self.last_status

        need = ""
        remain: Sequence[Point] = ()
        if self.follower is not None:
            remain = self.follower.waypoints[self.follower.idx :]
        blocked = bool(remain) and self.planner.path_blocked_ahead(x, y, remain)
        aligning = bool(remain) and not self._path_heads_into_forward(x, y, yaw)
        holding = (
            self.follower is not None
            and self.last_plan_s >= 0.0
            and now - self.last_plan_s < ROUTE_HOLD_S
        )
        if self.state in {"PLAN", "WAIT_SENSORS"} or not self._started:
            need = "start"
        elif blocked:
            need = "path_blocked"
        elif (
            math.isfinite(ahead)
            and ahead <= self.stop_dist
            and self._path_heads_into_forward(x, y, yaw)
        ):
            need = "scan_stop"
        elif self._off_t >= 0.50:
            need = "off_track"
        elif self._moved_t >= PROGRESS_WINDOW_S:
            need = "stuck"
        if need in {"path_blocked", "stuck", "off_track", "scan_stop"} and aligning:
            # Still turning onto the committed branch. A box in the current
            # cone is not "the other color is better".
            need = ""
        elif need in {"path_blocked", "stuck", "off_track"} and holding:
            need = ""

        if self.state == "BLOCKED":
            if self.follower is not None:
                self.state = "FOLLOW"
            elif now - self.last_plan_s >= FAIL_BACKOFF_S:
                if self._search(x, y, yaw, "blocked_retry", recover=True):
                    self.last_plan_s = now
                else:
                    return 0.0, 0.0, False, self.last_status
            else:
                return 0.0, 0.0, False, self.last_status

        if need:
            if need != "start" and self.state == "FOLLOW":
                self.state = "BRAKE_WAIT"
                self.last_status = f"BRAKE_WAIT {need}"
                if now - self.last_plan_s < REPLAN_MIN_S and self.last_plan_s >= 0.0:
                    return 0.0, 0.0, False, self.last_status
            can_search = self.last_plan_s < 0.0 or now - self.last_plan_s >= REPLAN_MIN_S
            if self._fail_plans >= 2:
                can_search = now - self.last_plan_s >= FAIL_BACKOFF_S
            if can_search or need == "start":
                self.state = "PLAN"
                if self._search(x, y, yaw, need or "start", recover=start_reason == "START_IN_FORBIDDEN"):
                    self.last_plan_s = now
                    if self.last_plan_reason == "hold_branch":
                        need = ""
                    else:
                        return 0.0, 0.0, False, self.last_status
                else:
                    self.last_plan_s = now
                    if self._face_blocked:
                        retreat = self._pick_retreat(x, y)
                        if retreat is not None:
                            self._observe_goal = retreat
                            if self._search(x, y, yaw, "retreat"):
                                self.last_plan_s = now
                                self._face_block_t = 0.0
                                self._face_blocked = False
                                return 0.0, 0.0, False, self.last_status
                            self._observe_goal = None
                    # delivery_20260906_092050: A* said NO_CONNECTIVITY at
                    # (-1.76, 0.36) with fwd=1.08 and 33 wps left. Keep them
                    # only while the nose is not already on a box.
                    if self.follower is not None and not self._face_blocked:
                        self.state = "FOLLOW"
                        self.last_status = f"keep_branch {self.last_plan_reason}"
                        need = ""
                    elif self._fail_plans >= 2:
                        observe = self._pick_observe(x, y)
                        if observe is not None:
                            self._observe_goal = observe
                            self.state = "EXPLORE"
                            self.last_status = "EXPLORE observe"
                            return 0.0, 0.0, False, self.last_status
                        self.state = "BLOCKED"
                        self.last_status = f"BLOCKED {self.last_plan_reason}"
                        return 0.0, 0.0, False, self.last_status
                    else:
                        self.state = "OBSERVE"
                        self._observe_scans = 0
                        self.last_status = f"OBSERVE {self.last_plan_reason}"
                        return 0.0, 0.0, False, self.last_status

        if self.state == "OBSERVE":
            self._observe_scans += 1
            if self._observe_scans >= 3:
                self.state = "PLAN"
            self.last_status = "OBSERVE"
            return 0.0, 0.0, False, self.last_status

        if self.state == "EXPLORE" and self._observe_goal is not None:
            if self._search(x, y, yaw, "explore", recover=False):
                self.last_plan_s = now
                return 0.0, 0.0, False, self.last_status
            self._observe_goal = None
            if self.follower is not None:
                self.state = "FOLLOW"
                self.last_status = "keep_branch explore_fail"
            else:
                self.state = "BLOCKED"
                self.last_status = "BLOCKED explore"
                return 0.0, 0.0, False, self.last_status

        if self.follower is None:
            self.state = "PLAN"
            self.last_status = "PLAN"
            return 0.0, 0.0, False, self.last_status

        cap = self._follow_speed_cap(x, y, ranges, angle_min, angle_increment)
        self.follower.max_lin = cap
        linear, angular, done = self.follower.step(x, y, yaw)
        event = getattr(self.follower, "last_guard_event", "") or ""
        if event:
            self.state = "RECOVER"
            self._event = event
            self._recover_budget = 15.0
            self.last_status = event
            return 0.0, 0.0, False, self.last_status

        moving = abs(linear) >= 0.08
        self._note_progress(x, y, dt, moving=moving)

        if done or (
            math.hypot(x - self.goal_xy[0], y - self.goal_xy[1]) <= 0.12
            and self._observe_goal is None
        ):
            self.state = "ALIGN_GOAL"
            yaw_err = wrap_to_pi(self.final_yaw - yaw)
            linear, angular = 0.0, max(-self.max_ang, min(self.max_ang, 1.8 * yaw_err))
            if abs(yaw_err) <= 0.05:
                linear, angular = 0.0, 0.0

        linear, angular, gate = sweep_gate(
            linear,
            angular,
            ranges,
            angle_min,
            angle_increment,
            stop_dist=stop_dist_for_speed(abs(linear), self.stop_dist),
        )
        if gate == "safety_stop":
            self.state = "BRAKE_WAIT"
            self.last_status = "safety_stop"
            if abs(linear) <= 0.04:
                return 0.0, angular, False, self.last_status
            return 0.0, 0.0, False, self.last_status
        if abs(linear) <= 0.04 and abs(angular) > 0.05:
            if abs(wrap_to_pi(float(yaw) - self._turn_yaw)) > 0.12:
                self._turn_t = 0.0
                self._turn_yaw = float(yaw)
            else:
                if self._turn_t <= 1e-9:
                    self._turn_yaw = float(yaw)
                self._turn_t += float(dt)
            # Keep publishing the turn. RECOVER here zeroed yaw rate and
            # immediately re-stalled after recover_exit (delivery_20260906_085558).
            if self._turn_t >= 2.0:
                self.last_status = "turn"
                self._turn_t = 0.0
        else:
            self._turn_t = 0.0
        if gate != "ok":
            self.last_status = gate
            return linear, angular, False, self.last_status

        if self._arrived(x, y, yaw, linear, angular, dt, now):
            self.state = "ARRIVED"
            self.last_status = "arrived"
            return 0.0, 0.0, True, self.last_status
        if self.state == "ALIGN_GOAL":
            self.last_status = "ALIGN_GOAL"
            return linear, angular, False, self.last_status
        self.state = "FOLLOW"
        self.last_status = "cruise" if abs(linear) > 0.04 else ("turn" if abs(angular) > 0.05 else "FOLLOW")
        return linear, angular, False, self.last_status
