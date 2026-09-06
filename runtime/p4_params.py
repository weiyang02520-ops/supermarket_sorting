"""P4 导航参数：场景几何 + 规划/雷达/跑图可调项。

改数只改这份文件。裁判区几何的权威值仍在 ``scene_zones``（Server 抄来的
拣货/配送框），这里原样导出，并补上出生范围、货架本体、东西走廊等
P4 命名区域。不要在 ``grid_planner`` / ``p4_nav`` 里再写一套坐标。

世界系：+X 东，+Y 北。单位米、弧度。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from runtime.scene_zones import (
    CARRY_OUT_DIST_M,
    CENTER_WALL_X,
    CENTER_WALL_Y,
    DELIVERY_APPROACH_XY,
    DELIVERY_BASE_ZONE,
    DELIVERY_BOX_ZONE,
    DELIVERY_FACE_YAW,
    DELIVERY_TABLE_XY,
    DELIVERY_TARGET_XYZ,
    DIVIDER_NW_X,
    DIVIDER_NW_Y,
    DROP_Z_M,
    EAST_STUB_X_MIN,
    EAST_STUB_Y_MAX,
    HUG_WEST_X,
    NAV_BOUNDS,
    NORTH_RACK_Y,
    PICKING_ZONE,
    ROUTE_DELIVERY_TO_SHELF,
    ROUTE_SHELF_TO_DELIVERY,
    SETTLE_SPEED_M_S,
    SHELF_AISLE_ENTRY_XY,
    SHELF_AISLE_Y,
    SHELF_APPROACH_XY,
    SHELF_CORNER_XY,
    SHELF_CORNER_Y,
    SHELF_FACE_YAW,
    SOUTH_CROSS_Y,
    SOUTH_EAST_X_MIN,
    SOUTH_EAST_Y_MIN,
    SOUTH_PEEL_Y,
    SPAWN_XY,
    SPAWN_YAW,
    SPAWN_ZONE,
    TIME_LIMIT_S,
    UPRIGHT_TOL_DEG,
    WEST_LANE_Y,
    in_center_wall_band,
    in_delivery_base,
    in_delivery_box,
    in_east_shelf_stub,
    in_north_racks,
    in_picking_zone,
    in_south_east_stub,
    in_spawn_zone,
    near_divider_nw_corner,
    point_in_zone,
)

# ---------------------------------------------------------------------------
# 1. 场地区块（由 scene_zones 拼出，给规划/日志用的名字）
# ---------------------------------------------------------------------------

# 货架本体 A–E：黄线以北。5 组货架共用这一条北墙，仓里没有各组分开的 MJCF 框。
SHELF_BODY_ZONE = {
    "x": (float(NAV_BOUNDS["x"][0]), float(NAV_BOUNDS["x"][1])),
    "y": (float(NORTH_RACK_Y), float(NAV_BOUNDS["y"][1])),
}
# 货架前拣货通道（黄线走廊）。再北就是货架本体。
PICKING_AISLE_ZONE = PICKING_ZONE
# 东通道：隔墙以东、货架以南，出生后北上走这里。
EAST_CORRIDOR_ZONE = {
    "x": (float(CENTER_WALL_X[1]), float(NAV_BOUNDS["x"][1])),
    "y": (float(SOUTH_EAST_Y_MIN), float(NORTH_RACK_Y)),
}
# 西走廊：贴隔墙内侧轨以南到剥离点。随机箱多在更西的外车道。
WEST_CORRIDOR_ZONE = {
    "x": (float(NAV_BOUNDS["x"][0]), float(HUG_WEST_X)),
    "y": (float(SOUTH_PEEL_Y), float(WEST_LANE_Y)),
}
# 西侧外车道（箱子常出现的一带，不是必须走的路）。
WEST_BOX_LANE_ZONE = {
    "x": (float(NAV_BOUNDS["x"][0]), -0.70),
    "y": (float(SOUTH_PEEL_Y), float(WEST_LANE_Y)),
}
# 中间隔墙实体。
CENTER_WALL_ZONE = {"x": CENTER_WALL_X, "y": CENTER_WALL_Y}
# 隔墙西北角：路径检查用，不画成静态墙。
DIVIDER_NW_ZONE = {"x": DIVIDER_NW_X, "y": DIVIDER_NW_Y}
# 配送台作业区 / 桌面放置框（存货投放）。
DELIVERY_WORK_ZONE = DELIVERY_BASE_ZONE
DELIVERY_PLACE_ZONE = DELIVERY_BOX_ZONE

# ---------------------------------------------------------------------------
# 2. 栅格与 A*
# ---------------------------------------------------------------------------

GRID_RES = 0.10
STATIC_INFLATE_M = 0.30
INFLATE_M = 0.22
RELAXED_INFLATE_M = 0.18
SELF_RANGE = 0.25
HIT_MIN = 0.25
HIT_MAX = 8.0
LETHAL_SCORE = 3
SCORE_MAX = 5
UNKNOWN_COST = 2.5
FREE_COST = 1.0
EDGE_COST = 0.15
CORRIDOR_COST = 0.35
TURN_COST = 0.15
CORRIDOR_WIDTH = 0.50
WALL_ECHO_M = 0.16
PATH_SKIP_M = 0.30
YELLOW_SOUTH_Y = WEST_LANE_Y - 0.18

# ---------------------------------------------------------------------------
# 3. 雷达反应（正前方清空距离随速度：stop + 0.30×v）
# ---------------------------------------------------------------------------

STOP_DIST = 0.43
AHEAD_CONE_HALF = 0.22
FORWARD_CONE_HALF = 0.40
CLEAR_S = 0.30
NUDGE_SPAN_MAX = 0.32
NUDGE_FAR_M = 0.90
SIDE_OPEN_M = 0.80
JAM_EXTRA_M = 0.10

# ---------------------------------------------------------------------------
# 4. 跟线 / 重规划
# ---------------------------------------------------------------------------

MAP_READY_HITS = 8
PLAN_HOLD_S = 1.50
REPLAN_MIN_S = 0.50
FAIL_BACKOFF_S = 2.00
OFF_TRACK_M = 0.35
PROGRESS_WINDOW_S = 1.50
PROGRESS_EPS_M = 0.05
PEEK_S = 0.90
PEEK_RAD = 0.55

# ---------------------------------------------------------------------------
# 5. 跑图入口
# ---------------------------------------------------------------------------

RATE_HZ = 20.0
BASE_LIN = 0.30
BASE_ANG = 0.55
SPEED_SCALE = 4.0
MAX_ANG_CAP = 1.65
HEAD_SETTLE_S = 1.0
DEFAULT_TIMEOUT_S = 150.0
LOG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "p4"


def in_shelf_body(x: float, y: float) -> bool:
    return point_in_zone((x, y), SHELF_BODY_ZONE)


def in_east_corridor(x: float, y: float) -> bool:
    return point_in_zone((x, y), EAST_CORRIDOR_ZONE)


def in_west_corridor(x: float, y: float) -> bool:
    return point_in_zone((x, y), WEST_CORRIDOR_ZONE)


def as_dict() -> Dict[str, Any]:
    """Snapshot for logs / last.json. Functions are omitted."""
    return {
        "frame": {"x_east": True, "y_north": True, "unit": "m"},
        "spawn_xy": list(SPAWN_XY),
        "spawn_yaw": SPAWN_YAW,
        "spawn_zone": SPAWN_ZONE,
        "shelf_body": SHELF_BODY_ZONE,
        "picking_aisle": PICKING_AISLE_ZONE,
        "shelf_approach_xy": list(SHELF_APPROACH_XY),
        "east_corridor": EAST_CORRIDOR_ZONE,
        "west_corridor": WEST_CORRIDOR_ZONE,
        "west_box_lane": WEST_BOX_LANE_ZONE,
        "center_wall": CENTER_WALL_ZONE,
        "divider_nw": DIVIDER_NW_ZONE,
        "delivery_work": DELIVERY_WORK_ZONE,
        "delivery_place": DELIVERY_PLACE_ZONE,
        "delivery_approach_xy": list(DELIVERY_APPROACH_XY),
        "delivery_table_xy": list(DELIVERY_TABLE_XY),
        "nav_bounds": NAV_BOUNDS,
        "hug_west_x": HUG_WEST_X,
        "west_lane_y": WEST_LANE_Y,
        "south_peel_y": SOUTH_PEEL_Y,
        "grid_res": GRID_RES,
        "inflate_m": INFLATE_M,
        "static_inflate_m": STATIC_INFLATE_M,
        "stop_dist": STOP_DIST,
        "map_ready_hits": MAP_READY_HITS,
        "rate_hz": RATE_HZ,
        "speed_scale": SPEED_SCALE,
        "time_limit_s": TIME_LIMIT_S,
    }
