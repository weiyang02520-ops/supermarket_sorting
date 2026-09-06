"""S0 geometry catalog: physical solids, center-forbidden, soft corridors.

Numbers still live in ``scene_zones`` / ``p4_params``. This module is the
semantic layer: every named region is tagged with ``kind``, ``source``,
``frame``, and a verification status. Unverified items stay unverified;
they are not turned into “safe” constants by loosening radii.

World frame: +X east, +Y north, metres. Planning uses the official odom
plane once ``frame_id`` checks pass.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from runtime.scene_zones import (
    CENTER_WALL_X,
    CENTER_WALL_Y,
    DELIVERY_APPROACH_XY,
    DELIVERY_BASE_ZONE,
    DELIVERY_FACE_YAW,
    DELIVERY_TABLE_XY,
    DIVIDER_NW_X,
    DIVIDER_NW_Y,
    HUG_WEST_X,
    NAV_BOUNDS,
    NORTH_RACK_Y,
    ROUTE_DELIVERY_TO_SHELF,
    ROUTE_SHELF_TO_DELIVERY,
    SHELF_APPROACH_XY,
    SHELF_CORNER_XY,
    SHELF_FACE_YAW,
    SOUTH_CROSS_Y,
    SOUTH_EAST_X_MIN,
    SOUTH_EAST_Y_MIN,
    SOUTH_PEEL_Y,
    SPAWN_XY,
    SPAWN_YAW,
    WEST_LANE_Y,
    in_center_wall_band,
    in_delivery_base,
    in_north_racks,
    in_south_east_stub,
    in_south_gap as _in_south_gap,
    near_divider_nw_corner,
    point_in_zone,
)

# Bump when solids / frames / footprint meaning change. Invalidates old paths.
GEOMETRY_VERSION = "s0-20260906"

# Planning circle. 0.30 m is the old starting reference, not a measured chassis.
R_BODY_UNVERIFIED_M = 0.30
MARGIN_TRACKING_UNVERIFIED_M = 0.00
R_PLAN_REF_M = R_BODY_UNVERIFIED_M + MARGIN_TRACKING_UNVERIFIED_M

# South gap: bidirectional centre ban under the divider, not a west/east ban.
SOUTH_GAP_ZONE = {
    "x": (0.05, float(CENTER_WALL_X[1])),
    "y": (float(NAV_BOUNDS["y"][0]), float(SOUTH_CROSS_Y) + 0.10),
}

# Laser in base. Official TF is required at runtime. Identity is only an
# explicit last-resort config and is not claimed to be the Server model.
LASER_FRAME = "laser"
BASE_FRAME = "base_link"
ODOM_FRAME = "odom"
LASER_EXTRINSIC = {
    "kind": "extrinsic",
    "source": "unverified",
    "status": "TF_REQUIRED",
    "frame_parent": BASE_FRAME,
    "frame_child": LASER_FRAME,
    "translation_xy": (0.0, 0.0),
    "yaw": 0.0,
    "note": (
        "Repo has no official T_base_laser. Runtime must look up TF; "
        "unit tests may pass an explicit identity. Do not silently guess zero."
    ),
}

SCAN_NO_ECHO = {
    "kind": "sensor_semantics",
    "source": "unverified",
    "status": "UNVERIFIED",
    "range_max_is_free": False,
    "inf_is_free": False,
    "note": (
        "Official LaserScan no-echo meaning is not in this repo. "
        "Treat +inf / range_max as unknown until a Server publisher check."
    ),
}

CLOCK_POLICY = {
    "kind": "clock",
    "source": "unverified",
    "status": "UNVERIFIED",
    "prefer": "/clock then message stamp",
    "note": (
        "Motion, evidence ageing, and the 420 s budget use sim time. "
        "Watchdogs and search budgets use wall time. /clock is not verified here."
    ),
}

CHASSIS_ENVELOPE = {
    "kind": "footprint",
    "source": "unverified",
    "status": "UNVERIFIED",
    "model": "circle",
    "r_body_m": R_BODY_UNVERIFIED_M,
    "margin_tracking_m": MARGIN_TRACKING_UNVERIFIED_M,
    "r_plan_m": R_PLAN_REF_M,
    "note": (
        "Official MMK2 collision geoms are not in this repo (only head FK). "
        "0.30 m is a starting reference. 0.22 / 0.18 m are not measurements. "
        "Do not shrink the circle to open a path."
    ),
}


def _item(
    name: str,
    kind: str,
    source: str,
    status: str,
    *,
    zone: Optional[dict] = None,
    xy: Optional[Sequence[float]] = None,
    yaw: Optional[float] = None,
    note: str = "",
    inflate: bool = False,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "name": name,
        "kind": kind,
        "source": source,
        "status": status,
        "frame": "odom",
        "inflate": bool(inflate),
        "note": note,
    }
    if zone is not None:
        item["zone"] = {
            "x": (float(zone["x"][0]), float(zone["x"][1])),
            "y": (float(zone["y"][0]), float(zone["y"][1])),
        }
    if xy is not None:
        item["xy"] = (float(xy[0]), float(xy[1]))
    if yaw is not None:
        item["yaw"] = float(yaw)
    return item


def geometry_catalog() -> Dict[str, Any]:
    """Traceable S0 table. status is verified | behavior | inferred | unverified."""
    physical = [
        _item(
            "center_wall",
            "physical_solid",
            "scene_zones.CENTER_WALL_* from collision logs / MJCF comments",
            "inferred",
            zone={"x": CENTER_WALL_X, "y": CENTER_WALL_Y},
            inflate=True,
            note="Divider slab. Inflate with R_plan. North face y=1.90 is log-inferred, not a surveyed surface.",
        ),
    ]
    forbidden = [
        _item(
            "north_racks",
            "center_forbidden",
            "scene_zones.NORTH_RACK_Y",
            "behavior",
            zone={"x": NAV_BOUNDS["x"], "y": (NORTH_RACK_Y, NAV_BOUNDS["y"][1])},
            inflate=False,
            note=(
                "Robot-centre threshold y>=2.48, not a measured shelf face. "
                "SHELF_APPROACH_XY.y=2.475 is 5 mm south of this line. "
                "If 2.48 were a physical face, a 0.30 m radius robot could not sit at 2.475. "
                "Do not Minkowski-inflate this band; do not carve a hole to make the approach free."
            ),
        ),
        _item(
            "south_gap",
            "center_forbidden",
            "scene_zones.SOUTH_CROSS_Y + divider x-span",
            "behavior",
            zone=SOUTH_GAP_ZONE,
            inflate=False,
            note=(
                "Bidirectional ban on crossing under the divider. "
                "Does not forbid the west corridor to delivery or east-corridor southbound."
            ),
        ),
        _item(
            "south_east_stub",
            "center_forbidden",
            "scene_zones.SOUTH_EAST_*",
            "behavior",
            zone={
                "x": (SOUTH_EAST_X_MIN, NAV_BOUNDS["x"][1]),
                "y": (NAV_BOUNDS["y"][0], SOUTH_EAST_Y_MIN),
            },
            inflate=False,
            note="Keep the base out of the south-east wall pocket. Not a surveyed outer wall.",
        ),
        _item(
            "divider_nw_corner",
            "center_forbidden",
            "scene_zones.DIVIDER_NW_*",
            "behavior",
            zone={"x": DIVIDER_NW_X, "y": DIVIDER_NW_Y},
            inflate=False,
            note="Local corner wedge. Path check only; not painted as a second inflated wall.",
        ),
    ]
    soft = [
        _item(
            "west_lane",
            "preferred_corridor",
            "scene_zones.WEST_LANE_Y / HUG_WEST_X / SOUTH_PEEL_Y / ROUTE_*",
            "behavior",
            note="Soft cost ribbon. Never paints free and never overrides occupied cells.",
        ),
    ]
    bounds = [
        _item(
            "nav_bounds",
            "search_bounds",
            "scene_zones.NAV_BOUNDS",
            "behavior",
            zone=NAV_BOUNDS,
            note="Planner AABB. Not physical outer walls. Cells outside the exact box are illegal.",
        ),
    ]
    goals = [
        _item(
            "shelf_approach",
            "nav_goal",
            "scene_zones.SHELF_APPROACH_XY",
            "behavior",
            xy=SHELF_APPROACH_XY,
            yaw=SHELF_FACE_YAW,
            note="Task approach for the yellow aisle. Valid because north_racks is a centre threshold, not a physical face.",
        ),
        _item(
            "delivery_approach",
            "nav_goal",
            "scene_zones.DELIVERY_APPROACH_XY",
            "behavior",
            xy=DELIVERY_APPROACH_XY,
            yaw=DELIVERY_FACE_YAW,
            note="Must finish inside delivery_base. Table body is not a goal-cell obstacle.",
        ),
        _item(
            "spawn",
            "nav_start",
            "scene_zones.SPAWN_XY",
            "inferred",
            xy=SPAWN_XY,
            yaw=SPAWN_YAW,
            note="Official spawn in the south-east. Not the start of every later plan.",
        ),
        _item(
            "delivery_table",
            "reference",
            "MJCF comment: delivery_table / delivery_target",
            "inferred",
            xy=DELIVERY_TABLE_XY,
            note="Referee table, not a planning solid in the first version.",
        ),
    ]
    return {
        "geometry_version": GEOMETRY_VERSION,
        "frame": {"x_east": True, "y_north": True, "unit": "m", "default": "odom"},
        "physical_solids": physical,
        "center_forbidden": forbidden,
        "preferred_corridors": soft,
        "search_bounds": bounds,
        "named_poses": goals,
        "chassis": CHASSIS_ENVELOPE,
        "laser": LASER_EXTRINSIC,
        "scan_no_echo": SCAN_NO_ECHO,
        "clock": CLOCK_POLICY,
        "conflicts": [
            {
                "name": "north_rack_vs_shelf_approach",
                "a": "NORTH_RACK_Y=2.48",
                "b": "SHELF_APPROACH_XY.y=2.475",
                "delta_m": 0.005,
                "resolution": (
                    "Treat 2.48 as a robot-centre behaviour line. "
                    "Treat 2.475 as a named approach. "
                    "Do not inflate 2.48 as a shelf face and do not carve the approach cell."
                ),
            },
            {
                "name": "divider_north_gap_vs_circle",
                "a": "CENTER_WALL_Y[1]=1.90",
                "b": "NORTH_RACK_Y=2.48",
                "gap_m": 0.58,
                "resolution": (
                    "A 0.60 m diameter circle does not fit if both edges are physical faces. "
                    "Only the divider rectangle is a solid; the rack line is centre-forbidden. "
                    "Unverified gap. Do not open it by clearing inflation."
                ),
            },
        ],
        "unverified": [
            "actual chassis collision radius",
            "fixed-structure physical surfaces beyond the inferred divider box",
            "T_base_laser",
            "T_world_odom (assumed identity with odom until TF is checked)",
            "MuJoCo /clock vs message stamp alignment",
            "LaserScan +inf / range_max no-echo meaning",
        ],
    }


def in_nav_bounds(x: float, y: float) -> bool:
    return point_in_zone((float(x), float(y)), NAV_BOUNDS)


def in_south_gap(x: float, y: float) -> bool:
    return _in_south_gap(x, y)


def in_physical_solid(x: float, y: float) -> bool:
    return in_center_wall_band(float(x), float(y))


def in_center_forbidden(x: float, y: float) -> bool:
    return (
        in_north_racks(float(x), float(y))
        or in_south_gap(float(x), float(y))
        or in_south_east_stub(float(x), float(y))
    )


def forbidden_intrusion_m(x: float, y: float) -> float:
    """How far a centre sits into a centre-forbidden band. 0 if outside."""
    px, py = float(x), float(y)
    worst = 0.0
    if in_north_racks(px, py):
        worst = max(worst, py - float(NORTH_RACK_Y))
    if in_south_gap(px, py):
        worst = max(worst, float(SOUTH_GAP_ZONE["y"][1]) - py)
    if in_south_east_stub(px, py):
        worst = max(worst, float(SOUTH_EAST_Y_MIN) - py)
    if near_divider_nw_corner(px, py) and not in_physical_solid(px, py):
        cx = min(abs(px - DIVIDER_NW_X[0]), abs(px - DIVIDER_NW_X[1]))
        cy = min(abs(py - DIVIDER_NW_Y[0]), abs(py - DIVIDER_NW_Y[1]))
        worst = max(worst, 0.01)
        _ = (cx, cy)
    return worst


def preferred_corridor_line(kind: str) -> List[Tuple[float, float]]:
    """Soft cost polyline. Kind is explicit; never sign(x)."""
    east = (float(SHELF_CORNER_XY[0]), float(WEST_LANE_Y))
    if kind == "delivery_approach":
        line = [east, *ROUTE_SHELF_TO_DELIVERY]
    elif kind == "shelf_approach":
        line = [(east[0], float(SPAWN_XY[1])), east, *ROUTE_DELIVERY_TO_SHELF, tuple(SHELF_APPROACH_XY)]
    else:
        line = [east, *ROUTE_DELIVERY_TO_SHELF]
    cleaned: List[Tuple[float, float]] = []
    for point in line:
        xy = (float(point[0]), float(point[1]))
        if cleaned and ((xy[0] - cleaned[-1][0]) ** 2 + (xy[1] - cleaned[-1][1]) ** 2) < 0.0025:
            continue
        cleaned.append(xy)
    return cleaned


def goal_region_ok(kind: str, x: float, y: float) -> bool:
    if kind == "delivery_approach":
        return in_delivery_base((float(x), float(y)))
    if kind == "shelf_approach":
        return (float(y) >= 1.70) and (float(y) < float(NORTH_RACK_Y)) and in_nav_bounds(x, y)
    return in_nav_bounds(x, y)


def catalog_item(name: str) -> Optional[Dict[str, Any]]:
    cat = geometry_catalog()
    for group in ("physical_solids", "center_forbidden", "preferred_corridors", "search_bounds", "named_poses"):
        for item in cat[group]:
            if item["name"] == name:
                return item
    return None
