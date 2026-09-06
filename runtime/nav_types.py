"""Shared P4 navigation data contracts. No ROS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class PoseSample:
    stamp_sim: float
    received_wall: float
    frame_id: str
    x: float
    y: float
    yaw: float
    linear_speed: float = 0.0
    angular_speed: float = 0.0
    child_frame_id: str = "base_link"


@dataclass(frozen=True)
class ScanSample:
    scan_id: int
    stamp_sim: float
    received_wall: float
    frame_id: str
    ranges: Tuple[float, ...]
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    time_increment: float = 0.0


@dataclass(frozen=True)
class NavGoal:
    goal_id: str
    kind: str
    xy: Point
    final_yaw: float
    xy_tolerance: float = 0.12
    yaw_tolerance: float = 0.05
    allowed_goal_region: Optional[dict] = None
    deadline_sim: Optional[float] = None
    intent: str = "travel"
    branch_id: str = ""


@dataclass
class PlanResult:
    status: str
    failure_reason: str = ""
    path_xy: List[Point] = field(default_factory=list)
    segments: List[dict] = field(default_factory=list)
    start_connector: Optional[Tuple[Point, Point]] = None
    accepted_goal: Optional[Point] = None
    map_version: int = 0
    occupancy_version: int = 0
    cost: float = 0.0
    estimated_travel_s: float = 0.0
    large_turn_count: int = 0
    known_safe_prefix: List[Point] = field(default_factory=list)
    first_unknown_region: Optional[Point] = None
    min_clearance: float = 0.0
    expanded_nodes: int = 0
    search_ms: float = 0.0
    unchanged: bool = False


@dataclass
class NavStepResult:
    linear: float
    angular: float
    state: str
    reason: str
    arrived: bool = False
    plan_id: int = 0
    plan_attempt_id: int = 0
    progress_s: float = 0.0
    goal_error: float = 0.0
    guard_event: str = ""


def infer_goal_kind(xy: Sequence[float]) -> str:
    """Classify a named approach by proximity. Never use sign(x)."""
    from runtime.scene_zones import DELIVERY_APPROACH_XY, SHELF_APPROACH_XY

    sx, sy = float(xy[0]), float(xy[1])
    if abs(sx - SHELF_APPROACH_XY[0]) <= 0.20 and abs(sy - SHELF_APPROACH_XY[1]) <= 0.20:
        return "shelf_approach"
    if abs(sx - DELIVERY_APPROACH_XY[0]) <= 0.20 and abs(sy - DELIVERY_APPROACH_XY[1]) <= 0.20:
        return "delivery_approach"
    return "inspection_approach"
