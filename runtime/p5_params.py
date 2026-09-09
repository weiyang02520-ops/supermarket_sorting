"""P5 grasp/place templates. Referee table/zones stay in scene_zones."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from runtime.head_camera_kinematics import AISLE_SCAN_HEAD, SLIDE_LIMITS
from runtime.scene_zones import (
    CARRY_OUT_DIST_M,
    DELIVERY_APPROACH_XY,
    DELIVERY_FACE_YAW,
    DELIVERY_TABLE_XY,
    DELIVERY_TARGET_XYZ,
    SETTLE_SPEED_M_S,
    SHELF_AISLE_Y,
    SHELF_APPROACH_XY,
    SHELF_FACE_YAW,
)

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs" / "p5"
DEFAULT_WEIGHTS = ROOT / "weights" / "kele.pt"
DEFAULT_KIND = "kele"
RECIPE_VERSION = "kele-v1"

GRASP_ARM = "r"
# Aisle / L2 look. Official close-creep pitch -0.6 stares at the floor from
# the shelf approach (~1 m), so C1 must not use it.
HEAD_YAW, HEAD_PITCH = float(AISLE_SCAN_HEAD[0]), -0.28
GRASP_HEAD_PITCH = -0.6
GRASP_ROT = np.eye(3)
T_EG = np.eye(4)

# G2 parallel gripper: command 0–1 maps to about 0–80 mm inner opening.
GRIPPER_OPEN_GAP_M = 0.080
GRIP_OPEN = 1.0
GRIP_CLOSE = 0.08

# Place candidates inside delivery_box, north-to-south table at y≈-3.41.
PLACE_SLOT_XY: Tuple[Tuple[float, float], ...] = (
    (-1.940, -3.410),
    (-1.820, -3.410),
    (-2.060, -3.410),
)
PLACE_OCCUPIED_RADIUS = 0.10


@dataclass(frozen=True)
class GraspRecipe:
    """Calibrated pick/place template. Positions still come from YOLO+depth."""

    kind: str
    version: str
    enabled: bool
    diameter: float
    height: float
    radius: float
    z_min: float
    z_max: float
    fwd_min: float
    fwd_max: float
    lateral_max: float
    z_bias: float
    slide: float
    pregrasp_along: float
    pregrasp_lateral: float
    pregrasp_z: float
    creep_stop_along: float
    grip_open: float
    grip_close: float
    lift_clearance: float
    close_hold_s: float
    settle_evidence_s: float
    two_frame_pos_tol: float
    associate_tol: float
    rgb_depth_skew_max: float
    follow_tol: float
    carry_out_min: float
    h_bottom_from_g: float
    release_clearance: float
    table_support_z: float
    creep_speed: float
    retreat_speed: float
    lateral_error_bound: float
    entry_margin: float
    actual_open_gap: float
    place_stable_s: float
    place_stable_speed: float
    dt_jump_s: float
    max_close_attempts: int
    place_above: float = 0.07
    carry_slide: float = -0.04
    grasp_rot: np.ndarray = field(default_factory=lambda: np.eye(3))
    t_eg: np.ndarray = field(default_factory=lambda: np.eye(4))


def _kele_recipe() -> GraspRecipe:
    height = 0.145
    return GraspRecipe(
        kind="kele",
        version=RECIPE_VERSION,
        enabled=True,
        diameter=0.053,
        height=height,
        radius=0.0265,
        z_min=0.70,
        z_max=1.15,
        fwd_min=0.30,
        fwd_max=1.50,
        lateral_max=0.20,
        # Vision center sat ~4 cm above MJCF cola; lower the pinch.
        z_bias=-0.035,
        slide=0.11,
        pregrasp_along=0.220,
        # Along robot forward, not world -Y. A lateral bias shifts X on
        # a yawed approach and the fingers hit the can side-on.
        pregrasp_lateral=0.0,
        pregrasp_z=-0.010,
        # T_EG is still I, so wrap 3.5 cm past vision center (official
        # CREEP_STOP_DY). Fold this into T_EG once fingertips are calibrated.
        creep_stop_along=-0.035,
        grip_open=GRIP_OPEN,
        grip_close=GRIP_CLOSE,
        lift_clearance=0.020,
        close_hold_s=0.80,
        settle_evidence_s=0.10,
        two_frame_pos_tol=0.040,
        associate_tol=0.080,
        rgb_depth_skew_max=0.05,
        # In-hand YOLO+depth vs FK EE is noisy (T_EG=I, z_bias, gripper occlusion).
        follow_tol=0.18,
        carry_out_min=CARRY_OUT_DIST_M,
        h_bottom_from_g=0.5 * height,
        release_clearance=0.004,
        table_support_z=float(DELIVERY_TARGET_XYZ[2]),
        creep_speed=0.06,
        retreat_speed=0.20,
        lateral_error_bound=0.008,
        entry_margin=0.004,
        actual_open_gap=GRIPPER_OPEN_GAP_M,
        place_stable_s=1.0,
        place_stable_speed=SETTLE_SPEED_M_S,
        dt_jump_s=0.20,
        max_close_attempts=2,
        place_above=0.07,
        carry_slide=float(SLIDE_LIMITS[0]),
        grasp_rot=np.eye(3),
        t_eg=np.eye(4),
    )


KELE_RECIPE = _kele_recipe()

# Later kinds share the flow but stay closed until a calibrated template exists.
_PLACEHOLDER: Dict[str, Dict[str, float]] = {
    "kouxiangtang": {"diameter": 0.049, "height": 0.080},
    "shupian": {"diameter": 0.065, "height": 0.210},
    "maidong": {"diameter": 0.065, "height": 0.210},
    "pingguo": {"diameter": 0.070, "height": 0.070},
    "chengzi": {"diameter": 0.074, "height": 0.074},
    "sanmingzhi": {"diameter": 0.065, "height": 0.099},
    "heweidao": {"diameter": 0.065, "height": 0.105},
    "zhijin": {"diameter": 0.085, "height": 0.088},
}


def recipe_for(kind: str) -> Optional[GraspRecipe]:
    name = str(kind)
    if name == KELE_RECIPE.kind:
        return KELE_RECIPE
    if name not in _PLACEHOLDER:
        return None
    spec = _PLACEHOLDER[name]
    base = KELE_RECIPE
    return GraspRecipe(
        kind=name,
        version=f"{name}-unverified",
        enabled=False,
        diameter=float(spec["diameter"]),
        height=float(spec["height"]),
        radius=0.5 * float(spec["diameter"]),
        z_min=base.z_min,
        z_max=base.z_max,
        fwd_min=base.fwd_min,
        fwd_max=base.fwd_max,
        lateral_max=base.lateral_max,
        z_bias=base.z_bias,
        slide=base.slide,
        pregrasp_along=base.pregrasp_along,
        pregrasp_lateral=base.pregrasp_lateral,
        pregrasp_z=base.pregrasp_z,
        creep_stop_along=base.creep_stop_along,
        grip_open=base.grip_open,
        grip_close=base.grip_close,
        lift_clearance=base.lift_clearance,
        close_hold_s=base.close_hold_s,
        settle_evidence_s=base.settle_evidence_s,
        two_frame_pos_tol=0.020 if spec["diameter"] >= 0.070 else base.two_frame_pos_tol,
        associate_tol=base.associate_tol,
        rgb_depth_skew_max=base.rgb_depth_skew_max,
        follow_tol=base.follow_tol,
        carry_out_min=base.carry_out_min,
        h_bottom_from_g=0.5 * float(spec["height"]),
        release_clearance=base.release_clearance,
        table_support_z=base.table_support_z,
        creep_speed=base.creep_speed,
        retreat_speed=base.retreat_speed,
        lateral_error_bound=0.002 if spec["diameter"] >= 0.070 else base.lateral_error_bound,
        entry_margin=0.001 if spec["diameter"] >= 0.070 else base.entry_margin,
        actual_open_gap=base.actual_open_gap,
        place_stable_s=base.place_stable_s,
        place_stable_speed=base.place_stable_speed,
        dt_jump_s=base.dt_jump_s,
        max_close_attempts=base.max_close_attempts,
        place_above=base.place_above,
        grasp_rot=np.eye(3),
        t_eg=np.eye(4),
    )


def gripper_fits(recipe: GraspRecipe) -> bool:
    need = recipe.diameter + 2.0 * (recipe.lateral_error_bound + recipe.entry_margin)
    return need <= recipe.actual_open_gap + 1e-9


SLIDE_GRASP = KELE_RECIPE.slide
VISION_SURFACE_TO_CENTER_FWD = KELE_RECIPE.radius
REACH_FWD_MIN = KELE_RECIPE.fwd_min
REACH_FWD_MAX = KELE_RECIPE.fwd_max
REACH_LATERAL_MAX = KELE_RECIPE.lateral_max
REACH_Z_MIN = KELE_RECIPE.z_min
REACH_Z_MAX = KELE_RECIPE.z_max

DEPLOY_CART_TOL = 0.065
DEPLOY_JOINT_TOL = 0.080
DEPLOY_ROT_TOL = 0.35
DEPLOY_SLIDE_TOL = 0.025
SETTLE_EVIDENCE_S = KELE_RECIPE.settle_evidence_s
# Old extra wait; kept as 0 so callers that still add it do not re-dwell.
DEPLOY_DWELL_S = 0.0

CREEP_SPEED = KELE_RECIPE.creep_speed
CREEP_YAW_KP = 4.0
CREEP_SLOW_ALONG = 0.080
CREEP_SLOW_SPEED = 0.03
LIFT_TIMEOUT_S = 3.0
C2_LOOK_S = 2.0
SLOT_LAYER_Z = (0.72, 1.10)
LIFT_AMOUNT = KELE_RECIPE.lift_clearance
CARRY_SLIDE = float(KELE_RECIPE.carry_slide)
RETREAT_SPEED = KELE_RECIPE.retreat_speed
RETREAT_Y = SHELF_AISLE_Y
POS_TOL = 0.06
GRASP_YAW = SHELF_FACE_YAW
CARRY_LIN = 0.20
CARRY_ANG = 0.35

INIT_ARM_L = np.array([0.0, -0.166, 0.032, 0.0, 1.571, 2.223])
INIT_ARM_R = np.array([0.0, -0.166, 0.032, 0.0, -1.571, -2.223])
INIT_DWELL_S = 0.0

RATE_HZ = 30.0
JOINT_SLEW = 1.2
MAX_LIN = 0.35
MAX_ANG = 0.65
MAX_LIN_ACC = 0.80
MAX_ANG_ACC = 5.0
DT_JUMP_S = KELE_RECIPE.dt_jump_s

CLOSE_HOLD_S = KELE_RECIPE.close_hold_s
PLACE_OPEN_HOLD_S = 0.0
PLACE_DEPLOY_Z = KELE_RECIPE.table_support_z + KELE_RECIPE.h_bottom_from_g + KELE_RECIPE.release_clearance + KELE_RECIPE.place_above
PLACE_LOWER_Z = KELE_RECIPE.table_support_z + KELE_RECIPE.h_bottom_from_g + KELE_RECIPE.release_clearance
PLACE_RETRACT_FWD = 0.18
PLACE_CART_TOL = 0.080
GRASP_TIMEOUT_S = 60.0
PLACE_TIMEOUT_S = 30.0
NAV_TIMEOUT_S = 150.0
LOCK_TIMEOUT_S = 12.0
LOCK_ATTEMPTS = 3
DETECT_MIN_SAMPLES = 2
RGB_DEPTH_SKEW_MAX = KELE_RECIPE.rgb_depth_skew_max

SLOT_LAYER_PITCH = {1: -0.12, 2: -0.28, 3: -0.50}
SLOT_COLUMN_YAW = {1: 0.30, 2: 0.0, 3: -0.30}

__all__ = [
    "CARRY_ANG",
    "CARRY_LIN",
    "CREEP_SPEED",
    "DEFAULT_KIND",
    "DEFAULT_WEIGHTS",
    "DELIVERY_APPROACH_XY",
    "DELIVERY_FACE_YAW",
    "DELIVERY_TABLE_XY",
    "DELIVERY_TARGET_XYZ",
    "GRASP_YAW",
    "KELE_RECIPE",
    "LOG_DIR",
    "MAX_ANG",
    "MAX_LIN",
    "NAV_TIMEOUT_S",
    "PLACE_SLOT_XY",
    "SHELF_APPROACH_XY",
    "SHELF_FACE_YAW",
    "gripper_fits",
    "recipe_for",
]
