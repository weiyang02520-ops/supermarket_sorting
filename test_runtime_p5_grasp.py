import math
import unittest

import numpy as np

from runtime.p5_grasp import (
    GraspObservation,
    GraspPhase,
    GraspSession,
    cam_to_world,
    clip_base,
    footprint_to_world,
    lock_from_samples,
    patch_depth_m,
    pick_centered_detection,
    place_world_targets,
    reachable_in_footprint,
    slot_head_hint,
    vision_to_object_center,
    world_to_footprint,
)
from runtime.p5_params import (
    CREEP_SPEED,
    DETECT_MIN_SAMPLES,
    GRASP_YAW,
    GRIP_CLOSE,
    INIT_ARM_L,
    INIT_ARM_R,
    MAX_ANG,
    MAX_LIN,
    PLACE_LOWER_Z,
    REACH_LATERAL_MAX,
    RETREAT_Y,
    SLIDE_GRASP,
)
from runtime.scene_zones import DELIVERY_TARGET_XYZ, SHELF_APPROACH_XY, in_delivery_box


SHELF_XY = tuple(SHELF_APPROACH_XY)


def _ahead(dist=0.80, z=0.90, xy=SHELF_XY, yaw=GRASP_YAW, lateral=0.0):
    return footprint_to_world([dist, lateral, z], xy, yaw)


def _obs(t, xy=SHELF_XY, yaw=GRASP_YAW, dets=(), ee=None, slide=SLIDE_GRASP,
         right_arm=None, right_grip=1.0):
    return GraspObservation(
        t=float(t),
        base_xy=xy,
        base_yaw=float(yaw),
        slide=float(slide),
        head_yaw=0.0,
        head_pitch=-0.6,
        left_arm=INIT_ARM_L.copy(),
        left_grip=1.0,
        right_arm=INIT_ARM_R.copy() if right_arm is None else np.asarray(right_arm, dtype=float),
        right_grip=float(right_grip),
        detections_world=list(dets),
        ee_world=None if ee is None else np.asarray(ee, dtype=float),
        ee_rot_footprint=np.eye(3),
    )


def _ik_ok(footprint, _rot, _slide, q_ref):
    return np.asarray(q_ref, dtype=float).copy()


class P5GeometryTests(unittest.TestCase):
    def test_footprint_round_trip(self):
        world = _ahead(0.80, 0.90)
        back = footprint_to_world(world_to_footprint(world, SHELF_XY, GRASP_YAW), SHELF_XY, GRASP_YAW)
        np.testing.assert_allclose(back, world, atol=1e-9)

    def test_reachable_rejects_floor_far_and_side(self):
        self.assertTrue(reachable_in_footprint(_ahead(0.80, 0.90), SHELF_XY, GRASP_YAW))
        self.assertFalse(reachable_in_footprint(_ahead(0.80, 0.20), SHELF_XY, GRASP_YAW))
        self.assertFalse(reachable_in_footprint(_ahead(2.00, 0.90), SHELF_XY, GRASP_YAW))
        self.assertFalse(reachable_in_footprint(
            _ahead(0.80, 0.90, lateral=REACH_LATERAL_MAX + 0.05), SHELF_XY, GRASP_YAW,
        ))

    def test_lock_needs_samples_and_applies_surface_offset(self):
        raw = _ahead(0.80, 0.90)
        self.assertIsNone(lock_from_samples([raw] * 4, SHELF_XY, GRASP_YAW))
        locked = lock_from_samples([raw] * DETECT_MIN_SAMPLES, SHELF_XY, GRASP_YAW)
        self.assertIsNotNone(locked)
        center = vision_to_object_center(raw, SHELF_XY, GRASP_YAW)
        np.testing.assert_allclose(locked.object_world, center, atol=1e-9)
        self.assertGreater(locked.creep_stop_y, center[1] - 1e-9)
        self.assertFalse(np.allclose(locked.deploy_world, locked.object_world))

    def test_picks_the_most_centered_bottle(self):
        left = _ahead(0.80, 0.90, lateral=0.10)
        center = _ahead(0.80, 0.90, lateral=0.01)
        chosen = pick_centered_detection([left, center], SHELF_XY, GRASP_YAW)
        np.testing.assert_allclose(chosen, center, atol=1e-9)

    def test_slot_hint_is_head_joints_not_xyz(self):
        yaw, pitch = slot_head_hint("D-L2-C3")
        self.assertEqual((yaw, pitch), (-0.30, -0.28))
        self.assertEqual(len(slot_head_hint("D-L2-C3")), 2)
        self.assertEqual(slot_head_hint(None), (0.0, -0.6))

    def test_place_target_is_the_fixed_table_not_task_json(self):
        deploy, lower = place_world_targets()
        self.assertEqual(tuple(np.round(lower[:2], 3)), tuple(np.round(DELIVERY_TARGET_XYZ[:2], 3)))
        self.assertAlmostEqual(lower[2], PLACE_LOWER_Z)
        self.assertTrue(in_delivery_box(lower))
        self.assertGreater(deploy[2], lower[2])

    def test_patch_depth_ignores_invalid_and_cam_to_world(self):
        depth = np.zeros((8, 8), dtype=float)
        depth[2:5, 2:5] = 1.2
        depth[3, 3] = np.nan
        self.assertAlmostEqual(patch_depth_m(depth, 3, 3, radius=1), 1.2)
        self.assertEqual(patch_depth_m(np.zeros((4, 4)), 1, 1), 0.0)
        t_cam = np.eye(4)
        t_cam[:3, 3] = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(cam_to_world([0.1, 0.0, 0.5], t_cam), [1.1, 2.0, 3.5])

    def test_base_command_stays_inside_controller_limits(self):
        linear, angular = clip_base(3.0, -4.0)
        self.assertAlmostEqual(linear, MAX_LIN)
        self.assertAlmostEqual(angular, -MAX_ANG)


class P5SessionTests(unittest.TestCase):
    def test_lock_timeout_does_not_invent_json_xyz(self):
        session = GraspSession(_ik_ok, now=0.0)
        session.phase = GraspPhase.AIM
        session.phase_t0 = 0.0
        session.lock_attempt = 3
        cmd = session.tick(_obs(13.0, dets=()))
        self.assertEqual(cmd.phase, GraspPhase.FAILED)
        self.assertEqual(cmd.reason, "lock_timeout")
        self.assertIsNone(session.locked)
        self.assertFalse(session.used_json_xyz)

    def test_grasp_then_place_without_task_coordinates(self):
        session = GraspSession(_ik_ok, now=0.0, dt=0.1)
        bottle = _ahead(0.80, 0.90)
        t = 0.0
        cmd = None
        while t < 2.5 and (session.phase in (GraspPhase.INIT, GraspPhase.AIM)):
            cmd = session.tick(_obs(t, dets=[bottle]))
            t += 0.1
        self.assertIsNotNone(session.locked)
        self.assertEqual(session.phase, GraspPhase.DEPLOY)
        self.assertFalse(session.used_json_xyz)

        deploy = session.locked.deploy_world
        while t < 4.0 and session.phase == GraspPhase.DEPLOY:
            cmd = session.tick(_obs(t, dets=[bottle], ee=deploy, right_arm=session.tc[10:16]))
            t += 0.1
        self.assertEqual(session.phase, GraspPhase.CREEP)

        cmd = session.tick(_obs(t, ee=_ahead(0.40, 0.90)))
        self.assertGreater(cmd.linear, 0.0)
        self.assertLessEqual(cmd.linear, CREEP_SPEED)
        t += 0.1
        cmd = session.tick(_obs(t, ee=np.array([
            session.locked.object_world[0],
            session.locked.creep_stop_y + 0.01,
            session.locked.object_world[2],
        ])))
        self.assertEqual(session.phase, GraspPhase.CLOSE)

        while t < 6.0 and session.phase == GraspPhase.CLOSE:
            t += 0.1
            cmd = session.tick(_obs(t, ee=deploy, right_grip=session.action[16]))
        self.assertEqual(session.phase, GraspPhase.LIFT)

        cmd = session.tick(_obs(t + 0.1, slide=SLIDE_GRASP - 0.05, right_grip=session.action[16]))
        self.assertEqual(session.phase, GraspPhase.RETREAT)

        cmd = session.tick(_obs(
            t + 0.2, xy=(SHELF_XY[0], RETREAT_Y), right_grip=session.action[16],
        ))
        self.assertEqual(session.phase, GraspPhase.HELD)
        self.assertTrue(cmd.done)
        self.assertLess(cmd.right_grip, 0.20)

        session.begin_place(now=0.0)
        self.assertEqual(session.phase, GraspPhase.PLACE_DEPLOY)
        place = session.place_lower_world
        t = 0.0
        while t < 2.0 and session.phase == GraspPhase.PLACE_DEPLOY:
            cmd = session.tick(_obs(
                t, xy=(-1.94, -2.90), yaw=-math.pi / 2.0,
                ee=session.place_deploy_world, right_arm=session.tc[10:16],
                right_grip=GRIP_CLOSE, slide=session.tc[0],
            ))
            t += 0.1
        self.assertEqual(session.phase, GraspPhase.PLACE_LOWER)
        while t < 4.0 and session.phase == GraspPhase.PLACE_LOWER:
            cmd = session.tick(_obs(
                t, xy=(-1.94, -2.90), yaw=-math.pi / 2.0,
                ee=place, right_arm=session.tc[10:16], right_grip=GRIP_CLOSE,
                slide=session.tc[0],
            ))
            t += 0.1
        self.assertEqual(session.phase, GraspPhase.PLACE_OPEN)
        t += 0.9
        cmd = session.tick(_obs(t, xy=(-1.94, -2.90), yaw=-math.pi / 2.0, ee=place))
        self.assertEqual(session.phase, GraspPhase.PLACE_RETRACT)
        t += 0.9
        cmd = session.tick(_obs(t, xy=(-1.94, -2.90), yaw=-math.pi / 2.0, ee=place))
        self.assertEqual(session.phase, GraspPhase.DONE)
        self.assertTrue(cmd.done)
        self.assertFalse(session.used_json_xyz)
        self.assertTrue(session.report_extras()["place_target_in_box"])

    def test_creep_yaw_is_clipped_to_base_limit(self):
        session = GraspSession(_ik_ok, now=0.0)
        bottle = _ahead(0.80, 0.90)
        session.phase = GraspPhase.AIM
        session.phase_t0 = 0.0
        for t in (1.1, 1.2, 1.3, 1.4, 1.5):
            session.tick(_obs(t, dets=[bottle]))
        session.phase = GraspPhase.CREEP
        session.phase_t0 = 2.0
        cmd = session.tick(_obs(2.05, yaw=0.0, ee=_ahead(0.40, 0.90)))
        self.assertLessEqual(abs(cmd.angular), MAX_ANG + 1e-9)
        self.assertGreater(cmd.linear, 0.0)

    def test_place_without_grasp_fails_closed(self):
        session = GraspSession(_ik_ok, now=0.0)
        session.begin_place(now=0.0)
        self.assertEqual(session.phase, GraspPhase.FAILED)
        self.assertEqual(session.reason, "place_without_grasp")


if __name__ == "__main__":
    unittest.main()
