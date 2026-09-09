import math
import time
import unittest

import numpy as np

from runtime.p5_grasp import (
    DetectionSample,
    GraspObservation,
    GraspPhase,
    GraspSession,
    LatestFrameWorker,
    ObjectHoldState,
    cam_to_world,
    clip_base,
    creep_remaining_along,
    footprint_to_world,
    locate_head_hint,
    lock_from_frames,
    look_at_world,
    patch_depth_m,
    pick_centered_detection,
    place_world_targets,
    pregrasp_world,
    reachable_in_footprint,
    slot_head_hint,
    slot_hint_agrees_with_height,
    vision_to_object_center,
    world_to_footprint,
)
from runtime.p5_params import (
    CARRY_SLIDE,
    CREEP_SPEED,
    GRASP_HEAD_PITCH,
    GRASP_YAW,
    GRIP_CLOSE,
    HEAD_PITCH,
    HEAD_YAW,
    INIT_ARM_L,
    INIT_ARM_R,
    KELE_RECIPE,
    MAX_ANG,
    MAX_LIN,
    PLACE_LOWER_Z,
    REACH_LATERAL_MAX,
    SLIDE_GRASP,
    gripper_fits,
    recipe_for,
)
from runtime.scene_zones import DELIVERY_TARGET_XYZ, SHELF_APPROACH_XY, in_delivery_box


SHELF_XY = tuple(SHELF_APPROACH_XY)


def _ahead(dist=0.80, z=0.90, xy=SHELF_XY, yaw=GRASP_YAW, lateral=0.0):
    return footprint_to_world([dist, lateral, z], xy, yaw)


def _obs(t, xy=SHELF_XY, yaw=GRASP_YAW, dets=(), ee=None, slide=SLIDE_GRASP,
         right_arm=None, right_grip=1.0, rgb_frame_id=None, linear_speed=0.0):
    return GraspObservation(
        t=float(t),
        base_xy=xy,
        base_yaw=float(yaw),
        slide=float(slide),
        head_yaw=HEAD_YAW,
        head_pitch=HEAD_PITCH,
        left_arm=INIT_ARM_L.copy(),
        left_grip=1.0,
        right_arm=INIT_ARM_R.copy() if right_arm is None else np.asarray(right_arm, dtype=float),
        right_grip=float(right_grip),
        detections_world=list(dets),
        ee_world=None if ee is None else np.asarray(ee, dtype=float),
        ee_rot_footprint=np.eye(3),
        rgb_frame_id=rgb_frame_id,
        linear_speed=float(linear_speed),
    )


def _ik_ok(footprint, _rot, _slide, q_ref):
    return np.asarray(q_ref, dtype=float).copy()


def _sample(point, frame_id, kind="kele"):
    point = np.asarray(point, dtype=float)
    return DetectionSample(
        kind=kind, stamp=float(frame_id), frame_id=int(frame_id),
        p_surface=point, p_center=point,
    )


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

    def test_two_frames_lock_and_reject_same_image(self):
        raw = vision_to_object_center(_ahead(0.80, 0.90), SHELF_XY, GRASP_YAW)
        self.assertIsNone(lock_from_frames([_sample(raw, 1)], SHELF_XY, GRASP_YAW))
        self.assertIsNone(lock_from_frames([_sample(raw, 1), _sample(raw, 1)], SHELF_XY, GRASP_YAW))
        locked = lock_from_frames([_sample(raw, 1), _sample(raw, 2)], SHELF_XY, GRASP_YAW)
        self.assertIsNotNone(locked)
        np.testing.assert_allclose(locked.object_world, raw, atol=1e-9)
        self.assertFalse(np.allclose(locked.deploy_world, locked.object_world))

    def test_two_frames_do_not_mix_two_bottles(self):
        left = vision_to_object_center(_ahead(0.80, 0.90, lateral=0.10), SHELF_XY, GRASP_YAW)
        right = vision_to_object_center(_ahead(0.80, 0.90, lateral=-0.10), SHELF_XY, GRASP_YAW)
        self.assertIsNone(lock_from_frames(
            [_sample(left, 1), _sample(right, 2)], SHELF_XY, GRASP_YAW,
        ))
        locked = lock_from_frames(
            [_sample(left, 1), _sample(right, 2), _sample(left, 3)], SHELF_XY, GRASP_YAW,
        )
        self.assertIsNotNone(locked)
        np.testing.assert_allclose(locked.object_world, left, atol=1e-6)

    def test_pregrasp_rotates_with_base_yaw(self):
        obj = np.array([1.0, 2.0, 0.90])
        facing_east = pregrasp_world(obj, 0.0)
        self.assertLess(facing_east[0], obj[0] - 0.15)
        self.assertAlmostEqual(facing_east[1], obj[1] + KELE_RECIPE.pregrasp_lateral, places=5)
        facing_north = pregrasp_world(obj, math.pi / 2.0)
        self.assertLess(facing_north[1], obj[1] - 0.15)
        self.assertAlmostEqual(facing_north[0], obj[0] - KELE_RECIPE.pregrasp_lateral, places=5)
        self.assertAlmostEqual(KELE_RECIPE.pregrasp_lateral, 0.0)
        self.assertAlmostEqual(KELE_RECIPE.creep_stop_along, -0.035)

    def test_vision_z_bias_lowers_pinch(self):
        raw = _ahead(0.80, 0.90)
        center = vision_to_object_center(raw, SHELF_XY, GRASP_YAW)
        self.assertAlmostEqual(center[2], raw[2] + KELE_RECIPE.z_bias, places=6)
        self.assertLess(center[2], raw[2])

    def test_wrong_layer_slot_hint_does_not_aim_at_l1(self):
        self.assertFalse(slot_hint_agrees_with_height("D-L1-C2", 0.97))
        self.assertTrue(slot_hint_agrees_with_height("D-L2-C3", 0.97))
        yaw, pitch = locate_head_hint("D-L1-C2", 1, height=0.97)
        self.assertEqual((yaw, pitch), (HEAD_YAW, -0.28))

    def test_look_at_world_points_at_l2_cola_not_the_floor(self):
        obj = _ahead(0.80, 0.90)
        yaw, pitch = look_at_world(obj, SHELF_XY, GRASP_YAW, SLIDE_GRASP)
        self.assertLess(pitch, HEAD_PITCH)
        self.assertGreater(pitch, -0.70)
        self.assertLess(abs(yaw), 0.35)
        high = look_at_world(_ahead(0.80, 1.20), SHELF_XY, GRASP_YAW, SLIDE_GRASP)
        self.assertGreater(high[1], pitch)

    def test_picks_the_most_centered_bottle(self):
        left = _ahead(0.80, 0.90, lateral=0.10)
        center = _ahead(0.80, 0.90, lateral=0.01)
        chosen = pick_centered_detection([left, center], SHELF_XY, GRASP_YAW)
        np.testing.assert_allclose(chosen, center, atol=1e-9)

    def test_slot_hint_is_head_joints_not_xyz(self):
        yaw, pitch = slot_head_hint("D-L2-C3")
        self.assertEqual((yaw, pitch), (-0.30, -0.28))
        self.assertEqual(len(slot_head_hint("D-L2-C3")), 2)
        self.assertEqual(slot_head_hint(None), (HEAD_YAW, HEAD_PITCH))
        self.assertEqual(locate_head_hint(None, 1), (HEAD_YAW, HEAD_PITCH))
        self.assertNotAlmostEqual(HEAD_PITCH, GRASP_HEAD_PITCH)
        self.assertGreater(locate_head_hint(None, 1)[1], GRASP_HEAD_PITCH)
        self.assertNotAlmostEqual(locate_head_hint(None, 2)[1], HEAD_PITCH)

    def test_place_height_uses_grasp_to_bottom_not_old_085(self):
        deploy, lower = place_world_targets()
        self.assertEqual(tuple(np.round(lower[:2], 3)), tuple(np.round(DELIVERY_TARGET_XYZ[:2], 3)))
        self.assertAlmostEqual(lower[2], PLACE_LOWER_Z)
        self.assertGreater(lower[2], 0.85)
        self.assertTrue(in_delivery_box(lower))
        self.assertGreater(deploy[2], lower[2])

    def test_kele_fits_gripper_orange_does_not_reuse_kele_budget(self):
        self.assertTrue(gripper_fits(KELE_RECIPE))
        kele_need = KELE_RECIPE.diameter + 2.0 * (KELE_RECIPE.lateral_error_bound + KELE_RECIPE.entry_margin)
        orange_need = 0.074 + 2.0 * (KELE_RECIPE.lateral_error_bound + KELE_RECIPE.entry_margin)
        self.assertLessEqual(kele_need, KELE_RECIPE.actual_open_gap)
        self.assertGreater(orange_need, KELE_RECIPE.actual_open_gap)
        self.assertFalse(recipe_for("zhijin").enabled)

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
    def _lock(self, session, bottle, t=0.0, dt=0.1):
        cmd = None
        while t < 4.0 and session.phase in (GraspPhase.INIT, GraspPhase.AIM):
            cmd = session.tick(_obs(t, dets=[bottle], rgb_frame_id=int(round(t * 10))))
            t += dt
        return t, cmd

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
        self.assertEqual(cmd.linear, 0.0)

    def test_same_cached_frame_is_not_two_observations(self):
        session = GraspSession(_ik_ok, now=0.0, dt=0.1)
        bottle = _ahead(0.80, 0.90)
        session.tick(_obs(0.0, dets=[bottle], rgb_frame_id=7))
        session.phase = GraspPhase.AIM
        session.phase_t0 = 0.2
        session.tick(_obs(0.3, dets=[bottle], rgb_frame_id=7))
        session.tick(_obs(0.4, dets=[bottle], rgb_frame_id=7))
        self.assertIsNone(session.locked)
        self.assertEqual(session.phase, GraspPhase.AIM)

    def test_large_dt_pauses_instead_of_jumping(self):
        session = GraspSession(_ik_ok, now=0.0, dt=1.0 / 30.0)
        bottle = _ahead(0.80, 0.90)
        t, _ = self._lock(session, bottle)
        session.phase = GraspPhase.CREEP
        session.phase_t0 = t
        session.tick(_obs(t, ee=_ahead(0.40, 0.90), rgb_frame_id=20))
        cmd = session.tick(_obs(t + 1.0, ee=_ahead(0.40, 0.90), rgb_frame_id=21))
        self.assertEqual(session.phase, GraspPhase.CREEP)
        self.assertEqual(cmd.linear, 0.0)
        self.assertTrue(cmd.hard_stop)
        self.assertIn("dt_pause", cmd.log)

    def test_grasp_then_place_without_task_coordinates(self):
        session = GraspSession(_ik_ok, now=0.0, dt=0.1)
        bottle = _ahead(0.80, 0.90)
        t, _ = self._lock(session, bottle)
        self.assertIsNotNone(session.locked)
        self.assertEqual(session.phase, GraspPhase.DEPLOY)
        self.assertTrue(session.c1)
        self.assertFalse(session.used_json_xyz)

        deploy = session.locked.deploy_world
        while t < 6.0 and session.phase == GraspPhase.DEPLOY:
            cmd = session.tick(_obs(t, dets=[bottle], ee=deploy, right_arm=session.tc[10:16],
                                    rgb_frame_id=int(round(t * 10))))
            t += 0.1
        self.assertEqual(session.phase, GraspPhase.CREEP)

        cmd = session.tick(_obs(t, ee=_ahead(0.40, 0.90), rgb_frame_id=int(round(t * 10))))
        self.assertGreater(cmd.linear, 0.0)
        self.assertLessEqual(cmd.linear, CREEP_SPEED)
        t += 0.1
        at_center = session.locked.object_world
        session.tick(_obs(t, ee=at_center, rgb_frame_id=int(round(t * 10))))
        self.assertEqual(session.phase, GraspPhase.CREEP)
        self.assertGreater(creep_remaining_along(
            at_center, at_center, session.locked.approach_world,
        ), 0.02)
        wrap = at_center + 0.04 * session.locked.approach_world
        t += 0.1
        session.tick(_obs(t, ee=wrap, rgb_frame_id=int(round(t * 10))))
        self.assertEqual(session.phase, GraspPhase.CLOSE)

        close_until = t + 5.0
        while t < close_until and session.phase == GraspPhase.CLOSE:
            t += 0.1
            session.tick(_obs(t, ee=wrap, rgb_frame_id=int(round(t * 10))))
        self.assertEqual(session.phase, GraspPhase.LIFT)

        held = session.locked.object_world + np.array([0.0, -0.25, 0.02])
        while t < 20.0 and session.phase in (GraspPhase.LIFT, GraspPhase.RETREAT):
            t += 0.1
            axis = session.locked.approach_world
            backed = min(0.30, max(0.0, (t - 8.0) * 0.20))
            xy = (
                session.locked.base_xy[0] - backed * axis[0],
                session.locked.base_xy[1] - backed * axis[1],
            )
            follow = held if session.phase == GraspPhase.RETREAT else deploy
            slide = CARRY_SLIDE if session.phase == GraspPhase.RETREAT else SLIDE_GRASP - 0.02
            session.tick(_obs(
                t, xy=xy, dets=[follow], ee=follow,
                slide=slide, right_grip=GRIP_CLOSE,
                rgb_frame_id=int(round(t * 10)),
            ))
        self.assertEqual(session.phase, GraspPhase.HELD)
        self.assertEqual(session.hold_state, ObjectHoldState.HELD)
        self.assertTrue(session.c2)
        self.assertLess(session.action[16], 0.20)
        self.assertAlmostEqual(session.tc[0], CARRY_SLIDE, places=3)

        session.begin_place(now=t)
        self.assertEqual(session.phase, GraspPhase.PLACE_DEPLOY)
        place = session.place_lower_world
        table_xy = (-1.94, -2.90)
        while t < 30.0 and session.phase == GraspPhase.PLACE_DEPLOY:
            t += 0.1
            session.tick(_obs(
                t, xy=table_xy, yaw=-math.pi / 2.0,
                ee=session.place_deploy_world, right_arm=session.tc[10:16],
                right_grip=GRIP_CLOSE, slide=session.tc[0],
                rgb_frame_id=int(round(t * 10)),
            ))
        self.assertEqual(session.phase, GraspPhase.PLACE_LOWER)
        while t < 35.0 and session.phase == GraspPhase.PLACE_LOWER:
            t += 0.1
            session.tick(_obs(
                t, xy=table_xy, yaw=-math.pi / 2.0,
                ee=place, right_arm=session.tc[10:16], right_grip=GRIP_CLOSE,
                slide=session.tc[0], rgb_frame_id=int(round(t * 10)),
            ))
        self.assertEqual(session.phase, GraspPhase.PLACE_OPEN)
        while t < 45.0 and session.phase != GraspPhase.DONE:
            t += 0.1
            session.tick(_obs(
                t, xy=table_xy, yaw=-math.pi / 2.0, ee=_ahead(0.40, 0.90, xy=table_xy, yaw=-math.pi / 2.0),
                dets=[place], rgb_frame_id=int(round(t * 10)),
            ))
        self.assertEqual(session.phase, GraspPhase.DONE)
        self.assertTrue(session.c3)
        self.assertTrue(session.place_verified_local)
        self.assertIsNone(session.referee_confirmed)
        self.assertFalse(session.used_json_xyz)
        self.assertTrue(session.report_extras()["place_target_in_box"])

    def test_empty_grasp_does_not_report_success(self):
        session = GraspSession(_ik_ok, now=0.0, dt=0.1)
        bottle = _ahead(0.80, 0.90)
        t, _ = self._lock(session, bottle)
        session.phase = GraspPhase.LIFT
        session.phase_t0 = t
        session.arm_target_set = True
        for _ in range(40):
            t += 0.1
            axis = session.locked.approach_world
            backed = min(0.30, t * 0.05)
            xy = (
                session.locked.base_xy[0] - backed * axis[0],
                session.locked.base_xy[1] - backed * axis[1],
            )
            cmd = session.tick(_obs(
                t, xy=xy, dets=[bottle], ee=session.locked.deploy_world,
                slide=SLIDE_GRASP - 0.02, rgb_frame_id=int(round(t * 10)),
            ))
            if cmd.failed or session.phase == GraspPhase.HELD:
                break
        self.assertTrue(cmd.failed)
        self.assertEqual(session.reason, "empty_grasp")
        self.assertNotEqual(session.phase, GraspPhase.HELD)

    def test_creep_yaw_is_clipped_to_base_limit(self):
        session = GraspSession(_ik_ok, now=0.0, dt=0.1)
        bottle = _ahead(0.80, 0.90)
        t, _ = self._lock(session, bottle)
        session.phase = GraspPhase.CREEP
        session.phase_t0 = t
        cmd = session.tick(_obs(t + 0.05, yaw=0.0, ee=_ahead(0.40, 0.90)))
        self.assertLessEqual(abs(cmd.angular), MAX_ANG + 1e-9)
        self.assertGreater(cmd.linear, 0.0)

    def test_place_without_grasp_fails_closed(self):
        session = GraspSession(_ik_ok, now=0.0)
        session.begin_place(now=0.0)
        self.assertEqual(session.phase, GraspPhase.FAILED)
        self.assertEqual(session.reason, "place_without_grasp")

    def test_deploy_solves_ik_once(self):
        calls = {"n": 0}

        def ik(footprint, _rot, _slide, q_ref):
            calls["n"] += 1
            return np.asarray(q_ref, dtype=float).copy()

        session = GraspSession(ik, now=0.0, dt=0.1)
        bottle = _ahead(0.80, 0.90)
        t, _ = self._lock(session, bottle)
        self.assertEqual(session.phase, GraspPhase.DEPLOY)
        after_lock = calls["n"]
        self.assertGreaterEqual(after_lock, 1)
        deploy = session.locked.deploy_world
        while t < 6.0 and session.phase == GraspPhase.DEPLOY:
            session.tick(_obs(t, dets=[bottle], ee=deploy, right_arm=session.tc[10:16],
                              rgb_frame_id=int(round(t * 10))))
            t += 0.1
        self.assertEqual(session.phase, GraspPhase.CREEP)
        self.assertEqual(calls["n"], after_lock)

    def test_creep_slows_before_wrap(self):
        session = GraspSession(_ik_ok, now=0.0, dt=0.1)
        bottle = _ahead(0.80, 0.90)
        t, _ = self._lock(session, bottle)
        session.phase = GraspPhase.CREEP
        session.phase_t0 = t
        near = session.locked.object_world - 0.03 * session.locked.approach_world
        cmd = session.tick(_obs(t + 0.05, ee=near, yaw=0.0))
        self.assertEqual(session.phase, GraspPhase.CREEP)
        self.assertLessEqual(cmd.linear, 0.03 + 1e-9)
        self.assertEqual(cmd.angular, 0.0)

    def test_lift_stall_extracts_instead_of_grinding(self):
        session = GraspSession(_ik_ok, now=0.0, dt=0.1)
        bottle = _ahead(0.80, 0.90)
        t, _ = self._lock(session, bottle)
        session.phase = GraspPhase.LIFT
        session.phase_t0 = t
        session.arm_target_set = True
        cmd = None
        while t < session.phase_t0 + 3.2 and session.phase == GraspPhase.LIFT:
            t += 0.1
            cmd = session.tick(_obs(
                t, ee=session.locked.object_world, slide=SLIDE_GRASP,
                right_grip=GRIP_CLOSE, rgb_frame_id=int(round(t * 10)),
            ))
        self.assertEqual(session.phase, GraspPhase.RETREAT)
        self.assertIn("lift stall", session.last_log)
        self.assertIsNotNone(cmd)
        self.assertFalse(cmd.failed)

    def test_lift_looks_down_at_the_gripper_not_the_shelf(self):
        session = GraspSession(_ik_ok, now=0.0, dt=0.1)
        bottle = _ahead(0.80, 0.90)
        t, _ = self._lock(session, bottle)
        session.phase = GraspPhase.LIFT
        session.phase_t0 = t
        session.tick(_obs(t + 0.1, ee=session.locked.object_world, slide=SLIDE_GRASP,
                          right_grip=GRIP_CLOSE, rgb_frame_id=30))
        want = look_at_world(session.locked.object_world, SHELF_XY, GRASP_YAW, SLIDE_GRASP)
        self.assertAlmostEqual(session.tc[2], want[1], places=2)
        self.assertLess(session.tc[2], HEAD_PITCH)
        self.assertNotAlmostEqual(session.tc[2], HEAD_PITCH, places=3)

    def test_creep_looks_at_the_locked_can(self):
        session = GraspSession(_ik_ok, now=0.0, dt=0.1)
        bottle = _ahead(0.80, 0.90)
        t, _ = self._lock(session, bottle)
        self.assertEqual(session.phase, GraspPhase.DEPLOY)
        session.tick(_obs(t + 0.1, ee=session.locked.deploy_world, right_arm=session.tc[10:16],
                          rgb_frame_id=int(round((t + 0.1) * 10))))
        want = look_at_world(session.locked.object_world, SHELF_XY, GRASP_YAW, SLIDE_GRASP)
        self.assertAlmostEqual(session.tc[2], want[1], places=2)

    def test_c2_unknown_waits_to_look_before_pause(self):
        session = GraspSession(_ik_ok, now=0.0, dt=0.1)
        bottle = _ahead(0.80, 0.90)
        t, _ = self._lock(session, bottle)
        session.phase = GraspPhase.RETREAT
        session.phase_t0 = t
        axis = session.locked.approach_world
        xy = (
            session.locked.base_xy[0] - 0.30 * axis[0],
            session.locked.base_xy[1] - 0.30 * axis[1],
        )
        look_t0 = t + 0.1
        session.tick(_obs(look_t0, xy=xy, ee=session.locked.deploy_world, right_grip=GRIP_CLOSE,
                          rgb_frame_id=40))
        self.assertEqual(session.phase, GraspPhase.RETREAT)
        t = look_t0
        while t < look_t0 + 1.5 and session.phase == GraspPhase.RETREAT:
            t += 0.1
            session.tick(_obs(t, xy=xy, ee=session.locked.deploy_world, right_grip=GRIP_CLOSE,
                              rgb_frame_id=int(round(t * 10))))
        self.assertEqual(session.phase, GraspPhase.RETREAT)
        while t < look_t0 + 2.4 and session.phase == GraspPhase.RETREAT:
            t += 0.1
            session.tick(_obs(t, xy=xy, ee=session.locked.deploy_world, right_grip=GRIP_CLOSE,
                              rgb_frame_id=int(round(t * 10))))
        self.assertEqual(session.phase, GraspPhase.PAUSED)
        self.assertEqual(session.reason, "c2_unknown")
        self.assertFalse(session.c2)
        self.assertAlmostEqual(session.tc[0], CARRY_SLIDE, places=3)

    def test_retreat_raises_torso_after_leaving_the_shelf(self):
        session = GraspSession(_ik_ok, now=0.0, dt=0.1)
        bottle = _ahead(0.80, 0.90)
        t, _ = self._lock(session, bottle)
        session.phase = GraspPhase.RETREAT
        session.phase_t0 = t
        session.tick(_obs(t + 0.05, ee=session.locked.object_world, slide=SLIDE_GRASP,
                          right_grip=GRIP_CLOSE, rgb_frame_id=50))
        self.assertGreater(session.tc[0], 0.05)
        axis = session.locked.approach_world
        xy = (
            session.locked.base_xy[0] - 0.30 * axis[0],
            session.locked.base_xy[1] - 0.30 * axis[1],
        )
        session.tick(_obs(t + 0.15, xy=xy, ee=session.locked.deploy_world, slide=SLIDE_GRASP,
                          right_grip=GRIP_CLOSE, rgb_frame_id=51))
        self.assertAlmostEqual(session.tc[0], CARRY_SLIDE, places=3)
        self.assertLess(CARRY_SLIDE, SLIDE_GRASP)

    def test_unsupported_kind_fails_closed(self):
        session = GraspSession(_ik_ok, kind="zhijin", now=0.0, dt=0.1)
        session.phase = GraspPhase.AIM
        session.phase_t0 = 0.0
        cmd = session.tick(_obs(0.1, dets=[_ahead(0.80, 0.90)], rgb_frame_id=1))
        self.assertTrue(cmd.failed)
        self.assertEqual(session.reason, "template_unavailable")

    def test_worker_keeps_latest_frame_and_skips_duplicates(self):
        seen = []

        def infer(payload):
            time.sleep(0.02)
            seen.append(payload)
            return payload

        worker = LatestFrameWorker(infer)
        try:
            self.assertTrue(worker.submit(1, "a"))
            time.sleep(0.08)
            self.assertFalse(worker.submit(1, "a"))
            self.assertTrue(worker.submit(2, "b"))
            time.sleep(0.08)
            got = worker.latest()
            self.assertIsNotNone(got)
            self.assertEqual(got[0], 2)
            self.assertEqual(got[1], "b")
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
