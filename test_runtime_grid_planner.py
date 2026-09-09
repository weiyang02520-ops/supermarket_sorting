import math
import unittest

from runtime.grid_planner import (
    OccupancyGrid,
    GlobalGridPlanner,
    GridNavController,
    illegal_shortcut,
    known_safe_prefix_of,
    plan_to_waypoints,
)
from runtime.lidar_avoid import safety_gate, sweep_gate
from runtime.nav_geometry import in_south_gap
from runtime.nav_types import NavGoal
from runtime.scene_zones import (
    DELIVERY_APPROACH_XY,
    HUG_WEST_X,
    SHELF_APPROACH_XY,
    WEST_LANE_Y,
    in_center_wall_band,
    in_delivery_base,
    in_north_racks,
    in_south_east_stub,
)
from runtime.waypoint_nav import wrap_to_pi


def _scan(n=72, hit_lo=-0.22, hit_hi=0.22, hit=0.30, clear=8.0):
    increment = 2.0 * math.pi / n
    angle_min = -math.pi
    ranges = []
    for index in range(n):
        angle = wrap_to_pi(angle_min + index * increment)
        ranges.append(hit if hit_lo <= angle <= hit_hi else clear)
    return ranges, angle_min, increment


class GridPlannerTests(unittest.TestCase):
    def test_static_layer_marks_solids_not_centre_bans(self):
        grid = OccupancyGrid()
        self.assertTrue(grid.is_static(0.80, 0.00))
        self.assertTrue(in_center_wall_band(0.80, 0.00))
        self.assertFalse(grid.is_physical(0.80, 2.60))
        self.assertTrue(in_north_racks(0.80, 2.60))
        self.assertTrue(grid.is_lethal(*grid.world_to_cell(0.80, 2.60)))
        self.assertTrue(in_south_east_stub(1.80, -3.60))
        self.assertFalse(grid.is_static(1.92, WEST_LANE_Y))
        self.assertFalse(grid.is_static(HUG_WEST_X, 0.50))
        self.assertFalse(grid.is_static(*SHELF_APPROACH_XY))
        self.assertFalse(grid.is_static(*DELIVERY_APPROACH_XY))

    def test_empty_map_plans_delivery_without_crossing_the_divider(self):
        planner = GlobalGridPlanner()
        route = planner.plan(1.92, -3.17, DELIVERY_APPROACH_XY, yaw=math.pi / 2.0)
        self.assertIsNotNone(route)
        self.assertGreater(len(route), 2)
        self.assertLess(math.hypot(route[-1][0] - DELIVERY_APPROACH_XY[0], route[-1][1] - DELIVERY_APPROACH_XY[1]), 0.20)
        west_of_aisle = False
        for px, py in route:
            self.assertFalse(in_center_wall_band(px, py))
            self.assertFalse(in_south_gap(px, py))
            if px < 0.10 and py > 1.70:
                west_of_aisle = True
        self.assertTrue(west_of_aisle)
        for index in range(len(route) - 1):
            self.assertFalse(illegal_shortcut(*route[index], *route[index + 1]))

    def test_empty_map_plans_shelf_along_the_yellow_lane(self):
        planner = GlobalGridPlanner()
        route = planner.plan(1.92, -3.17, SHELF_APPROACH_XY, yaw=math.pi / 2.0)
        self.assertIsNotNone(route)
        self.assertGreater(max(py for _px, py in route), 2.10)
        self.assertLess(min(abs(px - SHELF_APPROACH_XY[0]) for px, py in route[-3:]), 0.40)

    def test_east_corridor_southbound_is_legal(self):
        self.assertFalse(illegal_shortcut(1.92, 0.0, 1.92, -0.1))
        planner = GlobalGridPlanner()
        route = planner.plan(1.92, 0.0, (1.92, -2.80), yaw=-math.pi / 2.0, kind="inspection_approach")
        self.assertIsNotNone(route)
        self.assertTrue(any(py < -1.0 for _px, py in route))

    def test_a_box_on_the_hug_rail_still_reaches_delivery(self):
        planner = GlobalGridPlanner()
        planner.grid.mark_circle(-0.40, -0.20, 0.18, now=1.0)
        route = planner.plan(-0.40, 0.40, DELIVERY_APPROACH_XY, yaw=-math.pi / 2.0)
        self.assertIsNotNone(route)
        self.assertFalse(any(in_center_wall_band(px, py) for px, py in route))
        self.assertLess(route[-1][1], -2.40)

    def test_two_boxes_from_a_legal_start_leave_the_hug_rail(self):
        planner = GlobalGridPlanner()
        planner.grid.mark_circle(-0.40, -0.80, 0.22, now=1.0)
        planner.grid.mark_circle(-1.20, -0.80, 0.22, now=1.0)
        route = planner.plan(-0.40, 0.20, DELIVERY_APPROACH_XY, yaw=-math.pi / 2.0)
        self.assertIsNotNone(route)
        self.assertTrue(any(px < -1.30 for px, _py in route) or any(py < -1.40 for _px, py in route))
        self.assertLess(route[-1][1], -2.40)

    def test_start_inside_a_confirmed_box_is_rejected(self):
        planner = GlobalGridPlanner()
        planner.grid.mark_circle(-0.40, -0.80, 0.22, now=1.0)
        route = planner.plan(-0.44, -0.84, DELIVERY_APPROACH_XY, yaw=-math.pi / 2.0)
        self.assertIsNone(route)
        self.assertEqual(planner.last_result.failure_reason, "START_IN_COLLISION")

    def test_unknown_cells_do_not_make_the_start_unsolvable(self):
        planner = GlobalGridPlanner()
        route = planner.plan(1.92, -3.17, DELIVERY_APPROACH_XY, yaw=math.pi / 2.0)
        self.assertIsNotNone(route)

    def test_plan_to_waypoints_keeps_spacing(self):
        points = [(0.0, 0.0), (0.10, 0.0), (0.20, 0.0), (1.20, 0.0)]
        waypoints = plan_to_waypoints(points)
        self.assertGreaterEqual(waypoints[0][0], -0.01)
        for index in range(len(waypoints) - 1):
            self.assertGreaterEqual(
                math.hypot(waypoints[index + 1][0] - waypoints[index][0], waypoints[index + 1][1] - waypoints[index][1]),
                0.08,
            )

    def test_safety_gate_stops_and_does_not_steer(self):
        ranges, amin, inc = _scan(hit=0.30)
        linear, angular, status = safety_gate(1.2, 0.4, ranges, amin, inc, stop_dist=0.43)
        self.assertEqual(linear, 0.0)
        self.assertEqual(angular, 0.4)
        self.assertEqual(status, "safety_stop")
        slin, sang, sstatus = sweep_gate(1.2, 0.4, ranges, amin, inc, stop_dist=0.43)
        self.assertEqual(slin, 0.0)
        self.assertEqual(sstatus, "safety_stop")
        self.assertEqual(sang, 0.0)
        tlin, tang, tstatus = sweep_gate(0.0, 0.8, ranges, amin, inc, stop_dist=0.43)
        self.assertEqual(tlin, 0.0)
        self.assertEqual(tstatus, "safety_stop")
        self.assertAlmostEqual(tang, 0.8)

    def test_controller_plans_immediately_without_a_corridor_seed(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=0.30,
            max_ang=0.55,
        )
        ranges, amin, inc = _scan(hit=8.0)
        linear, _angular, done, status = nav.step(
            1.92, -3.17, math.pi / 2.0, ranges, amin, inc, dt=0.05, now=0.6, scan_id=1,
        )
        self.assertFalse(done)
        self.assertTrue(status.startswith("replan"))
        self.assertNotIn("corridor", status)
        self.assertEqual(nav.plan_id, 1)
        self.assertEqual(linear, 0.0)
        self.assertIsNotNone(nav.follower)
        first = list(nav.follower.waypoints)
        nav.planner.grid.mark_circle(1.70, 0.20, 0.16, now=1.0)
        reroute = nav.planner.plan(1.92, -3.17, DELIVERY_APPROACH_XY, yaw=math.pi / 2.0)
        self.assertIsNotNone(reroute)
        self.assertTrue(
            any(abs(px - 1.70) > 0.25 for px, py in reroute if abs(py - 0.20) < 0.35)
            or reroute != first
        )

    def test_delivery_route_ends_in_the_delivery_base(self):
        planner = GlobalGridPlanner()
        route = planner.plan(0.86, 2.32, DELIVERY_APPROACH_XY, yaw=math.pi)
        self.assertIsNotNone(route)
        self.assertTrue(in_delivery_base((route[-1][0], min(route[-1][1], -2.70))) or route[-1][1] <= -2.80)

    def test_divider_corner_is_not_a_second_physical_wall(self):
        grid = OccupancyGrid()
        self.assertFalse(grid.is_physical(-0.10, 2.10))
        self.assertFalse(grid.is_physical(HUG_WEST_X, 2.00))
        self.assertTrue(in_center_wall_band(0.40, 1.60) or grid.is_physical(0.40, 1.60))

    def test_repeated_scan_id_does_not_confirm_twice(self):
        planner = GlobalGridPlanner()
        ranges, amin, inc = _scan(n=72, hit_lo=-0.10, hit_hi=0.10, hit=0.80, clear=8.0)
        planner.integrate_scan(-0.40, 0.40, -math.pi / 2.0, ranges, amin, inc, now=1.0, scan_id=7)
        first = planner.grid.confirmed_count()
        planner.integrate_scan(-0.40, 0.40, -math.pi / 2.0, ranges, amin, inc, now=1.1, scan_id=7)
        self.assertEqual(planner.grid.confirmed_count(), first)

    def test_controller_holds_route_while_turning(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=0.30,
            max_ang=0.55,
        )
        ranges, amin, inc = _scan(hit=8.0)
        _lin, _ang, _done, status = nav.step(
            0.92, 2.39, 1.42, ranges, amin, inc, dt=0.05, now=0.6, scan_id=1,
        )
        self.assertTrue(status.startswith("replan"))
        plan_id = nav.plan_id
        for step in range(24):
            nav.step(
                0.92,
                2.39,
                1.42 + 0.06 * step,
                ranges,
                amin,
                inc,
                dt=0.05,
                now=0.65 + 0.05 * step,
                scan_id=1,
            )
        self.assertEqual(nav.plan_id, plan_id)

    def test_empty_plan_stays_under_budget(self):
        import time

        planner = GlobalGridPlanner()
        t0 = time.perf_counter()
        route = planner.plan(0.92, 2.39, DELIVERY_APPROACH_XY, yaw=math.pi)
        elapsed_ms = 1000.0 * (time.perf_counter() - t0)
        self.assertIsNotNone(route)
        self.assertLess(elapsed_ms, 180.0)

    def test_north_rack_pose_recovers_instead_of_holding(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=0.30,
            max_ang=0.55,
        )
        ranges, amin, inc = _scan(hit=8.0)
        saw_reverse = False
        for step in range(8):
            lin, _ang, done, status = nav.step(
                1.066, 2.486, 2.155, ranges, amin, inc, dt=0.05, now=0.2 + 0.05 * step, scan_id=step + 1,
            )
            if lin < 0.0:
                saw_reverse = True
            self.assertFalse(done)
            self.assertNotEqual(status, "hold")
        self.assertTrue(saw_reverse)

    def test_set_goal_keeps_the_map(self):
        nav = GridNavController(
            goal_xy=SHELF_APPROACH_XY,
            final_yaw=math.pi / 2.0,
            max_lin=0.30,
            max_ang=0.55,
        )
        nav.planner.grid.mark_circle(-0.90, 0.10, 0.16, now=1.0)
        before = nav.planner.grid.confirmed_count()
        nav.set_goal(
            NavGoal(
                goal_id="delivery_approach",
                kind="delivery_approach",
                xy=DELIVERY_APPROACH_XY,
                final_yaw=-math.pi / 2.0,
            )
        )
        self.assertEqual(nav.planner.grid.confirmed_count(), before)
        self.assertIsNone(nav.follower)
        self.assertEqual(nav.goal_kind, "delivery_approach")

    def test_no_unverified_yellow_fallback(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=0.30,
            max_ang=0.55,
        )
        self.assertFalse(hasattr(nav, "_corridor_seed") and callable(getattr(nav, "_corridor_seed")))
        blocked, amin, inc = _scan(hit=0.20)
        _lin, ang, _done, status = nav.step(
            -0.40, 0.90, -math.pi / 2.0, blocked, amin, inc, dt=0.05, now=0.6, scan_id=1,
        )
        self.assertNotIn(status, ("face_turn", "nudge", "replan corridor", "replan fallback"))

    def test_known_safe_prefix_stops_at_unobserved(self):
        planner = GlobalGridPlanner()
        result = planner.plan_to(1.92, 0.40, math.pi / 2.0, SHELF_APPROACH_XY, kind="shelf_approach")
        self.assertEqual(result.status, "OK")
        self.assertGreaterEqual(len(result.path_xy), 2)
        self.assertLessEqual(len(result.known_safe_prefix), 1)
        self.assertIsNotNone(result.first_unknown_region)
        grid = planner.grid
        for point in result.path_xy[:4]:
            col, row = grid.world_to_cell(point[0], point[1])
            if grid.in_grid(col, row) and not grid.is_lethal(col, row):
                grid.observed[grid._index(col, row)] = 1
        prefix, unknown = known_safe_prefix_of(grid, result.path_xy)
        self.assertGreaterEqual(len(prefix), 2)
        if unknown is not None:
            self.assertGreaterEqual(
                result.path_xy.index(unknown) if unknown in result.path_xy else len(prefix),
                len(prefix),
            )

    def test_clear_known_prefix_uses_eight_x_cap(self):
        nav = GridNavController(
            goal_xy=SHELF_APPROACH_XY,
            final_yaw=math.pi / 2.0,
            max_lin=2.4,
            max_ang=1.65,
            cruise_lin=0.30,
        )
        ranges, amin, inc = _scan(hit=8.0)
        nav.step(1.92, 0.20, math.pi / 2.0, ranges, amin, inc, dt=0.05, now=0.6, scan_id=1)
        self.assertIsNotNone(nav.follower)
        cap = nav._follow_speed_cap(1.92, 0.20, ranges, amin, inc)
        self.assertGreaterEqual(cap, 2.0)
        self.assertAlmostEqual(cap, 2.4, places=2)

    def test_close_scan_keeps_unknown_cruise(self):
        nav = GridNavController(
            goal_xy=SHELF_APPROACH_XY,
            final_yaw=math.pi / 2.0,
            max_lin=2.4,
            max_ang=1.65,
            cruise_lin=0.30,
        )
        clear, amin, inc = _scan(hit=8.0)
        nav.step(1.92, 0.20, math.pi / 2.0, clear, amin, inc, dt=0.05, now=0.6, scan_id=1)
        blocked, amin, inc = _scan(hit=0.50)
        cap = nav._follow_speed_cap(1.92, 0.20, blocked, amin, inc)
        self.assertLessEqual(cap, 0.35)

    def test_yellow_lane_turns_west_when_shelf_is_ahead(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=2.4,
            max_ang=1.65,
            cruise_lin=0.30,
        )
        nav._commit_route(
            [
                (1.92, 2.32),
                (0.85, 2.32),
                (-0.40, 2.32),
                (-0.40, -2.70),
                DELIVERY_APPROACH_XY,
            ],
            1.92,
            2.32,
            "yellow",
        )
        ranges, amin, inc = _scan(hit_lo=-0.22, hit_hi=0.22, hit=0.35, clear=3.0)
        for index, _raw in enumerate(ranges):
            angle = wrap_to_pi(amin + index * inc)
            if abs(abs(angle) - math.pi / 2.0) < 0.20:
                ranges[index] = 0.24
        seen_ang = 0.0
        for step in range(10):
            _linear, angular, done, status = nav.step(
                1.92,
                2.32,
                math.pi / 2.0,
                ranges,
                amin,
                inc,
                dt=0.05,
                now=1.0 + 0.05 * step,
                scan_id=step + 1,
            )
            seen_ang = max(seen_ang, abs(angular))
            self.assertFalse(done)
            self.assertNotEqual(status, "TURN_STALLED")
            self.assertNotIn("recover", status.lower())
            self.assertNotEqual(status, "scan_stop")
        self.assertGreater(seen_ang, 0.10)

    def test_shelf_east_aisle_turns_west_before_racks(self):
        nav = GridNavController(
            goal_xy=SHELF_APPROACH_XY,
            final_yaw=math.pi / 2.0,
            max_lin=2.4,
            max_ang=1.65,
            cruise_lin=0.30,
        )
        nav._commit_route(
            [(1.92, 2.32), (0.85, 2.32), SHELF_APPROACH_XY],
            1.92,
            2.14,
            "shelf",
        )
        ranges, amin, inc = _scan(hit=8.0)
        linear, angular, done, status = nav.step(
            1.92, 2.14, math.pi / 2.0, ranges, amin, inc, dt=0.05, now=1.0, scan_id=1,
        )
        self.assertFalse(done)
        self.assertLessEqual(linear, 0.05)
        self.assertGreater(abs(angular), 0.10)
        self.assertNotEqual(status, "CENTER_GUARD_VIOLATION")

    def test_east_aisle_goes_north_to_yellow_lane_before_west(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=2.4,
            max_ang=1.65,
            cruise_lin=0.30,
        )
        nav._commit_route(
            [(1.7, 2.3), (1.25, 2.3), (-0.40, 2.32), DELIVERY_APPROACH_XY],
            1.91,
            1.93,
            "aisle",
        )
        first = nav.follower.waypoints[0]
        self.assertGreater(first[1], 2.20)
        self.assertLess(abs(first[0] - 1.91), 0.08)
        self.assertGreater(max(px for px, _py in nav.follower.waypoints), 1.20)

    def test_spawn_shelf_route_hits_yellow_lane_before_cutting_west(self):
        nav = GridNavController(
            goal_xy=SHELF_APPROACH_XY,
            final_yaw=math.pi / 2.0,
            max_lin=2.4,
            max_ang=1.65,
            cruise_lin=0.30,
        )
        north = [(1.9, y) for y in (-2.6, -1.7, -0.8, 0.1, 1.0, 1.9)]
        nav._commit_route(
            north + [(1.6, 2.4), (1.15, 2.4), SHELF_APPROACH_XY],
            1.92,
            -3.17,
            "start",
        )
        waypoints = nav.follower.waypoints
        self.assertTrue(
            any(abs(px - 1.92) < 0.15 and abs(py - WEST_LANE_Y) < 0.12 for px, py in waypoints)
        )
        westbound = [(px, py) for px, py in waypoints if px < 1.50]
        self.assertTrue(westbound)
        self.assertGreaterEqual(min(py for _px, py in westbound), WEST_LANE_Y - 0.12)

    def test_shelf_corner_already_facing_west_keeps_moving(self):
        nav = GridNavController(
            goal_xy=SHELF_APPROACH_XY,
            final_yaw=math.pi / 2.0,
            max_lin=2.4,
            max_ang=1.65,
            cruise_lin=0.30,
        )
        nav._commit_route(
            [(1.6, 2.4), (1.15, 2.4), SHELF_APPROACH_XY],
            1.85,
            2.43,
            "stuck",
        )
        ranges, amin, inc = _scan(hit=8.0)
        linear, angular, done, status = nav.step(
            1.85, 2.43, 2.99, ranges, amin, inc, dt=0.05, now=1.0, scan_id=1,
        )
        self.assertFalse(done)
        self.assertGreater(linear, 0.15)
        self.assertGreater(angular, 0.0)
        self.assertNotEqual(status, "CENTER_GUARD_VIOLATION")

    def test_turn_timer_does_not_reenter_recover_after_replan(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=2.4,
            max_ang=1.65,
            cruise_lin=0.30,
        )
        nav._commit_route(
            [(1.92, 2.32), (0.85, 2.32), DELIVERY_APPROACH_XY],
            1.92,
            2.32,
            "yellow",
        )
        nav._turn_t = 2.0
        nav._turn_yaw = math.pi / 2.0
        ranges, amin, inc = _scan(hit=0.35, clear=3.0)
        _linear, angular, _done, status = nav.step(
            1.92, 2.32, math.pi / 2.0, ranges, amin, inc, dt=0.05, now=2.0, scan_id=3,
        )
        self.assertNotEqual(status, "TURN_STALLED")
        self.assertNotEqual(nav.state, "RECOVER")
        self.assertGreater(abs(angular), 0.05)

    def test_explore_without_a_route_stays_at_one_x(self):
        nav = GridNavController(
            goal_xy=SHELF_APPROACH_XY,
            final_yaw=math.pi / 2.0,
            max_lin=2.4,
            max_ang=1.65,
            cruise_lin=0.30,
            explore_lin=1.20,
        )
        nav._observe_goal = (1.70, 0.40)
        ranges, amin, inc = _scan(hit=8.0)
        cap = nav._follow_speed_cap(1.92, 0.20, ranges, amin, inc)
        self.assertAlmostEqual(cap, 0.30, places=2)

    def test_explore_route_uses_four_x_not_eight_x(self):
        nav = GridNavController(
            goal_xy=SHELF_APPROACH_XY,
            final_yaw=math.pi / 2.0,
            max_lin=2.4,
            max_ang=1.65,
            cruise_lin=0.30,
            explore_lin=1.20,
        )
        nav._observe_goal = (1.92, 1.80)
        ranges, amin, inc = _scan(hit=8.0)
        self.assertTrue(nav._search(1.92, 0.20, math.pi / 2.0, "explore"))
        cap = nav._follow_speed_cap(1.92, 0.20, ranges, amin, inc)
        self.assertGreaterEqual(cap, 1.0)
        self.assertAlmostEqual(cap, 1.20, places=2)
        self.assertLess(cap, 2.0)

    def test_yellow_lane_divider_echo_is_not_a_hard_collision(self):
        planner = GlobalGridPlanner()
        self.assertFalse(planner.grid.is_static(1.56, 2.23))
        planner.grid.mark_circle(1.45, 2.00, 0.12, now=1.0)
        self.assertNotEqual(planner.grid.pose_collision(1.56, 2.23), "START_IN_COLLISION")
        route = planner.plan(1.56, 2.23, DELIVERY_APPROACH_XY, yaw=math.pi)
        self.assertIsNotNone(route)
        self.assertGreater(max(py for _px, py in route[:12]), 2.10)

    def test_yellow_lane_divider_echo_keeps_westbound_follow(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=2.4,
            max_ang=1.65,
            cruise_lin=0.30,
        )
        nav.planner.grid.mark_circle(1.45, 2.00, 0.12, now=1.0)
        nav._commit_route(
            [(1.56, 2.23), (0.80, 2.32), (-0.40, 2.32), DELIVERY_APPROACH_XY],
            1.56,
            2.23,
            "yellow",
        )
        ranges, amin, inc = _scan(hit=8.0)
        linear, angular, done, status = nav.step(
            1.56, 2.23, math.pi, ranges, amin, inc, dt=0.05, now=2.0, scan_id=1,
        )
        self.assertFalse(done)
        self.assertNotEqual(nav.state, "BLOCKED")
        self.assertNotEqual(status, "START_IN_COLLISION")
        self.assertGreater(abs(linear) + abs(angular), 0.04)

    def test_side_inflate_does_not_count_as_path_blocked(self):
        planner = GlobalGridPlanner()
        planner.grid.mark_circle(-1.48, 0.40, 0.16, now=1.0)
        path = [(-1.20, 0.20), (-1.20, -0.80), DELIVERY_APPROACH_XY]
        self.assertFalse(planner.path_blocked_ahead(-1.17, 0.60, path))

    def test_west_corridor_keeps_the_south_branch_instead_of_turning_back(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=2.4,
            max_ang=1.65,
            cruise_lin=0.30,
        )
        south_west = [(-1.17, 0.60), (-1.55, 0.45), (-1.70, -0.20), DELIVERY_APPROACH_XY]
        east = [(-1.17, 0.60), (-0.85, 0.60), (-0.40, 0.40), (-0.40, -2.70), DELIVERY_APPROACH_XY]
        nav._commit_route(south_west, -1.17, 0.60, "southwest")
        self.assertTrue(nav._rejects_reverse(east, -1.17, 0.60))
        plan_id = nav.plan_id
        ranges, amin, inc = _scan(hit=0.48)
        seen_ang = 0.0
        for step in range(16):
            yaw = -2.70 + 0.06 * step
            _linear, angular, done, status = nav.step(
                -1.17, 0.60, yaw, ranges, amin, inc, dt=0.05, now=0.8 + 0.05 * step, scan_id=step + 2,
            )
            seen_ang = max(seen_ang, abs(angular))
            self.assertFalse(done)
            self.assertNotEqual(nav.state, "BLOCKED")
            self.assertNotIn("explore", status.lower())
        self.assertEqual(nav.plan_id, plan_id)
        self.assertGreater(seen_ang, 0.10)

    def test_blocked_explore_does_not_resume_a_stale_route(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=2.4,
            max_ang=1.65,
            cruise_lin=0.30,
        )
        nav._commit_route(
            [(-1.17, 0.60), (-0.85, 0.60), DELIVERY_APPROACH_XY],
            -1.17,
            0.60,
            "east",
        )
        nav.state = "BLOCKED"
        nav.last_status = "BLOCKED explore"
        nav.follower = None
        nav.last_plan_s = 5.0
        ranges, amin, inc = _scan(hit=8.0)
        linear, angular, done, status = nav.step(
            -1.17, 0.60, 0.07, ranges, amin, inc, dt=0.05, now=5.0, scan_id=4,
        )
        self.assertEqual(nav.state, "BLOCKED")
        self.assertEqual(linear, 0.0)
        self.assertEqual(angular, 0.0)
        self.assertFalse(done)

    def test_plan_fail_keeps_a_live_south_route(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=2.4,
            max_ang=1.65,
            cruise_lin=0.30,
        )
        nav._commit_route(
            [(-1.76, 0.36), (-1.76, 0.00), (-1.76, -1.20), DELIVERY_APPROACH_XY],
            -1.76,
            0.36,
            "south",
        )
        nav.last_plan_s = 0.0
        nav._fail_plans = 3
        nav.planner.grid.mark_circle(-1.76, -0.90, 0.16, now=1.0)
        nav.planner.last_result.failure_reason = "NO_CONNECTIVITY"

        def _fail(*_args, **_kwargs):
            nav.planner.last_result.failure_reason = "NO_CONNECTIVITY"
            nav.planner.last_result.status = "FAIL"
            nav.planner.last_result.path_xy = []
            return nav.planner.last_result

        nav.planner.plan_to = _fail  # type: ignore[method-assign]
        ranges, amin, inc = _scan(hit=8.0)
        linear, angular, done, status = nav.step(
            -1.76, 0.36, -1.76, ranges, amin, inc, dt=0.05, now=3.0, scan_id=2,
        )
        self.assertFalse(done)
        self.assertNotEqual(nav.state, "BLOCKED")
        self.assertIsNotNone(nav.follower)
        self.assertGreater(abs(linear) + abs(angular), 0.04)

    def test_clear_lidar_does_not_crawl_when_the_map_prefix_is_short(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=2.4,
            max_ang=1.65,
            cruise_lin=0.30,
            explore_lin=1.20,
        )
        nav._commit_route(
            [(-1.77, 0.33), (-1.80, -0.15), (-1.80, -0.60), DELIVERY_APPROACH_XY],
            -1.77,
            0.33,
            "south",
        )
        ranges, amin, inc = _scan(hit=8.0)
        cap = nav._follow_speed_cap(-1.77, 0.33, ranges, amin, inc)
        self.assertGreaterEqual(cap, 1.0)

    def test_face_blocked_allows_leaving_the_dead_branch(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=2.4,
            max_ang=1.65,
            cruise_lin=0.30,
        )
        south = [(-1.82, -0.37), (-1.82, -1.20), DELIVERY_APPROACH_XY]
        north = [(-1.82, -0.37), (-1.82, 0.60), (-0.40, 0.60)]
        nav._commit_route(south, -1.82, -0.37, "south")
        self.assertTrue(nav._rejects_reverse(north, -1.82, -0.37))
        nav._face_blocked = True
        self.assertFalse(nav._rejects_reverse(north, -1.82, -0.37))
        retreat = nav._pick_retreat(-1.82, -0.37)
        self.assertIsNotNone(retreat)
        self.assertGreater(retreat[0], -1.40)

    def test_face_blocked_replans_away_from_a_box(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=2.4,
            max_ang=1.65,
            cruise_lin=0.30,
        )
        nav._commit_route(
            [(-1.82, -0.37), (-1.82, -1.20), DELIVERY_APPROACH_XY],
            -1.82,
            -0.37,
            "south",
        )
        nav.last_plan_s = 0.0
        plan_id = nav.plan_id
        ranges, amin, inc = _scan(hit=0.35)
        seen_leave = False
        for step in range(16):
            _linear, angular, done, status = nav.step(
                -1.82,
                -0.37,
                -math.pi / 2.0,
                ranges,
                amin,
                inc,
                dt=0.05,
                now=3.0 + 0.05 * step,
                scan_id=step + 2,
            )
            self.assertFalse(done)
            if nav.plan_id > plan_id or nav._observe_goal is not None or "retreat" in status:
                seen_leave = True
                break
        self.assertTrue(seen_leave)
        if nav._observe_goal is not None:
            self.assertGreater(nav._observe_goal[0], -1.50)
        elif nav.follower is not None:
            bearing = nav._path_bearing(nav.follower.waypoints, -1.82, -0.37)
            self.assertIsNotNone(bearing)
            self.assertGreater(abs(wrap_to_pi(bearing + math.pi / 2.0)), 0.60)


if __name__ == "__main__":
    unittest.main()
