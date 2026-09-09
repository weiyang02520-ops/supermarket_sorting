import unittest

from runtime.orchestrator import MissionOrchestrator, MissionState, MissionStateError
from runtime.scene_zones import (
    DELIVERY_TARGET_XYZ,
    TIME_LIMIT_S,
    in_delivery_base,
    in_delivery_box,
    in_picking_zone,
)
from runtime.task_protocol import parse_task_payload


def five_order():
    return parse_task_payload({
        "schema_version": 1,
        "run_prefix": "run_test",
        "count": 5,
        "targets": [
            {"id": "t1", "kind": "kele"},
            {"id": "t2", "kind": "pingguo"},
            {"id": "t3", "kind": "chengzi"},
            {"id": "t4", "kind": "zhijin"},
            {"id": "t5", "kind": "shupian"},
        ],
    })


class SceneZoneTests(unittest.TestCase):
    def test_official_table_is_inside_delivery_box(self):
        from runtime.scene_zones import SHELF_APPROACH_XY

        self.assertTrue(in_picking_zone((1.92, 2.475)))
        self.assertTrue(in_picking_zone(SHELF_APPROACH_XY))
        self.assertTrue(in_delivery_base((-1.94, -2.90)))
        self.assertTrue(in_delivery_box(DELIVERY_TARGET_XYZ))
        self.assertFalse(in_delivery_box((-1.94, -2.90, 0.80)))


class OrchestratorTests(unittest.TestCase):
    def test_pick_then_deliver_before_next_item(self):
        orch = MissionOrchestrator()
        orch.load_task(five_order())
        first = orch.start_pick()
        self.assertEqual(first.kind, "kele")
        with self.assertRaises(MissionStateError):
            orch.start_pick()
        orch.complete_pick(True)
        self.assertEqual(orch.state, MissionState.DELIVERING)
        with self.assertRaises(MissionStateError):
            orch.start_pick()
        orch.complete_deliver(True)
        second = orch.start_pick()
        self.assertEqual(second.kind, "pingguo")

    def test_failed_pick_retries_same_remaining_target(self):
        orch = MissionOrchestrator()
        orch.load_task(five_order())
        orch.start_pick()
        orch.complete_pick(False, "no detection")
        self.assertEqual(orch.state, MissionState.WAITING_TASK)
        self.assertEqual(orch.remaining()[0].kind, "kele")
        self.assertFalse(orch.holding)

    def test_failed_deliver_does_not_consume_target(self):
        orch = MissionOrchestrator()
        orch.load_task(five_order())
        orch.start_pick()
        orch.complete_pick(True)
        orch.complete_deliver(False, "dropped")
        self.assertFalse(orch.holding)
        self.assertEqual(len(orch.remaining()), 5)
        self.assertEqual(orch.start_pick().kind, "kele")

    def test_cannot_replace_different_task_during_open_cycle(self):
        orch = MissionOrchestrator()
        orch.load_task(five_order())
        orch.start_pick()
        resent = orch.load_task(five_order())
        self.assertEqual(resent.run_prefix, "run_test")
        self.assertEqual(orch.state, MissionState.PICKING)
        other = parse_task_payload({
            "schema_version": 1,
            "run_prefix": "run_other",
            "count": 1,
            "targets": [{"id": "x1", "kind": "kele"}],
        })
        with self.assertRaises(MissionStateError):
            orch.load_task(other)

    def test_task_resend_does_not_reset_records_or_clock(self):
        orch = MissionOrchestrator()
        orch.load_task(five_order())
        orch.start_pick()
        orch.complete_pick(True)
        orch.tick(12.0)
        orch.load_task(five_order())
        self.assertEqual(orch.elapsed_seconds, 12.0)
        self.assertEqual(orch.state, MissionState.DELIVERING)
        self.assertEqual(len(orch.records), 1)
        self.assertTrue(orch.holding)

    def test_delivery_blocked_keeps_holding_and_order(self):
        orch = MissionOrchestrator()
        orch.load_task(five_order())
        orch.start_pick()
        orch.complete_pick(True)
        orch.mark_delivery_blocked("scan_blocked")
        self.assertEqual(orch.state, MissionState.DELIVERING)
        self.assertTrue(orch.holding)
        self.assertEqual(len(orch.remaining()), 5)
        with self.assertRaises(MissionStateError):
            orch.start_pick()

    def test_unverified_place_blocks_next_order(self):
        orch = MissionOrchestrator()
        orch.load_task(five_order())
        orch.start_pick()
        orch.complete_pick(True)
        orch.note_place_unverified("c3_timeout")
        with self.assertRaises(MissionStateError):
            orch.start_pick()
        self.assertEqual(orch.state, MissionState.DELIVERING)
        self.assertEqual(len(orch.remaining()), 5)

    def test_hold_unknown_blocks_next_pick(self):
        orch = MissionOrchestrator()
        orch.load_task(five_order())
        orch.start_pick()
        orch.mark_hold_unknown("c2_unknown")
        with self.assertRaises(MissionStateError):
            orch.start_pick()

    def test_time_limit_matches_server_referee(self):
        orch = MissionOrchestrator()
        orch.load_task(five_order())
        self.assertEqual(orch.time_limit_seconds, TIME_LIMIT_S)
        self.assertEqual(orch.tick(TIME_LIMIT_S), MissionState.TIMEOUT)

    def test_five_successful_cycles_finish(self):
        orch = MissionOrchestrator()
        orch.load_task(five_order())
        kinds = []
        while orch.remaining():
            target = orch.start_pick()
            kinds.append(target.kind)
            orch.complete_pick(True)
            orch.complete_deliver(True)
        self.assertEqual(kinds, ["kele", "pingguo", "chengzi", "zhijin", "shupian"])
        self.assertEqual(orch.state, MissionState.DONE)
        self.assertEqual(len([r for r in orch.records if r.phase == "deliver" and r.success]), 5)


if __name__ == "__main__":
    unittest.main()
