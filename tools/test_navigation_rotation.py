"""Rotation guard regressions: no capture or click while display settles."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools.test_page_navigation import navigation, ROOT, WithoutExitPopupTests
from tools.validate_schema import load_jsonc


class RotationGuardTests(WithoutExitPopupTests):
    def setUp(self):
        super().setUp()
        self.action = navigation.NavigationClickAction()
        self.context = Mock()
        self.context.tasker.stopping = False
        self.now = 0.0

    def advance(self, seconds):
        self.now += seconds

    def shell(self, value):
        return SimpleNamespace(success=True, result=SimpleNamespace(
            success=True, output=f"SurfaceOrientation: {value}"))

    def run_guard(self, values, deadline=15):
        values = iter(values)
        last = [0]

        def query(*args, **kwargs):
            last[0] = next(values, last[0])
            timeout = kwargs["pipeline_override"]["导航查询屏幕方向"]["action"]["param"]["shell_timeout"]
            self.assertLessEqual(timeout, max(1, int((min(12, deadline) - self.now) * 1000)))
            self.advance(0.05)
            return self.shell(last[0])

        self.context.run_action.side_effect = query
        with patch.object(navigation.time, "monotonic", side_effect=lambda: self.now), \
             patch.object(navigation.time, "sleep", side_effect=self.advance):
            result = self.action._wait_landscape(self.context, deadline)
        self.context.tasker.controller.post_screencap.assert_not_called()
        self.context.tasker.controller.post_connection.assert_not_called()
        self.context.run_action_direct.assert_not_called()
        return result

    def test_waits_for_one_landscape_orientation_half_second(self):
        self.assertTrue(self.run_guard([0, 0, 1]))
        self.assertGreaterEqual(self.now, 0.85)
        self.assertLess(self.now, 1)

    def test_portrait_resets_stability(self):
        self.assertTrue(self.run_guard([1, 1, 1, 0, 1]))
        self.assertGreaterEqual(self.now, 1.15)

    def test_change_between_landscape_directions_resets_stability(self):
        self.assertTrue(self.run_guard([1, 1, 1, 3]))
        self.assertGreaterEqual(self.now, 1)

    def test_portrait_times_out_without_resuming_capture(self):
        self.assertFalse(self.run_guard([0]))
        self.assertLessEqual(self.now, 12.05)

    def test_click_deadline_is_shared_and_cannot_return_late_success(self):
        self.assertFalse(self.run_guard([1], deadline=0.45))
        self.assertLessEqual(self.now, 0.5)

    def test_cancellation_does_not_query(self):
        self.context.tasker.stopping = True
        self.assertFalse(self.action._wait_landscape(self.context, 15))
        self.context.run_action.assert_not_called()

    def test_failed_missing_ambiguous_and_exception_query_stop(self):
        cases = [None, SimpleNamespace(success=False),
                 SimpleNamespace(success=True, result=SimpleNamespace(success=False)),
                 self.shell("unknown"), self.shell("1\nSurfaceOrientation: 3")]
        for result in cases:
            with self.subTest(result=result):
                self.context.run_action.return_value = result
                self.assertFalse(self.action._wait_landscape(self.context, navigation.time.monotonic() + 15))
        self.context.run_action.side_effect = RuntimeError("shell failed")
        self.assertFalse(self.action._wait_landscape(self.context, navigation.time.monotonic() + 15))

    def test_guard_runs_after_click_and_before_next_capture(self):
        events = []
        argv = SimpleNamespace(node_name="entry", custom_action_param=json.dumps({
            "source_nodes": ["home"], "target_nodes": ["dream"], "wait_landscape": True}))
        self.context.run_recognition.return_value = SimpleNamespace(
            hit=True, box=SimpleNamespace(x=1, y=2, w=3, h=4))

        def screenshot(*args):
            events.append("capture")
            return "dream" if "guard" in events else "home"

        def click(*args):
            events.append("click")
            return SimpleNamespace(success=True)

        def guard(*args):
            events.append("guard")
            return True

        def refresh(*args):
            events.append("refresh")
            return True

        self.context.run_action_direct.side_effect = click
        with patch.object(navigation.PageNavigateAction, "_screencap", side_effect=screenshot), \
             patch.object(navigation.PageNavigateAction, "_recognize", side_effect=lambda ctx, image, node: image == node), \
             patch.object(navigation.PageNavigateAction, "_intermediate_reason", return_value=""), \
             patch.object(self.action, "_wait_landscape", side_effect=guard), \
             patch.object(self.action, "_refresh_landscape", side_effect=refresh):
            self.assertTrue(self.action.run(self.context, argv))
        self.assertEqual(events, ["capture", "click", "guard", "refresh", "capture"])

    def test_refresh_reconnects_before_capturing_new_dimensions(self):
        events = []
        controller = self.context.tasker.controller
        controller.post_connection.return_value.wait.side_effect = lambda: (
            events.append("reconnect") or SimpleNamespace(succeeded=True))
        controller.post_screencap.return_value.wait.side_effect = lambda: (
            events.append("capture") or SimpleNamespace(succeeded=True))
        controller.cached_image = SimpleNamespace(shape=(720, 1280, 3))
        self.assertTrue(self.action._refresh_landscape(self.context, navigation.time.monotonic() + 15))
        self.assertEqual(events, ["reconnect", "capture"])

    def test_failed_reconnect_never_captures(self):
        controller = self.context.tasker.controller
        controller.post_connection.return_value.wait.return_value = SimpleNamespace(succeeded=False)
        self.assertFalse(self.action._refresh_landscape(self.context, navigation.time.monotonic() + 15))
        controller.post_screencap.assert_not_called()

    def test_entry_retry_does_not_repeat_landscape_reconnect(self):
        argv = SimpleNamespace(node_name="entry", custom_action_param=json.dumps({
            "source_nodes": ["home"], "target_nodes": ["dream"], "wait_landscape": True}))
        self.context.run_recognition.return_value = SimpleNamespace(
            hit=True, box=SimpleNamespace(x=1, y=2, w=3, h=4))
        self.context.run_action_direct.return_value = SimpleNamespace(success=True)
        with patch.object(navigation.time, "monotonic", side_effect=lambda: self.now), \
             patch.object(navigation.time, "sleep", side_effect=self.advance), \
             patch.object(navigation.PageNavigateAction, "_screencap", return_value=object()), \
             patch.object(navigation.PageNavigateAction, "_recognize", side_effect=lambda ctx, image, node: node == "home" or self.now >= 2), \
             patch.object(navigation.PageNavigateAction, "_intermediate_reason", return_value=""), \
             patch.object(self.action, "_wait_landscape", return_value=True) as guard, \
             patch.object(self.action, "_refresh_landscape", return_value=True) as refresh:
            self.assertTrue(self.action.run(self.context, argv))
        self.assertEqual(self.context.run_action_direct.call_count, 2)
        guard.assert_called_once()
        refresh.assert_called_once()

    def test_failed_or_portrait_capture_cannot_pass_refresh(self):
        controller = self.context.tasker.controller
        controller.post_connection.return_value.wait.return_value = SimpleNamespace(succeeded=True)
        for succeeded, image in ((False, SimpleNamespace(shape=(720, 1280, 3))),
                                 (True, SimpleNamespace(shape=(1280, 720, 3))), (True, None)):
            with self.subTest(succeeded=succeeded, image=image):
                controller.post_screencap.return_value.wait.return_value = SimpleNamespace(succeeded=succeeded)
                controller.cached_image = image
                self.assertFalse(self.action._refresh_landscape(self.context, navigation.time.monotonic() + 15))

    def test_reconnect_past_deadline_never_captures(self):
        controller = self.context.tasker.controller
        def slow_connection():
            self.now = 16
            return SimpleNamespace(succeeded=True)
        controller.post_connection.return_value.wait.side_effect = slow_connection
        with patch.object(navigation.time, "monotonic", side_effect=lambda: self.now):
            self.assertFalse(self.action._refresh_landscape(self.context, 15))
        controller.post_screencap.assert_not_called()

    def test_capture_past_deadline_cannot_report_late_arrival(self):
        argv = SimpleNamespace(node_name="entry", custom_action_param=json.dumps({
            "source_nodes": ["home"], "target_nodes": ["dream"]}))
        def slow_capture(*args):
            self.now = 31
            return "dream"
        with patch.object(navigation.time, "monotonic", side_effect=lambda: self.now), \
             patch.object(navigation.PageNavigateAction, "_screencap", side_effect=slow_capture), \
             patch.object(navigation.PageNavigateAction, "_recognize", return_value=True) as recognize:
            self.assertFalse(self.action.run(self.context, argv))
        recognize.assert_not_called()
        self.context.run_action_direct.assert_not_called()

    def test_guard_failure_stops_without_another_capture_or_retry(self):
        argv = SimpleNamespace(node_name="entry", custom_action_param=json.dumps({
            "source_nodes": ["home"], "target_nodes": ["dream"], "wait_landscape": True}))
        self.context.run_recognition.return_value = SimpleNamespace(
            hit=True, box=SimpleNamespace(x=1, y=2, w=3, h=4))
        self.context.run_action_direct.return_value = SimpleNamespace(success=True)
        with patch.object(navigation.PageNavigateAction, "_screencap", return_value="home") as cap, \
             patch.object(navigation.PageNavigateAction, "_recognize", side_effect=lambda ctx, image, node: node == "home"), \
             patch.object(navigation.PageNavigateAction, "_intermediate_reason", return_value=""), \
             patch.object(self.action, "_wait_landscape", return_value=False):
            self.assertFalse(self.action.run(self.context, argv))
        cap.assert_called_once()
        self.context.run_action_direct.assert_called_once()

    def test_only_dream_entry_enables_guard_and_shell_has_no_successors(self):
        nodes = load_jsonc(ROOT / "assets/resource/pipeline/navigation.json")
        guarded = [name for name, node in nodes.items()
                   if isinstance(node.get("action"), dict) and
                   node["action"].get("param", {}).get("custom_action_param", {}).get("wait_landscape")]
        self.assertEqual(guarded, ["导航识别并点击织梦空间"])
        self.assertNotIn("next", nodes["导航查询屏幕方向"])


if __name__ == "__main__":
    unittest.main()
