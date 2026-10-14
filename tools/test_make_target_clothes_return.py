"""State transitions, cancellation and deadlines for layered clothes returns."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from maa.define import Rect
from tools.test_time_corridor_challenge import module


class ClothingReturnTests(unittest.TestCase):
    def simulate(self, frames, *, button=True, action_success=True, cancel_after_click=False,
                 slow_recognition=False, button_box=(10, 20, 30, 40), failed_recognitions=()):
        context = Mock()
        context.tasker.stopping = False
        controller = context.tasker.controller
        controller.post_screencap.return_value.wait.return_value.status.succeeded = True
        now = [0.0]
        captures = []
        clicks = []
        self.click_times = []

        def capture():
            frame = frames[min(len(captures), len(frames) - 1)]
            captures.append(frame)
            controller.cached_image = frame
            return SimpleNamespace(wait=lambda: SimpleNamespace(status=SimpleNamespace(succeeded=True)))

        def recognize(node, image):
            if slow_recognition:
                now[0] += .3
            if node in failed_recognitions:
                return None
            hit = node in image
            if node == "back":
                hit = button(image) if callable(button) else button
            box = Rect(*button_box) if button_box is not None else None
            return SimpleNamespace(hit=hit, box=box)

        def click(kind, params):
            clicks.append((controller.cached_image, params.target))
            self.click_times.append(now[0])
            if cancel_after_click:
                context.tasker.stopping = True
            return SimpleNamespace(success=action_success)

        controller.post_screencap.side_effect = capture
        context.run_recognition.side_effect = recognize
        context.run_action_direct.side_effect = click
        argv = SimpleNamespace(custom_action_param=json.dumps({
            "ready_node": "ready", "back_node": "back",
            "timeout": 1000,
        }))
        with patch.object(module.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(module.time, "sleep", side_effect=lambda dt: now.__setitem__(0, now[0] + dt)):
            success = module.MakeTargetClothesReturnAction().run(context, argv)
        return success, clicks, captures, now[0]

    def test_layers_require_fresh_confirmation_and_loading_never_clicks(self):
        frames = [{"challenge"}] * 2 + [{"在加载页面", "stages"}] + [{"stages"}] * 2 + [{"ready"}] * 2
        success, clicks, _, _ = self.simulate(frames)
        self.assertTrue(success)
        self.assertEqual([f for f, _ in clicks], [{"challenge"}, {"stages"}])

    def test_visible_ready_background_does_not_skip_material_popup(self):
        popup = {"制衣返回材料弹窗识别", "ready"}
        success, clicks, _, _ = self.simulate([popup] * 2 + [{"ready"}] * 2)
        self.assertTrue(success)
        self.assertEqual(clicks, [(popup, [304, 144, 71, 36])])

    def test_already_ready_needs_two_non_loading_observations(self):
        success, clicks, captures, _ = self.simulate([
            {"ready"}, {"ready", "在加载页面"}, {"ready"}, {"ready"},
        ])
        self.assertTrue(success)
        self.assertEqual(len(captures), 4)
        self.assertEqual(clicks, [])

    def test_unknown_map_after_first_return_can_continue_to_clothes(self):
        frames = [{"challenge"}] * 2 + [{"unlisted_map"}] * 2 + [{"ready"}] * 2
        success, clicks, _, _ = self.simulate(frames)
        self.assertTrue(success)
        self.assertEqual([f for f, _ in clicks], [{"challenge"}, {"unlisted_map"}])
        self.assertAlmostEqual(self.click_times[1] - self.click_times[0], .2)

    def test_unknown_screen_with_visible_back_retries_until_total_timeout(self):
        success, clicks, _, elapsed = self.simulate([set()])
        self.assertFalse(success)
        self.assertGreater(len(clicks), 1)
        self.assertAlmostEqual(elapsed, 1)
        for previous, current in zip(self.click_times, self.click_times[1:]):
            self.assertGreaterEqual(current - previous, .2 - 1e-9)

    def test_persistent_source_retries_without_resetting_deadline(self):
        success, clicks, _, elapsed = self.simulate([{"challenge"}])
        self.assertFalse(success)
        self.assertGreater(len(clicks), 1)
        self.assertAlmostEqual(elapsed, 1)

    def test_second_return_recovers_ineffective_first_click(self):
        success, clicks, _, _ = self.simulate([{"challenge"}] * 4 + [{"ready"}] * 2)
        self.assertTrue(success)
        self.assertEqual(len(clicks), 2)
        self.assertAlmostEqual(self.click_times[1] - self.click_times[0], .2)

    def test_return_button_needs_two_consecutive_fresh_hits(self):
        frames = [{"back"}, set(), {"back"}, {"ready"}, {"ready"}]
        success, clicks, _, _ = self.simulate(frames, button=lambda image: "back" in image)
        self.assertTrue(success)
        self.assertEqual(clicks, [])

    def test_loading_breaks_return_button_confirmation(self):
        frames = [{"back"}, {"back", "在加载页面"}, {"back"}, {"ready"}, {"ready"}]
        success, clicks, _, _ = self.simulate(frames)
        self.assertTrue(success)
        self.assertEqual(clicks, [])

    def test_arrival_during_retry_interval_does_not_click_again(self):
        success, clicks, _, _ = self.simulate([{"challenge"}] * 2 + [{"ready"}] * 2)
        self.assertTrue(success)
        self.assertEqual(len(clicks), 1)

    def test_unknown_map_then_material_popup_then_clothes(self):
        popup = {"制衣返回材料弹窗识别", "ready"}
        frames = [{"challenge"}] * 2 + [set()] * 2 + [popup] * 2 + [{"ready"}] * 2
        success, clicks, _, _ = self.simulate(frames)
        self.assertTrue(success)
        self.assertEqual([target for _, target in clicks], [
            [10, 20, 30, 40], [10, 20, 30, 40], [304, 144, 71, 36],
        ])

    def test_loading_after_click_does_not_reset_total_deadline(self):
        success, clicks, _, elapsed = self.simulate([{"challenge"}] * 2 + [{"在加载页面"}])
        self.assertFalse(success)
        self.assertEqual(len(clicks), 1)
        self.assertAlmostEqual(elapsed, 1)

    def test_missing_button_and_expired_recognition_never_click(self):
        for options in ({"button": False}, {"slow_recognition": True},
                        {"button_box": None}, {"button_box": (10, 20, 0, 40)}):
            with self.subTest(options=options):
                success, clicks, _, _ = self.simulate([{"challenge"}], **options)
                self.assertFalse(success)
                self.assertEqual(clicks, [])

    def test_failed_recognition_does_not_mean_target_is_absent(self):
        for node in ("在加载页面", "制衣返回材料弹窗识别", "ready", "back"):
            with self.subTest(node=node):
                success, clicks, _, _ = self.simulate([set()], failed_recognitions=[node])
                self.assertFalse(success)
                self.assertEqual(clicks, [])

    def test_click_failure_and_cancellation_stop_the_chain(self):
        for options in ({"action_success": False}, {"cancel_after_click": True}):
            with self.subTest(options=options):
                success, clicks, _, _ = self.simulate([{"challenge"}] * 2 + [{"stages"}] * 2, **options)
                self.assertFalse(success)
                self.assertEqual(len(clicks), 1)

    def test_state_does_not_leak_between_runs(self):
        for _ in range(2):
            success, clicks, _, _ = self.simulate([{"challenge"}] * 2 + [{"ready"}] * 2)
            self.assertTrue(success)
            self.assertEqual(len(clicks), 1)


if __name__ == "__main__":
    unittest.main()
