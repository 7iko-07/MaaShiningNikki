"""活动签到弹窗的条件点击、就绪交接和累计超时。"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from maa.agent.agent_server import AgentServer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location(
    "activity_popup_under_test", ROOT / "agent/custom/click_and_wait_change.py"
)
click_module = importlib.util.module_from_spec(spec)
with patch.object(AgentServer, "register_custom_action", return_value=True):
    spec.loader.exec_module(click_module)
PIPELINE = json.loads((ROOT / "assets/resource/pipeline/caged_song.json").read_text(encoding="utf-8"))
PARAMS = PIPELINE["关闭笼中歌签到"]["action"]["param"]["custom_action_param"]


class ActivityPopupTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.context = Mock()
        self.context.tasker.stopping = False
        self.probe = self.context.clone.return_value
        self.controller = self.context.tasker.controller
        self.controller.cached_image = object()
        self.controller.post_screencap.return_value.wait.return_value.succeeded = True
        self.controller.post_click.return_value.wait.return_value.succeeded = True
        self.popup = lambda: False
        self.ready = lambda node: node == PARAMS["ready_nodes"][0]
        self.probe.run_recognition.side_effect = self.recognize
        self.click_times = []
        self.controller.post_click.side_effect = self.click
        for patcher in (
            patch.object(click_module.time, "monotonic", side_effect=lambda: self.now),
            patch.object(click_module.time, "sleep", side_effect=self.advance),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def advance(self, seconds):
        self.now += seconds

    def recognize(self, node, image, **kwargs):
        self.assertIs(image, self.controller.cached_image)
        if node == PARAMS["popup_node"]:
            return SimpleNamespace(hit=self.popup())
        self.assertEqual(kwargs, {"pipeline_override": {node: {"enabled": True}}})
        return SimpleNamespace(hit=self.ready(node))

    def click(self, x, y):
        self.assertEqual((x, y), (343, 142))
        self.click_times.append(self.now)
        return Mock(wait=Mock(return_value=SimpleNamespace(succeeded=True)))

    def run_action(self):
        return click_module.ClickAndWaitChangeAction().run(self.context, SimpleNamespace(
            custom_action_param=json.dumps(PARAMS), node_name="关闭笼中歌签到",
        ))

    def test_no_popup_and_ready_page_does_not_click_blank(self):
        self.assertTrue(self.run_action())
        self.controller.post_click.assert_not_called()
        self.assertEqual(self.now, 0)

    def test_ignored_click_retries_only_while_popup_is_still_visible(self):
        self.popup = lambda: self.controller.post_click.call_count < 2
        self.assertTrue(self.run_action())
        self.assertEqual(len(self.click_times), 2)
        self.assertGreaterEqual(self.click_times[1] - self.click_times[0], 0.5)
        names = [call.args[0] for call in self.probe.run_recognition.call_args_list]
        self.assertEqual(names[-1], PARAMS["ready_nodes"][0])
        self.assertTrue(all(name == PARAMS["popup_node"] for name in names[:-1]))

    def test_unknown_loading_page_waits_without_clicking_until_popup_arrives(self):
        self.popup = lambda: self.now >= 1 and not self.click_times
        self.ready = lambda node: bool(self.click_times)
        self.assertTrue(self.run_action())
        self.assertEqual(len(self.click_times), 1)
        self.assertGreaterEqual(self.click_times[0], 1)

    def test_popup_disappearing_is_not_enough_without_a_ready_page(self):
        self.popup = lambda: not self.click_times
        self.ready = lambda node: self.now >= 2 and node == PARAMS["ready_nodes"][0]
        self.assertTrue(self.run_action())
        self.assertEqual(len(self.click_times), 1)
        self.assertGreaterEqual(self.now, 2)
        self.context.run_recognition.assert_not_called()
        self.context.override_pipeline.assert_not_called()

    def test_stuck_popup_or_unknown_page_has_one_total_deadline(self):
        for visible in (False, True):
            with self.subTest(popup_visible=visible):
                self.now = 0
                self.click_times.clear()
                self.popup = lambda: visible
                self.ready = lambda node: False
                self.assertFalse(self.run_action())
                self.assertGreaterEqual(self.now, 30)
                self.assertLess(self.now, 30.2)
                self.assertTrue(all(b - a >= 0.5 for a, b in zip(self.click_times, self.click_times[1:])))
                self.assertEqual(bool(self.click_times), visible)

    def test_capture_failure_or_cancelled_task_never_clicks(self):
        self.popup = lambda: True
        self.controller.post_screencap.return_value.wait.return_value.succeeded = False
        self.assertFalse(self.run_action())
        self.context.tasker.stopping = True
        self.assertFalse(self.run_action())
        self.controller.post_click.assert_not_called()

    def test_click_failure_is_not_reported_as_completion(self):
        self.popup = lambda: True
        self.controller.post_click.side_effect = None
        self.controller.post_click.return_value.wait.return_value.succeeded = False
        self.assertFalse(self.run_action())
        self.controller.post_click.assert_called_once()

    def test_recognition_failure_or_deadline_cannot_cause_a_click(self):
        self.probe.run_recognition.side_effect = None
        self.probe.run_recognition.return_value = None
        self.assertFalse(self.run_action())

        def slow_recognition(*args, **kwargs):
            self.now += 31
            return SimpleNamespace(hit=True)

        self.probe.run_recognition.side_effect = slow_recognition
        self.assertFalse(self.run_action())
        self.controller.post_click.assert_not_called()


if __name__ == "__main__":
    unittest.main()
