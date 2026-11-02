"""日常采购限时查找、取消、异常及重入检查。"""

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.define import Rect

from tools.test_home_purchase_options import DEFAULTS, PIPELINE, configured_pipeline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))


def load(name):
    package = "_home_purchase_tests"
    if package not in sys.modules:
        parent = ModuleType(package)
        parent.__path__ = [str(ROOT / "agent/custom")]
        sys.modules[package] = parent
    spec = importlib.util.spec_from_file_location(f"{package}.{name}", ROOT / f"agent/custom/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with patch.object(AgentServer, "register_custom_action", return_value=True):
        spec.loader.exec_module(module)
    return module


home = load("home_purchase")
center = load("click_region")


class HomePurchaseFindTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.nodes = configured_pipeline(DEFAULTS)
        self.context = Mock()
        self.context.tasker.stopping = False
        self.context.tasker.controller.cached_image = object()
        self.context.tasker.controller.post_screencap.return_value.wait.side_effect = self.capture
        self.context.tasker.controller.post_click.return_value.wait.return_value.succeeded = True
        self.hit = SimpleNamespace(hit=True, box=Rect(84, 758, 130, 28))
        self.context.run_recognition.return_value = SimpleNamespace(hit=False)
        self.context.run_recognition_direct.return_value = SimpleNamespace(hit=False)
        self.context.run_task.side_effect = self.swipe
        self.context.get_node_data.side_effect = self.nodes.__getitem__
        self.context.override_next.side_effect = self.override_next
        self.node = "查找流心奶黄月饼"
        params = self.nodes[self.node]["action"]["param"]["custom_action_param"]
        self.argv = CustomAction.RunArg(
            task_detail=None, node_name=self.node, custom_action_name="home_purchase_find",
            custom_action_param=json.dumps(params), reco_detail=SimpleNamespace(hit=True), box=Rect(),
        )
        self.action = home.HomePurchaseFind()
        self.clock = patch.object(home.time, "monotonic", side_effect=lambda: self.now)
        self.sleep = patch.object(home.time, "sleep", side_effect=self.advance)
        self.clock.start()
        self.sleep.start()
        self.addCleanup(self.clock.stop)
        self.addCleanup(self.sleep.stop)

    def advance(self, seconds):
        self.now += seconds

    def capture(self):
        self.now += 0.1
        return SimpleNamespace(succeeded=True)

    def swipe(self, node):
        self.assertEqual(node, "上划日常采购")
        self.now += 1
        return SimpleNamespace(status=SimpleNamespace(succeeded=True))

    def override_next(self, node, successors):
        self.nodes[node]["next"] = successors
        return True

    def test_visible_target_clicks_first_ocr_box_once_without_rechecking(self):
        self.context.run_recognition.return_value = self.hit
        self.assertTrue(self.action.run(self.context, self.argv))
        self.assertEqual(self.nodes[self.node]["next"], ["点击购买流心奶黄月饼"])
        self.context.run_recognition.assert_called_once()
        self.context.tasker.controller.post_click.assert_called_once_with(149, 772)
        self.context.run_task.assert_not_called()
        self.context.run_action.assert_not_called()
        self.context.run_action_direct.assert_not_called()

    def test_find_after_swipe_and_timeout_skip(self):
        self.context.run_recognition.side_effect = [SimpleNamespace(hit=False), self.hit]
        self.assertTrue(self.action.run(self.context, self.argv))
        self.context.run_task.assert_called_once_with("上划日常采购")
        self.context.run_recognition.side_effect = None
        self.context.run_recognition.return_value = SimpleNamespace(hit=False)
        with patch.object(home.logger, "warning") as warning:
            self.assertTrue(self.action.run(self.context, self.argv))
        self.assertGreaterEqual(self.now, 30)
        self.assertLess(self.now, 33)
        warning.assert_called_once_with("未找到流心奶黄月饼，已跳过")
        self.assertEqual(self.nodes[self.node]["next"], ["采购玫玫山楂月饼分流"])

    def test_native_next_attributes_are_converted_to_node_names(self):
        self.nodes["点击流心奶黄月饼"]["next"] = [
            {"name": "点击购买流心奶黄月饼", "jump_back": False, "anchor": False},
        ]
        self.context.run_recognition.return_value = self.hit
        self.assertTrue(self.action.run(self.context, self.argv))
        self.assertEqual(self.nodes[self.node]["next"], ["点击购买流心奶黄月饼"])
        self.context.tasker.controller.post_click.assert_called_once_with(149, 772)

    def test_reentry_restores_found_path_after_timeout(self):
        self.assertTrue(self.action.run(self.context, self.argv))
        self.context.run_recognition.return_value = self.hit
        self.assertTrue(self.action.run(self.context, self.argv))
        self.assertEqual(self.nodes[self.node]["next"], ["点击购买流心奶黄月饼"])

    def test_later_visible_product_is_not_scrolled_past(self):
        self.context.run_recognition_direct.return_value = SimpleNamespace(hit=True)
        self.assertTrue(self.action.run(self.context, self.argv))
        self.context.run_task.assert_not_called()
        self.assertEqual(self.nodes[self.node]["next"], ["采购玫玫山楂月饼分流"])
        call = self.context.run_recognition_direct.call_args
        self.assertIn("^玫玫山楂月饼$", call.args[1].expected)
        self.assertNotIn("^电影票$", call.args[1].expected)

    def test_unselected_later_product_does_not_block_swiping(self):
        self.nodes = copy.deepcopy(PIPELINE)
        self.context.get_node_data.side_effect = self.nodes.__getitem__
        self.assertEqual(home.HomePurchaseFind._later_patterns(self.context, "采购玫玫山楂月饼分流"), [])

    def test_cancellation_does_not_click_or_report_missing(self):
        self.context.tasker.stopping = True
        with patch.object(home.logger, "warning") as warning:
            self.assertFalse(self.action.run(self.context, self.argv))
        self.context.run_task.assert_not_called()
        self.context.run_recognition.assert_not_called()
        self.context.tasker.controller.post_click.assert_not_called()
        warning.assert_not_called()

    def test_cancel_during_capture_stops_before_recognition(self):
        def cancel():
            self.context.tasker.stopping = True
            return SimpleNamespace(succeeded=True)
        self.context.tasker.controller.post_screencap.return_value.wait.side_effect = cancel
        self.assertFalse(self.action.run(self.context, self.argv))
        self.context.run_recognition.assert_not_called()
        self.context.tasker.controller.post_click.assert_not_called()

    def test_cancel_or_deadline_during_recognition_never_clicks(self):
        def cancel(*_):
            self.context.tasker.stopping = True
            return self.hit
        self.context.run_recognition.side_effect = cancel
        self.assertFalse(self.action.run(self.context, self.argv))
        self.context.tasker.controller.post_click.assert_not_called()
        self.context.tasker.stopping = False
        def slow(*_):
            self.now += 31
            return self.hit
        self.context.run_recognition.side_effect = slow
        self.assertTrue(self.action.run(self.context, self.argv))
        self.assertEqual(self.nodes[self.node]["next"], ["采购玫玫山楂月饼分流"])
        self.context.tasker.controller.post_click.assert_not_called()

    def test_failed_click_does_not_skip_or_report_success(self):
        self.context.run_recognition.return_value = self.hit
        self.context.tasker.controller.post_click.return_value.wait.return_value.succeeded = False
        with patch.object(home.logger, "warning") as warning:
            self.assertFalse(self.action.run(self.context, self.argv))
        self.context.tasker.controller.post_click.assert_called_once_with(149, 772)
        warning.assert_not_called()

    def test_capture_recognition_swipe_and_override_failures_are_errors(self):
        scenarios = ("capture", "recognition", "later_recognition", "swipe", "override")
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                if scenario == "capture":
                    self.context.tasker.controller.post_screencap.return_value.wait.side_effect = (
                        lambda: SimpleNamespace(succeeded=False))
                elif scenario == "recognition":
                    self.context.run_recognition.return_value = None
                elif scenario == "later_recognition":
                    self.context.run_recognition_direct.return_value = None
                elif scenario == "swipe":
                    self.context.run_task.side_effect = lambda _: SimpleNamespace(status=SimpleNamespace(succeeded=False))
                else:
                    self.context.override_next.side_effect = lambda *_: False
                with patch.object(home.logger, "error") as error, patch.object(home.logger, "warning") as warning:
                    self.assertFalse(self.action.run(self.context, self.argv))
                error.assert_called_once()
                warning.assert_not_called()
                self.context.tasker.controller.post_screencap.return_value.wait.side_effect = self.capture
                self.context.run_recognition.return_value = SimpleNamespace(hit=False)
                self.context.run_recognition_direct.return_value = SimpleNamespace(hit=False)
                self.context.run_task.side_effect = self.swipe

    def test_failed_skip_override_is_not_reported_as_skipped(self):
        self.context.override_next.side_effect = [True, False]
        with patch.object(home.logger, "warning") as warning:
            self.assertFalse(self.action.run(self.context, self.argv))
        warning.assert_not_called()

    def test_existing_center_click_uses_actual_ocr_box(self):
        context = Mock()
        context.tasker.controller.post_click.return_value.wait.return_value.succeeded = True
        argv = SimpleNamespace(box=Rect(73, 653, 149, 28), reco_detail=SimpleNamespace(hit=True))
        self.assertTrue(center.ClickCenterAction().run(context, argv))
        context.tasker.controller.post_click.assert_called_once_with(147, 667)
        context.run_action_direct.assert_not_called()
        context.tasker.controller.post_click.reset_mock()
        for hit, box in [(False, Rect(1, 2, 3, 4)), (True, Rect(1, 2, 0, 4)), (True, None)]:
            argv.reco_detail.hit, argv.box = hit, box
            self.assertFalse(center.ClickCenterAction().run(context, argv))
        context.tasker.controller.post_click.assert_not_called()


if __name__ == "__main__":
    unittest.main()
