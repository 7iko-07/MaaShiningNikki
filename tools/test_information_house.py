"""情报屋关卡查找、到底退出、取消及选项覆盖检查。"""

import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.define import Rect


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
PACKAGE = "_information_house_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "agent/custom")]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(
    f"{PACKAGE}.information_house_action", ROOT / "agent/custom/information_house_action.py",
)
house = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = house
with patch.object(AgentServer, "register_custom_action", return_value=True):
    spec.loader.exec_module(house)

PIPELINE = json.loads((ROOT / "assets/resource/pipeline/information_house.json").read_text(encoding="utf-8"))
OPTION = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))["option"]["获取情报对象选择"]


class FindTests(unittest.TestCase):
    def setUp(self):
        self.context = Mock()
        self.context.tasker.stopping = False
        self.controller = self.context.tasker.controller
        self.frame = np.zeros((1280, 720, 3), dtype=np.uint8)
        self.controller.cached_image = self.frame
        self.controller.post_screencap.return_value.wait.return_value.succeeded = True
        self.context.wait_freezes.return_value = True
        self.miss = SimpleNamespace(hit=False)
        self.hit = SimpleNamespace(hit=True, box=Rect(346, 897, 91, 35))
        self.context.run_recognition.return_value = self.miss
        self.context.run_action.return_value = SimpleNamespace(success=True)
        self.context.run_task.return_value = SimpleNamespace(status=SimpleNamespace(succeeded=True))
        self.action = house.InformationHouseFindAction()
        self.now = 0
        self.clock = patch.object(house.time, "monotonic", side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def run_find(self):
        return self.action.run(self.context, None)

    def test_visible_target_clicks_actual_box_without_scrolling(self):
        self.context.run_recognition.return_value = self.hit
        self.assertTrue(self.run_find())
        self.context.run_task.assert_not_called()
        self.context.run_action.assert_called_once_with("点击情报屋关卡", box=self.hit.box)

    def test_bottom_reports_failure_after_one_unchanged_swipe(self):
        with patch.object(house.logger, "error") as error:
            self.assertFalse(self.run_find())
        error.assert_called_once_with("情报屋：未找到所选关卡")
        self.context.run_task.assert_called_once_with("上滑情报屋")
        self.context.run_action.assert_not_called()

    def test_target_on_unchanged_final_screen_is_checked_before_bottom(self):
        self.context.run_recognition.side_effect = [self.miss, self.hit]
        self.assertTrue(self.run_find())
        self.context.run_task.assert_called_once_with("上滑情报屋")
        self.context.run_action.assert_called_once()

    def test_changed_list_keeps_scrolling_until_target_is_found(self):
        def swipe(_):
            self.frame[198:, :295] += 80
            return SimpleNamespace(status=SimpleNamespace(succeeded=True))
        self.context.run_task.side_effect = swipe
        self.context.run_recognition.side_effect = [self.miss, self.miss, self.hit]
        self.assertTrue(self.run_find())
        self.assertEqual(self.context.run_task.call_count, 2)

    def test_changes_outside_compare_roi_do_not_prevent_bottom_exit(self):
        def swipe(_):
            self.frame[:198] = 255
            self.frame[:, 295:] = 255
            return SimpleNamespace(status=SimpleNamespace(succeeded=True))
        self.context.run_task.side_effect = swipe
        self.assertFalse(self.run_find())
        self.context.run_task.assert_called_once_with("上滑情报屋")

    def test_small_pixel_noise_is_tolerated(self):
        def swipe(_):
            self.frame[198:, :295] = 5
            return SimpleNamespace(status=SimpleNamespace(succeeded=True))
        self.context.run_task.side_effect = swipe
        self.assertFalse(self.run_find())
        self.context.run_task.assert_called_once()

    def test_stability_screenshot_recognition_swipe_and_click_failures(self):
        scenarios = ("stable", "capture", "empty", "recognition", "swipe", "box", "click")
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                self.context.wait_freezes.return_value = scenario != "stable"
                self.controller.post_screencap.return_value.wait.return_value.succeeded = scenario != "capture"
                self.controller.cached_image = np.empty((0, 0, 3)) if scenario == "empty" else self.frame
                self.context.run_recognition.return_value = (
                    None if scenario == "recognition" else
                    SimpleNamespace(hit=True, box=Rect()) if scenario == "box" else
                    self.hit if scenario == "click" else self.miss
                )
                self.context.run_task.return_value.status.succeeded = scenario != "swipe"
                self.context.run_action.return_value.success = scenario != "click"
                with patch.object(house.logger, "error") as error:
                    self.assertFalse(self.run_find())
                error.assert_called_once()
                self.assertNotEqual(error.call_args.args[0], "情报屋：未找到所选关卡")

    def test_cancellation_never_clicks_or_reports_missing(self):
        for phase in ("before", "stable", "capture", "recognition", "swipe"):
            with self.subTest(phase=phase):
                self.context.reset_mock(side_effect=True)
                self.context.tasker.stopping = False
                def cancel(*_args, **_kwargs):
                    self.context.tasker.stopping = True
                    return True
                if phase == "before":
                    self.context.tasker.stopping = True
                elif phase == "stable":
                    self.context.wait_freezes.side_effect = cancel
                elif phase == "capture":
                    def capture():
                        cancel()
                        return SimpleNamespace(succeeded=True)
                    self.controller.post_screencap.return_value.wait.side_effect = capture
                elif phase == "recognition":
                    def recognize(*_):
                        cancel()
                        return self.hit
                    self.context.run_recognition.side_effect = recognize
                else:
                    def swipe(_):
                        cancel()
                        return SimpleNamespace(status=SimpleNamespace(succeeded=True))
                    self.context.run_task.side_effect = swipe
                with patch.object(house.logger, "error") as error:
                    self.assertFalse(self.run_find())
                self.context.run_action.assert_not_called()
                error.assert_not_called()

    def test_cumulative_deadline_stops_a_changing_list(self):
        def swipe(_):
            self.now += 31
            return SimpleNamespace(status=SimpleNamespace(succeeded=True))
        self.context.run_task.side_effect = swipe
        with patch.object(house.logger, "error") as error:
            self.assertFalse(self.run_find())
        error.assert_called_once_with("情报屋：查找所选关卡超时")
        self.context.run_task.assert_called_once()

    def test_reentry_does_not_reuse_previous_bottom_state(self):
        self.assertFalse(self.run_find())
        self.context.run_recognition.return_value = self.hit
        self.assertTrue(self.run_find())
        self.context.run_action.assert_called_once()


class InvestigationReturnTests(unittest.TestCase):
    def setUp(self):
        self.action = house.InformationHouseAutoInvestigateAction()
        self.context = Mock()
        self.context.tasker.stopping = False
        self.controller = self.context.tasker.controller
        self.controller.post_screencap.return_value.wait.return_value.succeeded = True
        self.controller.post_click.return_value.wait.return_value.succeeded = True
        self.miss = SimpleNamespace(hit=False)
        self.hit = SimpleNamespace(hit=True, box=Rect(24, 70, 36, 37))
        self.now = 0
        for item in (patch.object(house.time, "monotonic", side_effect=lambda: self.now),
                     patch.object(house.time, "sleep", side_effect=self.advance)):
            item.start()
            self.addCleanup(item.stop)

    def advance(self, seconds):
        self.now += seconds

    def test_waits_for_travel_then_returns_and_confirms_investigation(self):
        self.context.run_recognition_direct.side_effect = [self.miss, self.hit, self.miss]
        self.context.run_recognition.side_effect = [self.miss, self.hit, self.hit]
        self.assertTrue(self.action._return_to_investigation(self.context, self.controller))
        self.controller.post_click.assert_called_once_with(42, 88)
        self.assertEqual([call.args[0] for call in self.context.run_recognition.call_args_list],
                         ["点击调查中", "导航点击返回", "点击调查中"])
        params = self.context.run_recognition.call_args.kwargs["pipeline_override"]
        self.assertEqual(params["点击调查中"]["recognition"]["param"]["expected"], ["调查中"])

    def test_retry_clicks_only_while_travel_is_recognized(self):
        self.context.run_recognition_direct.side_effect = lambda *_: self.hit if self.now < 2 else self.miss
        self.context.run_recognition.side_effect = lambda node, *_args, **_kwargs: (
            self.hit if node == "导航点击返回" or self.now >= 3 else self.miss
        )
        clicks = []
        click = self.controller.post_click
        click.side_effect = lambda *_: clicks.append(self.now) or click.return_value
        self.assertTrue(self.action._return_to_investigation(self.context, self.controller))
        self.assertEqual(clicks, [0, 1.5])

    def test_unknown_page_times_out_without_clicking(self):
        self.context.run_recognition_direct.return_value = self.miss
        self.context.run_recognition.return_value = self.miss
        self.assertFalse(self.action._return_to_investigation(self.context, self.controller))
        self.controller.post_click.assert_not_called()


class OptionTests(unittest.TestCase):
    def test_all_17_choices_are_distinct_full_title_matches(self):
        names = {
            "秦衣·下", "江西桐", "佐森", "佑果", "佐森佑果", "叶维冬", "洛洛梨",
            "秦衣·上", "夜宵·上", "夜宵·下", "倾羽墨", "收视搭档", "海哲",
            "夏梦露", "左一", "夜骸", "叶苏夏",
        }
        self.assertEqual(len(OPTION["cases"]), 17)
        self.assertEqual({case["name"] for case in OPTION["cases"]}, names)
        self.assertEqual(OPTION["default_case"], "左一")
        for case in OPTION["cases"]:
            pattern, = case["pipeline_override"]["点击情报屋关卡"]["recognition"]["param"]["expected"]
            for name in names:
                self.assertEqual(bool(re.search(pattern, name.replace("·", ""))), name == case["name"])
            self.assertFalse(re.search(pattern, "奖励预览" + case["name"].replace("·", "")))

    def test_shared_flow_uses_selected_roi_and_existing_investigation_successors(self):
        params = PIPELINE["点击情报屋关卡"]["recognition"]["param"]
        self.assertEqual(params["roi"], [340, 205, 380, 1075])
        self.assertEqual(PIPELINE["情报屋任务链入口"]["next"], ["查找情报屋关卡"])
        self.assertEqual(PIPELINE["查找情报屋关卡"]["next"], ["点击托管调查", "点击调查中"])
        self.assertNotIn("next", PIPELINE["上滑情报屋"])


if __name__ == "__main__":
    unittest.main()
