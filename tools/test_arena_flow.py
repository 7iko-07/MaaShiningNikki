"""竞技场结束、错误传播及页面交接的离线回归测试。"""

import copy
import json
import subprocess
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from maa.controller import CustomController
from maa.custom_action import CustomAction
from maa.custom_recognition import CustomRecognition
from maa.define import MaaStatusEnum, Status, TaskDetail, LoggingLevelEnum
from maa.resource import Resource
from maa.tasker import Tasker

from tools.test_arena_weekly import ROOT, arena, load_action

navigation = load_action("page_navigation")
ARENA = json.loads((ROOT / "assets/resource/pipeline/arena.json").read_text(encoding="utf-8"))


def task_detail(succeeded):
    return TaskDetail(1, "test", [], Status(
        MaaStatusEnum.succeeded if succeeded else MaaStatusEnum.failed))


class ArenaActionTests(unittest.TestCase):
    def setUp(self):
        self.action = arena.ArenaCompareAction()
        self.context = Mock()
        self.context.override_next.return_value = True
        self.controller = self.context.tasker.controller
        self.controller.post_screencap.return_value.wait.return_value.succeeded = True
        self.controller.post_click.return_value.wait.return_value.succeeded = True
        self.context.run_recognition.return_value = SimpleNamespace(hit=False)
        self.context.run_task.return_value = task_detail(True)
        for patcher in (patch.object(arena, "in_settlement", return_value=False),
                        patch.object(arena.time, "sleep")):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.params = copy.deepcopy(ARENA["竞技场战力对比"]["action"]["param"]["custom_action_param"])
        self.params["max_rounds"] = 0

    def run_action(self, node="竞技场战力对比"):
        return self.action.run(self.context, SimpleNamespace(
            node_name=node, custom_action_param=json.dumps(self.params)))

    def test_both_strategies_finish_all_challenges_without_outer_wait(self):
        for node in ("竞技场战力对比", "竞技场稳守段位"):
            with self.subTest(node=node):
                self.action._read_challenge_times = Mock(return_value=3)
                self.action._challenge_once = Mock(return_value=arena.ChallengeOutcome.CHALLENGED)
                self.context.run_task.reset_mock()
                self.assertTrue(self.run_action(node))
                self.assertEqual(self.context.run_task.call_count, 3)
                self.context.override_next.assert_called_with(node, [])

    def test_zero_count_is_normal_stop(self):
        with patch.object(arena, "read_ocr_number", return_value=0):
            self.assertTrue(self.run_action())
        self.controller.post_click.assert_not_called()
        self.context.run_task.assert_not_called()

    def test_unknown_count_never_challenges_and_retries_are_finite(self):
        with patch.object(arena, "read_ocr_number", return_value=None) as read:
            self.assertFalse(self.run_action())
        self.assertEqual(read.call_count, 3)
        self.controller.post_click.assert_not_called()
        self.context.run_task.assert_not_called()

    def test_transient_count_ocr_failure_recovers(self):
        with patch.object(arena, "read_ocr_number", side_effect=[None, None, 0]) as read:
            self.assertTrue(self.run_action())
        self.assertEqual(read.call_count, 3)

    def test_missing_count_roi_fails_before_click(self):
        del self.params["剩余次数"]
        self.assertFalse(self.run_action())
        self.controller.post_click.assert_not_called()

    def test_known_stronger_opponents_stop_but_unreadable_power_fails(self):
        self.action._read_challenge_times = Mock(return_value=1)
        for own, opponent, expected in ((100, 200, True), (None, 200, False), (100, None, False)):
            with self.subTest(own=own, opponent=opponent):
                self.action._read_player_power = Mock(return_value=(own, object()))
                with patch.object(arena, "read_ocr_number", return_value=opponent):
                    self.assertEqual(self.run_action(), expected)
                self.controller.post_click.assert_not_called()
                self.context.run_task.assert_not_called()

    def test_override_failure_is_not_success_on_any_normal_exit(self):
        self.context.override_next.return_value = False
        self.action._read_challenge_times = Mock(return_value=0)
        self.assertFalse(self.run_action())
        self.action._read_challenge_times.return_value = 1
        for outcome in (arena.ChallengeOutcome.CHALLENGED, arena.ChallengeOutcome.STOPPED):
            self.action._challenge_once = Mock(return_value=outcome)
            self.assertFalse(self.run_action())
        with patch.object(arena, "in_settlement", return_value=True):
            self.assertFalse(self.run_action())

    def test_popup_outcomes_use_real_task_status(self):
        self.assertIs(self.action._handle_optional_ok_popup(self.context, self.controller, "ok", 0),
                      arena.PopupOutcome.ABSENT)
        self.context.run_task.assert_not_called()
        self.context.run_recognition.return_value.hit = True
        for result, expected in ((None, arena.PopupOutcome.FAILED),
                                 (task_detail(False), arena.PopupOutcome.FAILED),
                                 (task_detail(True), arena.PopupOutcome.HANDLED)):
            self.context.run_task.return_value = result
            self.assertIs(self.action._handle_optional_ok_popup(self.context, self.controller, "ok", 0),
                          expected)

    def test_failed_popup_stops_without_entering_result_flow(self):
        self.action._read_challenge_times = Mock(return_value=1)
        self.action._read_player_power = Mock(return_value=(100, object()))
        self.context.run_recognition.return_value.hit = True
        self.context.run_task.return_value = task_detail(False)
        with patch.object(arena, "read_ocr_number", return_value=10):
            self.assertFalse(self.run_action())
        self.context.run_task.assert_called_once_with("点击OK-竞技场")

    def test_failed_result_flow_stops_next_challenge(self):
        self.action._read_challenge_times = Mock(return_value=3)
        self.action._challenge_once = Mock(return_value=arena.ChallengeOutcome.CHALLENGED)
        self.context.run_task.return_value = task_detail(False)
        self.assertFalse(self.run_action())
        self.assertEqual(self.action._challenge_once.call_count, 1)

    def test_failed_opponent_click_never_enters_result_flow(self):
        self.action._read_challenge_times = Mock(return_value=1)
        self.action._read_player_power = Mock(return_value=(100, object()))
        self.controller.post_click.return_value.wait.return_value.succeeded = False
        with patch.object(arena, "read_ocr_number", return_value=10):
            self.assertFalse(self.run_action())
        self.context.run_task.assert_not_called()


class ScreenController(CustomController):
    """仅生成空白截图，通过识别回调模拟页面，不连接设备。"""
    def __init__(self):
        super().__init__()
        self.screen = ""
        self.on_click = lambda: None
        self.on_capture = lambda: None

    def connect(self):
        return True

    def request_uuid(self):
        return "arena-offline"

    def get_features(self):
        return 0

    def screencap(self):
        self.on_capture()
        return np.zeros((1280, 720, 3), dtype=np.uint8)

    def click(self, x, y):
        self.on_click()
        return True

    def unused(self, *args):
        return False

    start_app = stop_app = swipe = touch_down = touch_move = touch_up = unused
    click_key = input_text = key_down = key_up = unused


class ScreenRecognition(CustomRecognition):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def analyze(self, context, argv):
        if argv.node_name == self.controller.screen:
            return (10, 10, 20, 20)
        return None


class ScreenAction(CustomAction):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def run(self, context, argv):
        return self.callback(context, argv)


class ArenaPipelineTests(unittest.TestCase):
    """执行原生 Pipeline 调度，保留真实 next / JumpBack / Or，仅模拟画面和点击。"""
    def setUp(self):
        Tasker.set_stdout_level(LoggingLevelEnum.Off)
        self.resource = Resource()
        self.controller = ScreenController()
        self.assertTrue(self.controller.post_connection().wait().succeeded)
        self.tasker = Tasker()
        self.assertTrue(self.tasker.bind(self.resource, self.controller))
        self.assertTrue(self.resource.register_custom_recognition("screen", ScreenRecognition(self.controller)))
        self.assertTrue(self.resource.register_custom_action("screen_action", ScreenAction(self.advance)))
        self.transitions = {}
        self.executed = []
        self.nodes = {}
        for path in (ROOT / "assets/resource/pipeline").rglob("*.json"):
            self.nodes.update(json.loads(path.read_text(encoding="utf-8")))
        # Keep composite recognition and edges; replace image-dependent leaves.
        for name, node in self.nodes.items():
            recognition = node.get("recognition", "DirectHit")
            kind = recognition if isinstance(recognition, str) else recognition["type"]
            if kind not in ("DirectHit", "Or"):
                node["recognition"] = {"type": "Custom", "param": {"custom_recognition": "screen"}}
            node["action"] = {"type": "Custom", "param": {"custom_action": "screen_action"}}
            node["pre_delay"] = node["post_delay"] = 0
            node["rate_limit"] = 10
            node["timeout"] = 150
        self.assertTrue(self.resource.override_pipeline(self.nodes))

    def advance(self, context, argv):
        self.executed.append(argv.node_name)
        if argv.node_name in self.transitions:
            self.controller.screen = self.transitions[argv.node_name]
        return True

    def execute(self, entry, overrides=None):
        job = self.tasker.post_task(entry, overrides or {})
        deadline = time.monotonic() + 5
        while not job.done and time.monotonic() < deadline:
            time.sleep(.01)
        if not job.done:
            self.tasker.post_stop().wait()
            self.fail("Pipeline 未在测试期限内退出")
        return job.get()

    def test_challenge_leaves_finish_without_loading_wait(self):
        for node in ("竞技场战力对比", "竞技场稳守段位"):
            self.controller.screen = node
            self.assertTrue(self.execute(node).status.succeeded)

    def test_ok_leaf_returns_to_parent_and_does_not_wait_for_loading(self):
        self.controller.screen = "点击OK-竞技场"
        self.transitions["点击OK-竞技场"] = "竞技场挑战返回页面"
        overrides = {"test_parent": {"recognition": "DirectHit", "pre_delay": 0, "post_delay": 0,
                      "next": ["[JumpBack]点击OK-竞技场", "竞技场挑战返回页面"], "timeout": 150}}
        self.assertTrue(self.execute("test_parent", overrides).status.succeeded)
        self.assertEqual(self.executed, ["点击OK-竞技场", "竞技场挑战返回页面"])

    def test_persistent_loading_times_out(self):
        self.controller.screen = "在加载页面"
        self.assertFalse(self.execute("竞技场挑战结果处理").status.succeeded)
        self.assertNotIn("在加载页面", self.executed)

    def test_result_pages_finish_at_arena(self):
        self.controller.screen = "竞技场快速挑战结果页面"
        self.transitions = {"竞技场快速挑战结果页面": "竞技场挑战奖励页面",
                            "竞技场挑战奖励页面": "竞技场挑战返回页面"}
        self.assertTrue(self.execute("竞技场挑战结果处理").status.succeeded)
        self.assertEqual(self.executed, ["竞技场挑战结果处理", "竞技场快速挑战结果页面",
                                         "竞技场挑战奖励页面", "竞技场挑战返回页面"])

    def test_rank_popup_precedes_visible_arena_after_rewards(self):
        self.controller.screen = "竞技场挑战奖励页面"
        self.transitions = {"竞技场挑战奖励页面": "竞技场排名提升页面",
                            "竞技场排名提升页面": "竞技场挑战返回页面"}
        # The arena can remain visible behind the translucent rank popup.
        overrides = {"竞技场挑战返回页面": {"recognition": "DirectHit"}}
        self.assertTrue(self.execute("竞技场挑战奖励页面", overrides).status.succeeded)
        self.assertEqual(self.executed, ["竞技场挑战奖励页面", "竞技场排名提升页面",
                                         "竞技场挑战返回页面"])

    def test_rank_popup_action_failure_stops_result_flow(self):
        self.controller.screen = "竞技场挑战奖励页面"
        self.transitions = {"竞技场挑战奖励页面": "竞技场排名提升页面"}
        self.assertTrue(self.resource.register_custom_action(
            "failed_close", ScreenAction(lambda context, argv: False)))
        overrides = {
            "竞技场排名提升页面": {"action": {"type": "Custom", "param": {
                "custom_action": "failed_close"}}},
            "竞技场挑战返回页面": {"recognition": "DirectHit"},
        }
        self.assertFalse(self.execute("竞技场挑战奖励页面", overrides).status.succeeded)
        self.assertNotIn("竞技场挑战返回页面", self.executed)

    def test_delayed_weekly_page_is_processed_before_dispatch(self):
        captures = [0]
        def capture():
            captures[0] += 1
            if captures[0] == 3:
                self.controller.screen = "竞技场上周成绩页面"
        self.controller.on_capture = capture
        self.transitions = {"竞技场上周成绩页面": "竞技场每周奖励页面",
                            "竞技场每周奖励页面": "竞技场本周搭配主题页面",
                            "竞技场本周搭配主题页面": "竞技场挑战返回页面"}
        overrides = {"竞技场搭配分流": {"next": []}}
        self.assertTrue(self.execute("竞技场每周页面处理", overrides).status.succeeded)
        self.assertGreater(self.executed.index("竞技场搭配分流"),
                           self.executed.index("竞技场本周搭配主题页面"))

    def test_unready_weekly_page_does_not_dispatch(self):
        self.controller.screen = "unknown"
        self.assertFalse(self.execute("竞技场每周页面处理").status.succeeded)
        self.assertNotIn("竞技场搭配分流", self.executed)

    def test_navigation_hands_off_each_weekly_page_before_closing_it(self):
        self.assertTrue(self.resource.register_custom_action("page_navigate", navigation.PageNavigateAction()))
        self.assertTrue(self.resource.register_custom_action("navigation_click", navigation.NavigationClickAction()))
        for page in ("竞技场上周成绩页面", "竞技场每周奖励页面", "竞技场本周搭配主题页面"):
            with self.subTest(page=page):
                self.executed.clear()
                self.controller.screen = "导航在独自页面"
                self.controller.on_click = lambda: setattr(self.controller, "screen", page)
                overrides = {
                    "竞技场任务链准备": {"action": ARENA["竞技场任务链准备"]["action"], "next": []},
                    "竞技场导航点击入口": {"action": ARENA["竞技场导航点击入口"]["action"]},
                    "导航点击钻石竞技场": {"recognition": "DirectHit"},
                }
                self.assertTrue(self.execute("竞技场任务链准备", overrides).status.succeeded)
                self.assertEqual(self.controller.screen, page)
                self.assertNotIn(page, self.executed)


class NativePipelineProcessTests(unittest.TestCase):
    def test_native_pipeline_in_separate_client_process(self):
        # Importing maa.agent selects server mode. Native Tasker tests need a
        # fresh client process, independent of other offline test imports.
        code = (
            "from maa.library import Library; Library.version(); "
            "import unittest, sys; "
            "from tools.test_arena_flow import ArenaPipelineTests; "
            "suite = unittest.defaultTestLoader.loadTestsFromTestCase(ArenaPipelineTests); "
            "result = unittest.TextTestRunner(verbosity=2).run(suite); "
            "sys.exit(not result.wasSuccessful())"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, encoding="utf-8", errors="replace", timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([
        loader.loadTestsFromTestCase(ArenaActionTests),
        loader.loadTestsFromTestCase(NativePipelineProcessTests),
    ])


if __name__ == "__main__":
    unittest.main()
