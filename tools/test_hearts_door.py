"""心之门免费优先、累计次数及失败停止的离线回归。"""

import importlib.util
import json
import subprocess
import sys
import unittest
from collections import Counter
from copy import deepcopy
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from maa.agent.agent_server import AgentServer
from maa.define import Rect


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location(
    "hearts_door_under_test", ROOT / "agent/custom/hearts_door.py",
)
module = importlib.util.module_from_spec(spec)
with patch.object(AgentServer, "register_custom_action", return_value=True):
    spec.loader.exec_module(module)

PIPELINE = json.loads((ROOT / "assets/resource/pipeline/hearts_door.json").read_text(encoding="utf-8"))
INTERFACE = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))


class HeartsDoorTests(unittest.TestCase):
    def setUp(self):
        self.action = module.HeartsDoorAction()
        self.source = MagicMock()
        self.source.tasker.stopping = False
        self.config = {module.FREE_NODE: {"enabled": True},
                       "点击谜之海": {"enabled": True},
                       module.RESULT_NODE: {"enabled": True}}
        self.source.get_node_data.side_effect = lambda node: self.config[node]
        self.source.clone.side_effect = self.clone
        self.free = {"绮海": True, "幻海": True, "谜海": False}
        self.sea = "谜海"
        self.pending = False
        self.events = []
        self.checks = Counter()
        self.now = 0
        self.hide_tabs = 0
        self.no_result = self.stuck_close = self.insufficient = False
        self.loading = self.invalid_box = False
        self.failed_action = self.failed_recognition = None
        self.sharing_succeeds = self.screenshot_succeeds = True
        self.share_closes_result = False
        self.display = self.display_after_result = False
        self.stuck_display = False
        self.designer_after_draw = self.stuck_designer = False
        self.designer_clicks = 0
        for name, replacement in (("sleep", self.sleep), ("monotonic", lambda: self.now)):
            mock = patch.object(module.time, name, replacement)
            mock.start()
            self.addCleanup(mock.stop)

    def sleep(self, seconds):
        self.now += seconds

    def clone(self):
        self.context = MagicMock()
        self.context.tasker.stopping = False
        self.context.tasker.controller.post_screencap.return_value.wait.return_value.succeeded = self.screenshot_succeeds
        self.context.override_pipeline.side_effect = self.override
        self.context.get_node_data.side_effect = lambda node: PIPELINE[node]
        self.context.run_recognition.side_effect = self.recognize
        self.context.run_action.side_effect = self.operate
        self.context.run_action_direct.side_effect = self.click
        self.context.run_task.side_effect = self.share
        self.enabled = {node: data["enabled"] for node, data in self.config.items()}
        self.target = "绮海"
        return self.context

    def override(self, data):
        for node, fields in data.items():
            if "enabled" in fields:
                self.enabled[node] = fields["enabled"]
        if "心之门阁名" in data:
            self.target = data["心之门阁名"]["recognition"]["param"]["expected"][0]
        return True

    def recognize(self, node, image):
        self.checks[(node, self.sea)] += 1
        if node == self.failed_recognition or not self.enabled.get(node, True):
            return None
        if node in ("心之门阁名", "点击谜之海"):
            hit = self.hide_tabs == 0
        elif node == module.RESULT_NODE:
            hit = self.pending and not self.display and not self.no_result and not self.insufficient
        elif node == module.READY_NODE:
            hit = not self.pending and not self.display and not self.loading
        elif node == module.DISPLAY_NODE:
            hit = self.display
        elif node == module.DESIGNER_NODE:
            hit = self.pending and self.designer_clicks == 3
        elif node == module.FREE_NODE:
            hit = self.free[self.sea]
        elif node == "点击体力用尽确定":
            hit = self.pending and self.insufficient
        else:
            hit = self.sea == "谜海" and not self.pending
        box = Rect(101, 1061, 63, 34) if not self.invalid_box else Rect()
        return SimpleNamespace(hit=hit, box=box)

    def operate(self, node, box):
        if node == self.failed_action:
            return SimpleNamespace(success=False)
        if node == "左划心之门的阁":
            self.events.append(("swipe", self.sea))
            self.hide_tabs = max(0, self.hide_tabs - 1)
        elif node in ("心之门阁名", "点击谜之海"):
            self.sea = "谜海" if node == "点击谜之海" else self.target
            self.events.append(("select", self.sea))
        elif node == "点击体力用尽确定":
            self.events.append(("cancel", self.sea))
        else:
            self.fail(f"不应单独调用自定义动作：{node}")
        return SimpleNamespace(success=True)

    def click(self, kind, params):
        self.assertEqual(kind, module.JActionType.Click)
        if tuple(params.target) == (659, 211):
            if self.failed_action == module.CLOSE_NODE:
                return SimpleNamespace(success=False)
            if self.designer_clicks:
                self.events.append(("designer_close", self.sea))
                if not self.stuck_designer:
                    self.designer_clicks -= 1
                    self.pending = self.designer_clicks > 0
                return SimpleNamespace(success=True)
            if self.display:
                self.events.append(("display_close", self.sea))
                if not self.stuck_display:
                    self.display = False
            else:
                self.events.append(("close", self.sea))
                if not self.stuck_close:
                    self.pending = False
                    self.display = self.display_after_result
            return SimpleNamespace(success=True)
        self.assertEqual(tuple(params.target), (132, 1078))
        self.assertFalse(self.pending)
        self.events.append(("draw", self.sea))
        self.pending = True
        self.designer_clicks = 3 if self.designer_after_draw else 0
        return SimpleNamespace(success=self.failed_action != "free")

    def share(self, node):
        self.assertTrue(self.pending)
        self.events.append(("share", self.sea))
        if self.share_closes_result:
            self.pending = False
            self.display = self.display_after_result
        return SimpleNamespace(status=SimpleNamespace(succeeded=self.sharing_succeeds))

    def run_action(self):
        return self.action.run(self.source, SimpleNamespace())

    def draws(self):
        return [sea for kind, sea in self.events if kind == "draw"]

    def test_all_free_combinations_fill_only_remaining_with_mystery(self):
        for qi, fantasy, mystery in product((False, True), repeat=3):
            with self.subTest(qi=qi, fantasy=fantasy, mystery=mystery):
                self.events.clear()
                self.checks.clear()
                self.free = {"绮海": qi, "幻海": fantasy, "谜海": mystery}
                self.assertTrue(self.run_action())
                expected = [sea for sea, available in self.free.items() if available]
                paid = max(0, 2 - len(expected))
                expected += ["谜海"] * paid
                self.assertEqual(self.draws(), expected)
                self.assertEqual([e for e in self.events if e[0] == "share"], [("share", expected[0])])
                self.assertEqual(sum(kind == "close" for kind, _ in self.events), len(expected))
                for sea in ("绮海", "幻海", "谜海"):
                    self.assertEqual(self.checks[module.FREE_NODE, sea], 1)
                self.assertEqual(self.events.count(("select", "谜海")), 1)
                self.assertEqual(sum(self.checks[node, "谜海"] for node in (
                    "点击谜之海-感应第一次", "点击谜之海-感应第二次")), paid)

    def test_switch_combinations_and_reentry_have_no_shared_count(self):
        self.free = {"绮海": True, "幻海": True, "谜海": True}
        for free in (False, True):
            for mystery in (False, True):
                for share in (False, True):
                    with self.subTest(free=free, mystery=mystery, share=share):
                        self.events.clear()
                        self.checks.clear()
                        self.config[module.FREE_NODE]["enabled"] = free
                        self.config["点击谜之海"]["enabled"] = mystery
                        self.config[module.RESULT_NODE]["enabled"] = share
                        before = deepcopy(self.config)
                        self.assertTrue(self.run_action())
                        self.assertEqual(len(self.draws()), 3 if free else 2 if mystery else 0)
                        self.assertEqual(sum(e[0] == "share" for e in self.events), int(share and (free or mystery)))
                        self.assertEqual(self.config, before)
                        if not free:
                            self.assertFalse(any(n == module.FREE_NODE for n, _ in self.checks))

    def test_free_only_includes_mystery_without_paid_supplement(self):
        self.config["点击谜之海"]["enabled"] = False
        for qi, fantasy, mystery in product((False, True), repeat=3):
            with self.subTest(qi=qi, fantasy=fantasy, mystery=mystery):
                self.events.clear()
                self.checks.clear()
                self.free = {"绮海": qi, "幻海": fantasy, "谜海": mystery}
                self.assertTrue(self.run_action())
                expected = [sea for sea, available in self.free.items() if available]
                self.assertEqual(self.draws(), expected)
                self.assertFalse(self.config["点击谜之海"]["enabled"])
                self.assertEqual(sum(self.checks[node, "谜海"] for node in (
                    "点击谜之海-感应第一次", "点击谜之海-感应第二次")), 0)

    def test_selects_qi_before_checking_and_swipes_when_missing(self):
        self.hide_tabs = 1
        self.assertTrue(self.run_action())
        self.assertEqual(self.events[:3], [("swipe", "谜海"), ("select", "绮海"), ("draw", "绮海")])

    def test_missing_tab_or_unready_page_times_out_without_checking_free(self):
        for field, value in (("hide_tabs", 1000), ("loading", True)):
            with self.subTest(field=field):
                self.events.clear()
                setattr(self, field, value)
                self.assertFalse(self.run_action())
                self.assertEqual(self.draws(), [])
                self.assertFalse(any(n == module.FREE_NODE for n, _ in self.checks))
                setattr(self, field, 0 if field == "hide_tabs" else False)

    def test_missing_result_or_stuck_close_never_draws_again(self):
        for field in ("no_result", "stuck_close"):
            with self.subTest(field=field):
                self.events.clear()
                self.pending = False
                setattr(self, field, True)
                self.assertFalse(self.run_action())
                self.assertEqual(self.draws(), ["绮海"])
                self.assertLessEqual(sum(e[0] == "close" for e in self.events), 1)
                setattr(self, field, False)

    def test_failed_actions_or_share_stop_before_next_draw(self):
        for node in ("心之门阁名", "free", module.CLOSE_NODE, "share"):
            with self.subTest(node=node):
                self.events.clear()
                self.pending = False
                self.failed_action = node
                self.sharing_succeeds = node != "share"
                self.assertFalse(self.run_action())
                self.assertLessEqual(len(self.draws()), 1)

    def test_capture_or_recognition_failure_and_bad_box_never_click_free(self):
        self.screenshot_succeeds = False
        self.assertFalse(self.run_action())
        self.screenshot_succeeds = True
        self.failed_recognition = module.FREE_NODE
        self.assertFalse(self.run_action())
        self.failed_recognition = None
        self.invalid_box = True
        self.assertFalse(self.run_action())
        self.assertEqual(self.draws(), [])

    def test_resource_popup_is_cancelled_without_retry(self):
        self.config[module.FREE_NODE]["enabled"] = False
        self.insufficient = True
        self.assertFalse(self.run_action())
        self.assertEqual(self.draws(), ["谜海"])
        self.assertIn(("cancel", "谜海"), self.events)

    def test_share_can_already_close_result_without_extra_click(self):
        self.share_closes_result = True
        self.assertTrue(self.run_action())
        self.assertEqual([e for e in self.events if e[0] == "close"], [("close", "幻海")])

    def test_cancel_before_click_returns_without_draw(self):
        original_clone = self.clone
        def cancelled_clone():
            context = original_clone()
            context.tasker.stopping = True
            return context
        self.source.clone.side_effect = cancelled_clone
        self.assertFalse(self.run_action())
        self.assertEqual(self.events, [])

    def test_display_after_share_or_result_close_reuses_close_without_extra_draw(self):
        for free in (False, True):
            for share in (False, True):
                with self.subTest(free=free, share=share):
                    self.events.clear()
                    self.config[module.FREE_NODE]["enabled"] = free
                    self.config[module.RESULT_NODE]["enabled"] = share
                    self.share_closes_result = share
                    self.display_after_result = True
                    self.assertTrue(self.run_action())
                    expected = ["绮海", "幻海"] if free else ["谜海", "谜海"]
                    self.assertEqual(self.draws(), expected)
                    self.assertEqual(sum(e[0] == "display_close" for e in self.events), 2)
                    self.assertEqual(sum(e[0] == "close" for e in self.events), 2 - int(share))
                    preceding = "share" if share else "close"
                    self.assertLess(self.events.index((preceding, expected[0])),
                                    self.events.index(("display_close", expected[0])))

    def test_display_is_not_checked_before_result(self):
        context = self.clone()
        with patch.object(self.action, "_close_display") as close:
            self.action._wait_page(context)
            self.pending = self.no_result = True
            with self.assertRaises(RuntimeError):
                self.action._wait_hit(context, module.RESULT_NODE)
            close.assert_not_called()

    def test_stuck_display_keeps_original_deadline_and_never_draws_again(self):
        self.display_after_result = self.stuck_display = True
        self.assertFalse(self.run_action())
        self.assertEqual(self.draws(), ["绮海"])
        self.assertLess(self.now, 32)
        self.assertEqual(sum(e[0] == "close" for e in self.events), 1)

    def test_failed_display_close_stops_before_any_later_draw(self):
        self.display_after_result = self.share_closes_result = True
        self.failed_action = module.CLOSE_NODE
        self.assertFalse(self.run_action())
        self.assertEqual(self.draws(), ["绮海"])

    def test_designer_clicks_continue_after_text_disappears_and_stop_when_ready(self):
        for share in (False, True):
            with self.subTest(share=share):
                self.events.clear()
                self.config[module.RESULT_NODE]["enabled"] = share
                self.designer_after_draw = True
                self.assertTrue(self.run_action())
                self.assertEqual(self.draws(), ["绮海", "幻海"])
                for sea in ("绮海", "幻海"):
                    self.assertEqual(self.events.count(("designer_close", sea)), 3)
                if share:
                    self.assertLess(self.events.index(("share", "绮海")),
                                    self.events.index(("designer_close", "绮海")))
                self.assertEqual(self.designer_clicks, 0)
                self.assertFalse(any(event[0] == "close" for event in self.events))

    def test_designer_is_not_checked_in_mystery_free_or_paid_draws(self):
        for free in (False, True):
            with self.subTest(free=free):
                self.events.clear()
                self.checks.clear()
                self.config[module.FREE_NODE]["enabled"] = free
                self.free = {"绮海": False, "幻海": False, "谜海": True}
                self.assertTrue(self.run_action())
                self.assertEqual(self.draws(), ["谜海", "谜海"])
                self.assertEqual(self.checks[module.DESIGNER_NODE, "谜海"], 0)
                self.assertEqual(self.events.count(("close", "谜海")), 2)

    def test_designer_timeout_keeps_deadline_and_never_draws_again(self):
        self.designer_after_draw = self.stuck_designer = True
        self.assertFalse(self.run_action())
        self.assertEqual(self.draws(), ["绮海"])
        self.assertLess(self.now, 32)
        self.assertGreater(self.events.count(("designer_close", "绮海")), 1)

    def test_designer_click_failure_stops_before_any_later_draw(self):
        self.designer_after_draw = True
        self.failed_action = module.CLOSE_NODE
        self.assertFalse(self.run_action())
        self.assertEqual(self.draws(), ["绮海"])

    def test_cancel_during_designer_clicks_stops_before_any_later_draw(self):
        self.designer_after_draw = True
        original_click = self.click

        def cancel_after_click(kind, params):
            result = original_click(kind, params)
            if self.events[-1][0] == "designer_close":
                self.context.tasker.stopping = True
            return result

        self.click = cancel_after_click
        self.assertFalse(self.run_action())
        self.assertEqual(self.draws(), ["绮海"])
        self.assertEqual(self.events.count(("designer_close", "绮海")), 1)


class HeartsDoorOptionsTests(unittest.TestCase):
    def test_default_options_and_confirmation_are_independent(self):
        options = INTERFACE["option"]
        self.assertEqual(options["免费感应"]["default_case"], "No")
        self.assertEqual(options["谜海感应"]["default_case"], "Yes")
        self.assertEqual(options["免费感应"]["__mfa_confirm_on_enable"], "可能会误用资源，谨慎开启")
        for option, node in (("免费感应", module.FREE_NODE), ("谜海感应", "点击谜之海"),
                             ("是否在心之门这里分享", module.RESULT_NODE)):
            for case in options[option]["cases"]:
                self.assertEqual(case["pipeline_override"], {node: {"enabled": case["name"] == "Yes"}})

    def test_free_roi_and_main_route(self):
        self.assertEqual(PIPELINE[module.FREE_NODE]["recognition"]["param"]["roi"], [0, 1060, 336, 128])
        self.assertEqual(PIPELINE["心之门任务链准备"]["next"], ["心之门感应调度"])
        task = next(task for task in INTERFACE["task"] if task["name"] == "心之门")
        self.assertEqual(task["option"], ["免费感应", "谜海感应", "是否在心之门这里分享"])
        self.assertIn("from . import hearts_door", (ROOT / "agent/custom/__init__.py").read_text(encoding="utf-8"))

    def test_display_supports_all_three_qualities(self):
        param = PIPELINE[module.DISPLAY_NODE]["recognition"]["param"]
        self.assertEqual(param["roi"], [270, 170, 180, 90])
        self.assertEqual(param["expected"], ["非凡", "稀有", "闪耀"])


def native_builtin_clicks():
    from maa.custom_action import CustomAction
    from maa.custom_recognition import CustomRecognition
    from maa.resource import Resource
    from maa.tasker import Tasker, LoggingLevelEnum
    from tools.test_make_target_clothes_readiness import ScreenController

    Tasker.set_stdout_level(LoggingLevelEnum.Off)

    class Screen(ScreenController):
        stage = "page"
        draw_point = (198, 1096)

        def click(self, x, y):
            self.clicks.append((x, y))
            if self.stage == "page":
                assert (x, y) == self.draw_point
                self.stage = "result"
            else:
                assert (x, y) == (659, 211)
                assert self.stage in ("result", "display")
                self.stage = "display" if self.stage == "result" else "page"
            return True

    screen = Screen()

    class Recognition(CustomRecognition):
        def analyze(self, context, argv):
            expected = {module.RESULT_NODE: "result", module.DISPLAY_NODE: "display",
                        module.READY_NODE: "page"}
            if screen.stage == expected[argv.node_name]:
                return self.AnalyzeResult(box=(633, 1188, 62, 37))
            return None

    class Action(CustomAction):
        def run(self, context, argv):
            helper = module.HeartsDoorAction()
            # 使用真实 Context / Custom 回调检验两处免费点击和谜海点击，
            # 防止 mock 掩盖 reco_id=0 导致自定义关窗动作未执行的问题。
            for box in (Rect(90, 1077, 217, 38), Rect(90, 1077, 217, 38),
                        Rect(35, 1097, 21, 16)):
                screen.draw_point = (box.x + box.w // 2, box.y + box.h // 2)
                helper._click_center(context, box)
                helper._finish_draw(context, share=False)
                assert screen.stage == "page"
            return True

    resource = Resource()
    assert resource.post_bundle(ROOT / "assets/resource").wait().succeeded
    assert resource.register_custom_recognition("heart_stage", Recognition())
    assert resource.register_custom_action("heart_click_check", Action())
    overrides = {node: {"recognition": {"type": "Custom", "param": {
        "custom_recognition": "heart_stage"}}}
        for node in (module.RESULT_NODE, module.DISPLAY_NODE, module.READY_NODE)}
    overrides["心之门原生点击检查"] = {"action": {"type": "Custom", "param": {
        "custom_action": "heart_click_check"}}}
    assert resource.override_pipeline(overrides)
    assert screen.post_connection().wait().succeeded
    tasker = Tasker()
    assert tasker.bind(resource, screen)
    assert tasker.post_task("心之门原生点击检查").wait().status.succeeded
    assert screen.clicks == [(198, 1096), (659, 211), (659, 211)] * 2 + [
        (45, 1105), (659, 211), (659, 211)]
    print("Native builtin clicks passed: Qi, Fantasy, Mystery and both popup stages")


class HeartsDoorNativeTests(unittest.TestCase):
    def test_builtin_clicks_in_real_context(self):
        result = subprocess.run(
            [sys.executable, "-c", "from maa.library import Library; Library.version(); "
             "from tools.test_hearts_door import native_builtin_clicks; native_builtin_clicks()"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
