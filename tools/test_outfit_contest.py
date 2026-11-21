"""评选赛时段、默认开关、完整投稿及限时退出的离线验证。"""

import importlib.util
import json
import subprocess
import sys
import textwrap
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from maa.agent.agent_server import AgentServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location("outfit_contest_under_test", ROOT / "agent/custom/outfit_contest.py")
module = importlib.util.module_from_spec(spec)
with patch.object(AgentServer, "register_custom_action", return_value=True):
    spec.loader.exec_module(module)
PIPELINE = json.loads((ROOT / "assets/resource/pipeline/outfit_contest.json").read_text(encoding="utf-8"))
INTERFACE = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
PREFIX = "搭配评选赛"


class PeriodTests(unittest.TestCase):
    def test_boundaries_in_beijing_and_utc(self):
        for date, period in [
            ("2026-10-30T20:59:59", "judging"),
            ("2026-10-30T21:00:00", "settlement"),
            ("2026-10-31T04:59:59", "settlement"),
            ("2026-10-31T05:00:00", "submission"),
            ("2026-11-01T20:00:00", "submission"),
            ("2026-11-02T04:59:59", "submission"),
            ("2026-11-02T05:00:00", "judging"),
        ]:
            now = datetime.fromisoformat(date).replace(tzinfo=module.BEIJING)
            with self.subTest(date=date):
                self.assertEqual(module.contest_period(now), period)
                self.assertEqual(module.contest_period(now.astimezone(timezone.utc)), period)

    def test_guard_skips_before_navigation_and_restores_next_on_reentry(self):
        context = Mock()
        context.tasker.stopping = False
        context.override_next.return_value = True
        context.override_pipeline.return_value = True
        argv = SimpleNamespace(node_name=PREFIX + "任务链入口")
        prepare = PREFIX + "任务链准备"
        action = module.OutfitContestGuard()
        for period, enabled, expected in [
            ("settlement", False, []), ("settlement", True, []),
            ("submission", False, []), ("submission", True, [PREFIX + "自动参赛"]),
            ("judging", False, ["点击我当评委"]), ("judging", True, ["点击我当评委"]),
        ]:
            context.reset_mock()
            context.get_node_data.side_effect = lambda name: (
                {"attach": {"auto_submit": enabled}} if name == argv.node_name else deepcopy(PIPELINE[name])
            )
            with patch.object(module, "contest_period", return_value=period):
                self.assertTrue(action.run(context, argv))
            if not expected:
                context.override_pipeline.assert_not_called()
                context.override_next.assert_called_once_with(argv.node_name, [])
            else:
                self.assertEqual(context.override_next.call_args_list[0].args,
                                 (prepare, expected))
                context.override_next.assert_called_with(argv.node_name, [prepare])
                override = context.override_pipeline.call_args.args[0][prepare]
                self.assertEqual(override["action"]["param"]["custom_action_param"], {
                    "target": "搭配评选赛",
                    "wait_timeout": 30,
                    "ignore_background_download": period == "judging",
                })
            context.run_action.assert_not_called()

    def test_override_failure_stops_and_default_option_is_off(self):
        context = Mock()
        context.tasker.stopping = False
        context.get_node_data.side_effect = lambda name: deepcopy(PIPELINE[name])
        context.override_next.return_value = False
        with patch.object(module, "contest_period", return_value="judging"):
            self.assertFalse(module.OutfitContestGuard().run(context, SimpleNamespace(node_name=PREFIX + "任务链入口")))
        option = INTERFACE["option"][PREFIX + "自动参赛"]
        self.assertEqual(option["default_case"], "No")
        self.assertFalse(PIPELINE[PREFIX + "任务链入口"]["attach"]["auto_submit"])


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.context = Mock()
        self.context.tasker.stopping = False
        self.controller = self.context.tasker.controller
        self.controller.post_screencap.return_value.wait.return_value.succeeded = True
        self.controller.cached_image = object()
        self.context.get_node_data.side_effect = lambda name: PIPELINE[name]
        self.context.run_recognition.side_effect = self.recognize
        self.context.run_action.side_effect = self.operate
        self.context.run_task.side_effect = self.run_leaf
        self.action = module.OutfitContestSubmit()
        self.now = 0.
        self.state = "entry"
        self.checked = {"保存模板已勾选": True, "上传定格已勾选": True}
        self.events = []
        self.blocked = self.failed_action = self.failed_recognition = None
        self.loading_until = 0.
        self.action_delay = 0.
        self.recognition_delay = 0.
        self.keep_checked = False
        for patcher in (
            patch.object(module.time, "monotonic", side_effect=lambda: self.now),
            patch.object(module.time, "sleep", side_effect=self.advance),
            patch.object(module, "logger"),
            patch.object(module, "debug_exception"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def advance(self, seconds):
        self.now += seconds

    def recognize(self, node, image):
        if node == self.failed_recognition:
            return None
        name = node.removeprefix(PREFIX)
        states = {
            "entry": ["点击我要参赛"], "dressing": ["换好了"], "category": ["点击套装"],
            "list": ["选择首套"], "dressed": ["换好了"], "photo": ["投稿参赛"],
            "crop": ["拍摄确认上传"], "cover": ["上传封面", "封面确认上传"],
            "prompt": ["不公开提示确定"], "shared": ["分享页面"], "submitted": ["已参赛"],
        }
        hit = name in states.get(self.state, [])
        if self.state == "cover" and name in self.checked:
            hit = self.checked[name]
        if node == "在加载页面":
            hit = self.now < self.loading_until
        elif self.blocked == self.state:
            hit = False
        self.advance(self.recognition_delay)
        return SimpleNamespace(hit=hit, box=(10, 20, 50, 40))

    def operate(self, node, box, pipeline_override=None):
        name = node.removeprefix(PREFIX)
        self.events.append(name)
        self.advance(self.action_delay)
        if self.failed_action == name:
            return SimpleNamespace(success=False)
        if name in self.checked:
            if not self.keep_checked:
                self.checked[name] = False
        else:
            next_states = {
                "点击我要参赛": "dressing", "上划分类": "category", "点击套装": "list",
                "选择首套": "dressed", "换好了": "photo", "投稿参赛": "crop",
                "拍摄确认上传": "cover", "封面确认上传": "prompt",
                "不公开提示确定": "shared", "投稿后返回": "submitted",
            }
            if name == "封面确认上传":
                self.assertFalse(any(self.checked.values()))
            self.state = next_states[name]
        if PIPELINE[node]["action"]["type"] == "Custom":
            params = pipeline_override[node]["action"]["param"]["custom_action_param"]
            self.assertEqual(params["max_retries"], 1)
            self.assertTrue(params["fail_on_timeout"])
            self.assertLessEqual(params["timeout"], 30000)
        return SimpleNamespace(success=True)

    def run_leaf(self, node, pipeline_override):
        self.assertFalse(PIPELINE[node].get("next"))
        result = self.operate(node, (10, 20, 50, 40), pipeline_override)
        return SimpleNamespace(status=SimpleNamespace(succeeded=result.success))

    def run_submission(self):
        return self.action.run(self.context, SimpleNamespace())

    def test_full_submission_unchecks_then_confirms_upload_and_returns(self):
        self.assertTrue(self.run_submission())
        self.assertEqual(self.events, [
            "点击我要参赛", "上划分类", "点击套装", "选择首套", "换好了", "投稿参赛",
            "拍摄确认上传", "保存模板已勾选", "上传定格已勾选", "封面确认上传",
            "不公开提示确定", "投稿后返回",
        ])
        self.assertEqual(self.state, "submitted")
        self.assertEqual(self.context.run_task.call_count, 4)

    def test_already_submitted_skips_without_clicking(self):
        self.state = "submitted"
        self.assertTrue(self.run_submission())
        self.assertEqual(self.events, [])

    def test_absence_of_entry_button_is_not_treated_as_submitted(self):
        self.state = "unknown"
        self.assertFalse(self.run_submission())
        self.assertAlmostEqual(self.now, 30)
        self.assertEqual(self.events, [])

    def test_unchecked_options_are_not_clicked_back_on(self):
        for checked in (False, True):
            self.state = "entry"
            self.events.clear()
            self.checked = {"保存模板已勾选": False, "上传定格已勾选": checked}
            self.assertTrue(self.run_submission())
            self.assertNotIn("保存模板已勾选", self.events)
            self.assertEqual("上传定格已勾选" in self.events, checked)

    def test_persistent_checkmark_stops_before_upload(self):
        self.keep_checked = True
        self.assertFalse(self.run_submission())
        self.assertAlmostEqual(self.now, 30)
        self.assertNotIn("封面确认上传", self.events)

    def test_loading_does_not_reset_deadline_or_allow_blind_click(self):
        self.loading_until = 60
        self.assertFalse(self.run_submission())
        self.assertAlmostEqual(self.now, 30)
        self.assertEqual(self.events, [])

    def test_loading_can_finish_within_deadline(self):
        self.loading_until = 3
        self.assertTrue(self.run_submission())
        self.assertGreaterEqual(self.now, 3)

    def test_each_transition_timeout_prevents_later_actions(self):
        for blocked in ("dressing", "category", "list", "dressed", "photo", "crop", "cover", "prompt", "shared", "submitted"):
            with self.subTest(blocked=blocked):
                self.now = 0.
                self.state = "entry"
                self.events.clear()
                self.checked = {"保存模板已勾选": True, "上传定格已勾选": True}
                self.blocked = blocked
                self.assertFalse(self.run_submission())
                self.assertAlmostEqual(self.now, 30)
                self.assertEqual(len(self.events), len(set(self.events)))

    def test_action_and_recognition_time_are_included(self):
        self.blocked = "dressing"
        self.action_delay = 12
        self.assertFalse(self.run_submission())
        self.assertAlmostEqual(self.now, 30)
        self.events.clear()
        self.now = 0.
        self.state = "entry"
        self.action_delay = 0
        self.recognition_delay = 15
        self.assertFalse(self.run_submission())
        self.assertEqual(self.events, [])

    def test_failure_of_capture_recognition_or_action_stops(self):
        self.controller.post_screencap.return_value.wait.return_value.succeeded = False
        self.assertFalse(self.run_submission())
        self.context.run_recognition.assert_not_called()
        self.controller.post_screencap.return_value.wait.return_value.succeeded = True
        self.failed_recognition = PREFIX + "已参赛"
        self.assertFalse(self.run_submission())
        self.assertEqual(self.events, [])
        self.failed_recognition = None
        self.failed_action = "点击套装"
        self.assertFalse(self.run_submission())
        self.assertNotIn("选择首套", self.events)

    def test_cancel_never_clicks_or_reports_completion(self):
        self.context.tasker.stopping = True
        self.assertFalse(self.run_submission())
        self.assertEqual(self.events, [])
        module.logger.error.assert_not_called()

    def test_native_submission_calls_custom_clicks_with_recognition_records(self):
        script = textwrap.dedent('''
            import copy
            import importlib.util
            import numpy as np
            from maa.library import Library
            Library.version()
            from maa.tasker import Tasker
            from maa.define import LoggingLevelEnum
            Tasker.set_stdout_level(LoggingLevelEnum.Off)
            from maa.resource import Resource
            from maa.controller import CustomController
            from maa.custom_recognition import CustomRecognition
            from maa.agent.agent_server import AgentServer
            from unittest.mock import patch
            from tools.test_outfit_contest import module, PIPELINE, PREFIX

            spec = importlib.util.spec_from_file_location('click_under_test', 'agent/custom/click_and_wait_change.py')
            clicks = importlib.util.module_from_spec(spec)
            with patch.object(AgentServer, 'register_custom_action', return_value=True):
                spec.loader.exec_module(clicks)

            sequence = ['点击我要参赛', '上划分类', '点击套装', '选择首套', '换好了', '投稿参赛',
                        '拍摄确认上传', '保存模板已勾选', '上传定格已勾选', '封面确认上传',
                        '不公开提示确定', '投稿后返回']
            ready = ['点击我要参赛', '换好了', '点击套装', '选择首套', '换好了', '投稿参赛',
                     '拍摄确认上传', '上传封面', '上传封面', '封面确认上传',
                     '不公开提示确定', '分享页面', '已参赛']
            checked = {'保存模板已勾选': True, '上传定格已勾选': True}
            boxes = {name: (20 + index * 20, 400, 10, 10) for index, name in enumerate(sequence)}
            class Screen(CustomController):
                index = 0
                def connect(self): return True
                def request_uuid(self): return 'outfit-contest-offline'
                def get_features(self): return 0
                def screencap(self): return np.full((1280, 720, 3), (self.index % 2) * 255, dtype=np.uint8)
                def click(self, x, y):
                    name = sequence[self.index]
                    if name in checked:
                        checked[name] = False
                    self.index += 1
                    return True
                def swipe(self, *args):
                    self.index += 1
                    return True
                def unused(self, *args): return False
                start_app = stop_app = touch_down = touch_move = touch_up = unused
                click_key = input_text = key_down = key_up = unused

            class Recognition(CustomRecognition):
                def analyze(self, context, argv):
                    name = argv.node_name.removeprefix(PREFIX)
                    hit = name == ready[screen.index]
                    if name == '上传封面':
                        hit = 7 <= screen.index <= 9
                    if name in checked:
                        hit = 7 <= screen.index <= 9 and checked[name]
                    if hit:
                        return boxes.get(name, (10, 20, 30, 40))
                    return None

            screen = Screen()
            assert screen.post_connection().wait().succeeded
            resource = Resource()
            assert resource.register_custom_recognition('state', Recognition())
            assert resource.register_custom_action('outfit_contest_submit', module.OutfitContestSubmit())
            assert resource.register_custom_action('click_and_wait_change', clicks.ClickAndWaitChangeAction())
            nodes = copy.deepcopy(PIPELINE)
            for name, data in nodes.items():
                if name != PREFIX + '自动参赛':
                    data['recognition'] = {'type': 'Custom', 'param': {'custom_recognition': 'state'}}
            nodes['在加载页面'] = {'recognition': {'type': 'Custom', 'param': {'custom_recognition': 'state'}}}
            assert resource.override_pipeline(nodes)
            tasker = Tasker()
            assert tasker.bind(resource, screen)
            assert tasker.post_task(PREFIX + '自动参赛').wait().succeeded
            assert screen.index == len(sequence), screen.index
            assert not any(checked.values())
            # 重新进入，必须从当前画面确认已参赛，不能再次点击或依赖账号缓存。
            assert tasker.post_task(PREFIX + '自动参赛').wait().succeeded
            assert screen.index == len(sequence)
            print('native outfit submission and reentry passed')
        ''')
        result = subprocess.run([sys.executable, "-X", "utf8", "-"], input=script,
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("native outfit submission and reentry passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
