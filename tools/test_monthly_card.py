"""月卡特惠弹窗关闭、累计超时和领取交接的离线测试。"""

import importlib.util
import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from maa.agent.agent_server import AgentServer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location(
    "monthly_card_under_test", ROOT / "agent/custom/monthly_card.py"
)
monthly_card = importlib.util.module_from_spec(spec)
with patch.object(AgentServer, "register_custom_action", return_value=True):
    spec.loader.exec_module(monthly_card)
PIPELINE = json.loads(
    (ROOT / "assets/resource/pipeline/monthly_card.json").read_text(encoding="utf-8")
)


class MonthlyCardPopupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # AgentServer 模式不能创建 Resource，单独进程读取原生框架解析的默认配置。
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c",
             "import json; from pathlib import Path; from maa.resource import Resource; "
             "r=Resource(); "
             "assert r.override_pipeline(json.loads(Path("
             "'assets/resource/pipeline/monthly_card.json').read_text(encoding='utf-8'))); "
             "print(json.dumps(r.get_node_data('月卡弹窗检查')))"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
        )
        cls.effective_node = json.loads(result.stdout.strip().splitlines()[-1])

    def setUp(self):
        self.context = Mock()
        self.context.tasker.stopping = False
        self.controller = self.context.tasker.controller
        self.controller.post_screencap.return_value.wait.return_value.succeeded = True
        self.controller.post_click.return_value.wait.return_value.succeeded = True
        self.context.get_node_data.return_value = dict(self.effective_node)
        self.now = 0.0
        self.popup_visible = lambda: False
        self.tab_visible = lambda: True
        self.box = SimpleNamespace(x=470, y=140, w=60, h=30)
        self.context.run_recognition.side_effect = self.recognize
        self.click_times = []
        self.controller.post_click.side_effect = self.click
        for patcher in (
            patch.object(monthly_card.time, "monotonic", side_effect=lambda: self.now),
            patch.object(monthly_card.time, "sleep", side_effect=self.advance),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def advance(self, duration):
        self.now += duration

    def recognize(self, name, image):
        if name == "月卡特惠弹窗":
            return SimpleNamespace(hit=self.popup_visible())
        self.assertEqual(name, "商城点击月卡")
        return SimpleNamespace(hit=self.tab_visible(), box=self.box)

    def click(self, x, y):
        self.click_times.append(self.now)
        return Mock(wait=Mock(return_value=SimpleNamespace(succeeded=True)))

    def run_action(self):
        return monthly_card.MonthlyCardPopupAction().run(
            self.context, SimpleNamespace(node_name="月卡弹窗检查")
        )

    def test_no_popup_continues_without_clicking(self):
        self.assertTrue(self.run_action())
        self.controller.post_click.assert_not_called()

    def test_popup_closed_before_continuing(self):
        self.popup_visible = lambda: not self.click_times
        self.assertTrue(self.run_action())
        self.controller.post_click.assert_called_once_with(500, 155)
        self.assertGreater(self.now, self.click_times[-1])

    def test_retries_use_fresh_tab_center(self):
        self.popup_visible = lambda: len(self.click_times) < 2
        original_click = self.click

        def moved_tab(x, y):
            result = original_click(x, y)
            self.box = SimpleNamespace(x=480, y=150, w=40, h=20)
            return result

        self.controller.post_click.side_effect = moved_tab
        self.assertTrue(self.run_action())
        self.assertEqual(
            [call.args for call in self.controller.post_click.call_args_list],
            [(500, 155), (500, 160)],
        )

    def test_persistent_popup_uses_one_deadline(self):
        self.popup_visible = lambda: True
        self.assertFalse(self.run_action())
        node = self.context.get_node_data.return_value
        self.assertAlmostEqual(self.now, node["timeout"] / 1000)
        self.assertTrue(all(
            b - a >= node["rate_limit"] / 1000
            for a, b in zip(self.click_times, self.click_times[1:])
        ))
        self.context.run_task.assert_not_called()

    def test_effective_node_timing_is_honored(self):
        self.context.get_node_data.return_value = {"timeout": 2500, "rate_limit": 700}
        self.popup_visible = lambda: True
        self.assertFalse(self.run_action())
        self.assertAlmostEqual(self.now, 2.5)
        for actual, expected in zip(self.click_times, (0, .7, 1.4, 2.1)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(len(self.click_times), 4)

    def test_single_ocr_miss_does_not_allow_claiming(self):
        self.popup_visible = lambda: self.now != 1
        self.assertFalse(self.run_action())

    def test_missing_tab_never_clicks_or_continues(self):
        for visible in (False, True):
            with self.subTest(popup=visible):
                self.now = 0
                self.popup_visible = lambda: visible
                self.tab_visible = lambda: False
                self.assertFalse(self.run_action())
        self.controller.post_click.assert_not_called()

    def test_page_not_ready_waits_until_tab_is_visible(self):
        self.tab_visible = lambda: self.now >= 2
        self.assertTrue(self.run_action())
        self.assertGreater(self.now, 2)
        self.controller.post_click.assert_not_called()

    def test_invalid_tab_box_never_clicks(self):
        self.popup_visible = lambda: True
        self.box.w = 0
        self.assertFalse(self.run_action())
        self.controller.post_click.assert_not_called()

    def test_capture_recognition_click_and_cancel_failures_stop(self):
        self.popup_visible = lambda: True
        self.controller.post_screencap.return_value.wait.return_value.succeeded = False
        self.assertFalse(self.run_action())
        self.controller.post_click.assert_not_called()
        self.controller.post_screencap.return_value.wait.return_value.succeeded = True
        with patch.object(self.context, "run_recognition", return_value=None):
            self.assertFalse(self.run_action())
        self.controller.post_click.assert_not_called()
        self.controller.post_click.side_effect = None
        self.controller.post_click.return_value.wait.return_value.succeeded = False
        self.assertFalse(self.run_action())
        self.controller.post_click.reset_mock()
        self.context.tasker.stopping = True
        self.assertFalse(self.run_action())
        self.controller.post_click.assert_not_called()

    def test_slow_recognition_cannot_click_after_deadline(self):
        self.popup_visible = lambda: True
        original = self.recognize

        def slow_recognition(name, image):
            self.advance(self.context.get_node_data.return_value["timeout"] / 1000)
            return original(name, image)

        self.context.run_recognition.side_effect = slow_recognition
        self.assertFalse(self.run_action())
        self.controller.post_click.assert_not_called()

    def test_unbounded_or_missing_timing_fails(self):
        for node in (None, {}, {"timeout": -1, "rate_limit": 1000},
                     {"timeout": 20000, "rate_limit": 0}):
            with self.subTest(node=node):
                self.context.get_node_data.return_value = node
                self.assertFalse(self.run_action())
        self.controller.post_click.assert_not_called()

    def test_pipeline_checks_before_claiming_without_redeclaring_defaults(self):
        self.assertEqual(PIPELINE["商城点击月卡"]["next"], ["月卡弹窗检查"])
        self.assertEqual(PIPELINE["月卡弹窗检查"]["next"], ["点击月卡"])
        node = PIPELINE["月卡弹窗检查"]
        self.assertNotIn("rate_limit", node)
        self.assertNotIn("timeout", node)
        self.assertNotIn("custom_action_param", node["action"]["param"])
        self.assertNotIn("on_error", node)

    def test_native_task_failure_does_not_run_claim(self):
        script = textwrap.dedent('''
            from maa.library import Library
            Library.version()
            import copy
            import numpy as np
            from maa.resource import Resource
            from maa.tasker import Tasker
            from maa.define import LoggingLevelEnum
            Tasker.set_stdout_level(LoggingLevelEnum.Off)
            resource = Resource()
            from maa.controller import CustomController
            from maa.custom_action import CustomAction
            from maa.custom_recognition import CustomRecognition
            from tools.test_monthly_card import monthly_card, PIPELINE

            class Screen(CustomController):
                def connect(self): return True
                def request_uuid(self): return 'monthly-offline'
                def get_features(self): return 0
                def screencap(self): return np.zeros((1280, 720, 3), dtype=np.uint8)
                def click(self, x, y):
                    assert (x, y) == (500, 150)
                    self.clicks += 1
                    if self.mode == 'close': self.visible = False
                    return True
                def unused(self, *args): return False
                start_app = stop_app = swipe = touch_down = touch_move = touch_up = unused
                click_key = input_text = key_down = key_up = unused

            controller = Screen()
            assert controller.post_connection().wait().succeeded
            tasker = Tasker()
            assert tasker.bind(resource, controller)

            class Recognition(CustomRecognition):
                def analyze(self, context, argv):
                    if argv.node_name == '商城点击月卡' or controller.visible:
                        return (480, 140, 40, 20)
                    return None

            claims = []
            class Claim(CustomAction):
                def run(self, context, argv):
                    claims.append(argv.node_name)
                    return True

            assert resource.register_custom_recognition('screen', Recognition())
            assert resource.register_custom_action('monthly_card_popup', monthly_card.MonthlyCardPopupAction())
            assert resource.register_custom_action('claim', Claim())
            nodes = copy.deepcopy(PIPELINE)
            for node in nodes.values(): node['post_delay'] = 0
            for name in ('月卡特惠弹窗', '商城点击月卡'):
                nodes[name]['recognition'] = {'type': 'Custom', 'param': {'custom_recognition': 'screen'}}
            nodes['月卡弹窗检查'].update(rate_limit=10, timeout=150)
            nodes['点击月卡'].update(action={'type': 'Custom', 'param': {'custom_action': 'claim'}}, next=[])
            assert resource.override_pipeline(nodes)
            for mode in ('absent', 'close', 'stuck'):
                controller.mode = mode
                controller.visible = mode != 'absent'
                controller.clicks = 0
                claims.clear()
                result = tasker.post_task('月卡弹窗检查').wait()
                assert result.succeeded == (mode != 'stuck'), mode
                assert bool(claims) == (mode != 'stuck'), (mode, claims)
                if mode == 'absent': assert controller.clicks == 0
                if mode == 'close': assert controller.clicks == 1
            print('native monthly popup cases passed')
        ''')
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-"], input=script,
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("native monthly popup cases passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
