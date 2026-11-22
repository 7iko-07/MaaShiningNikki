"""签到页面连续两帧就绪、累计超时及奖励流程交接。"""

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
    "benefits_under_test", ROOT / "agent/custom/benefits.py"
)
benefits = importlib.util.module_from_spec(spec)
with patch.object(AgentServer, "register_custom_action", return_value=True):
    spec.loader.exec_module(benefits)
PIPELINE = json.loads(
    (ROOT / "assets/resource/pipeline/benefits.json").read_text(encoding="utf-8")
)


class BenefitsWaitReadyTests(unittest.TestCase):
    def setUp(self):
        self.context = Mock()
        self.context.tasker.stopping = False
        self.context.get_node_data.return_value = PIPELINE["等待签到页面就绪"]
        self.controller = self.context.tasker.controller
        self.controller.post_screencap.return_value.wait.return_value.succeeded = True
        self.controller.cached_image = object()
        self.now = 0.0
        self.frames = []
        self.ready = lambda: True
        self.context.run_recognition.side_effect = self.recognize
        for patcher in (
            patch.object(benefits.time, "monotonic", side_effect=lambda: self.now),
            patch.object(benefits.time, "sleep", side_effect=self.advance),
            patch.object(benefits, "logger"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def advance(self, duration):
        self.now += duration

    def recognize(self, name, image):
        self.assertEqual(name, "签到页面就绪")
        self.frames.append(image)
        return SimpleNamespace(hit=self.ready())

    def run_action(self):
        return benefits.BenefitsWaitReadyAction().run(
            self.context, SimpleNamespace(node_name="等待签到页面就绪")
        )

    def test_two_fresh_frames_before_continuing(self):
        images = [object(), object()]

        def capture():
            self.controller.cached_image = images[len(self.frames)]
            return Mock(wait=Mock(return_value=SimpleNamespace(succeeded=True)))

        self.controller.post_screencap.side_effect = capture
        self.assertTrue(self.run_action())
        self.assertEqual(self.frames, images)
        self.assertAlmostEqual(self.now, .3)
        self.controller.post_click.assert_not_called()

    def test_miss_resets_confirmation_count(self):
        frames = iter([True, False, True, True])
        self.ready = lambda: next(frames)
        self.assertTrue(self.run_action())
        self.assertEqual(len(self.frames), 4)
        self.assertAlmostEqual(self.now, .9)

    def test_delayed_content_waits_before_continuing(self):
        self.ready = lambda: self.now >= 2
        self.assertTrue(self.run_action())
        self.assertGreaterEqual(self.now, 2.4)

    def test_timeout_includes_recognition_and_never_resets_on_hits(self):
        def intermittent():
            self.advance(.12)
            return len(self.frames) % 2 == 1

        self.ready = intermittent
        self.assertFalse(self.run_action())
        self.assertAlmostEqual(self.now, 30)
        benefits.logger.error.assert_called_with("每日签到：等待签到页面超时")
        self.context.run_task.assert_not_called()

    def test_late_second_hit_does_not_continue(self):
        def slow_hit():
            self.advance(15)
            return True

        self.ready = slow_hit
        self.assertFalse(self.run_action())
        self.assertEqual(len(self.frames), 2)

    def test_failed_capture_never_uses_cached_frame(self):
        self.controller.post_screencap.return_value.wait.return_value.succeeded = False
        self.assertFalse(self.run_action())
        self.context.run_recognition.assert_not_called()

    def test_missing_image_fails(self):
        self.controller.cached_image = None
        self.assertFalse(self.run_action())
        self.context.run_recognition.assert_not_called()

    def test_recognition_failure_stops(self):
        self.context.run_recognition.side_effect = None
        self.context.run_recognition.return_value = None
        self.assertFalse(self.run_action())
        benefits.logger.error.assert_called_with("每日签到：页面识别执行失败")

    def test_cancel_before_or_during_recognition_stops(self):
        self.context.tasker.stopping = True
        self.assertFalse(self.run_action())
        self.controller.post_screencap.assert_not_called()
        self.context.tasker.stopping = False

        def cancel():
            self.context.tasker.stopping = True
            return True

        self.ready = cancel
        self.assertFalse(self.run_action())
        benefits.logger.error.assert_not_called()

    def test_reentry_requires_two_new_hits(self):
        self.assertTrue(self.run_action())
        self.assertTrue(self.run_action())
        self.assertEqual(len(self.frames), 4)

    def test_pipeline_gate_keeps_reward_branches_after_waiting(self):
        self.assertEqual(PIPELINE["点击每日签到"]["next"], ["等待签到页面就绪"])
        self.assertEqual(PIPELINE["等待签到页面就绪"]["next"], [
            "点击感叹号所在位置", "退出签到",
        ])
        self.assertEqual(PIPELINE["点击感叹号所在位置"]["next"], ["退出签到"])
        exit_node = PIPELINE["退出签到"]
        self.assertEqual(exit_node["next"][0], "点击活跃度感叹号所在位置")
        action = exit_node["action"]["param"]
        self.assertEqual(action["custom_action"], "click_and_wait_change")
        params = action["custom_action_param"]
        self.assertEqual(params["popup_node"], "签到奖励弹窗")
        self.assertEqual(params["ready_nodes"], ["签到页面就绪"])
        self.assertEqual(PIPELINE["签到页面就绪"]["recognition"]["param"], {
            "roi": [0, 590, 550, 690], "expected": ["月卡", "特权卡", "^金币$", "^粉钻$"],
        })
        self.assertNotIn("on_error", PIPELINE["等待签到页面就绪"])

    def test_native_timeout_cannot_reach_reward_or_exit(self):
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
            from tools.test_benefits import benefits, PIPELINE

            class Screen(CustomController):
                def connect(self): return True
                def request_uuid(self): return 'benefits-offline'
                def get_features(self): return 0
                def screencap(self): return np.zeros((1280, 720, 3), dtype=np.uint8)
                def unused(self, *args): return False
                start_app = stop_app = click = swipe = touch_down = touch_move = touch_up = unused
                click_key = input_text = key_down = key_up = unused

            controller = Screen()
            assert controller.post_connection().wait().succeeded
            tasker = Tasker()
            assert tasker.bind(resource, controller)

            class Ready(CustomRecognition):
                def analyze(self, context, argv):
                    return (40, 600, 80, 30) if self.visible else None

            reached = []
            class Reward(CustomAction):
                def run(self, context, argv):
                    reached.append(argv.node_name)
                    return True

            ready = Ready()
            assert resource.register_custom_recognition('ready', ready)
            assert resource.register_custom_action('benefits_wait_ready', benefits.BenefitsWaitReadyAction())
            assert resource.register_custom_action('reward', Reward())
            nodes = copy.deepcopy(PIPELINE)
            nodes['签到页面就绪']['recognition'] = {'type': 'Custom', 'param': {'custom_recognition': 'ready'}}
            nodes['等待签到页面就绪'].update(rate_limit=5, timeout=150)
            for name in nodes['等待签到页面就绪']['next']:
                nodes[name].update(recognition='DirectHit', action={'type': 'Custom', 'param': {'custom_action': 'reward'}}, next=[], post_delay=0)
            assert resource.override_pipeline(nodes)
            for visible in (True, False):
                ready.visible = visible
                reached.clear()
                job = tasker.post_task('等待签到页面就绪').wait()
                assert job.succeeded == visible, (visible, reached)
                assert reached == (['点击感叹号所在位置'] if visible else []), reached
            print('native benefits readiness cases passed')
        ''')
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-"], input=script,
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("native benefits readiness cases passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
