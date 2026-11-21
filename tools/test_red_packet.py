"""红包弹窗的单次发送、取消、重入和检测限频检查。"""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from maa.agent.agent_server import AgentServer
from maa.define import Rect


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location("red_packet_under_test", ROOT / "agent/custom/red_packet.py")
module = importlib.util.module_from_spec(spec)
with patch.object(AgentServer, "register_custom_action", return_value=True), \
        patch.object(AgentServer, "add_context_sink"):
    spec.loader.exec_module(module)


class RedPacketTest(unittest.TestCase):
    def context(self):
        context = Mock()
        context.tasker.stopping = False
        context.tasker._handle = 42
        context.tasker.controller.cached_image = SimpleNamespace(size=100)
        return context

    def run_popup(self, *, send_stuck=False, original_reward=False):
        context = self.context()
        stages = [
            {module.ICON},
            {module.HEADER, module.REWARD, module.THANK},
            {module.HEADER, module.THANK},
            {module.HEADER, module.DIALOG, module.SEND},
            {module.HEADER, module.EXIT},
            set(),
        ]
        if original_reward:
            stages[-1] = {module.REWARD}
        stage = 0
        clicks = []
        runner = module.RedPacketRunner(context)
        runner.frame = lambda: stage

        def recognize(name, image):
            return SimpleNamespace(hit=name in stages[image], box=Rect(10, 20, 30, 40))

        def click(action, params):
            nonlocal stage
            clicks.append(stage)
            if not (stage == 3 and send_stuck):
                stage += 1
            return SimpleNamespace(success=True)

        context.run_recognition.side_effect = recognize
        context.run_action_direct.side_effect = click
        return runner, clicks

    def test_reward_then_default_thanks_sent_once_and_exit(self):
        runner, clicks = self.run_popup()
        self.assertTrue(runner.run())
        self.assertEqual(clicks, [0, 1, 2, 3, 4])
        self.assertEqual(runner.context.run_action_direct.call_args.args[1].target, (10, 20, 30, 40))

    def test_send_timeout_does_not_send_again(self):
        runner, clicks = self.run_popup(send_stuck=True)
        runner.deadline = 10
        with patch.object(module.time, "monotonic", side_effect=range(100)), \
                patch.object(module.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "退出按钮超时"):
                runner.run()
        self.assertEqual(clicks.count(3), 1)
        self.assertNotIn(4, clicks)

    def test_exit_can_resume_business_reward_popup(self):
        runner, clicks = self.run_popup(original_reward=True)
        self.assertTrue(runner.run())
        self.assertEqual(clicks, [0, 1, 2, 3, 4])

    def test_manual_stop_does_not_click(self):
        context = self.context()
        context.tasker.stopping = True
        with self.assertRaises(InterruptedError):
            module.RedPacketRunner(context).run()
        context.run_action_direct.assert_not_called()

    def test_invalid_box_does_not_click(self):
        context = self.context()
        with self.assertRaisesRegex(RuntimeError, "点击位置"):
            module.RedPacketRunner(context).click(SimpleNamespace(box=Rect(10, 20, 0, 40)))
        context.run_action_direct.assert_not_called()

    def test_sink_reentry_and_one_second_check_interval(self):
        sink = module.RedPacketSink()
        context = self.context()

        def recognize(name, image):
            sink.on_raw_notification(context, "Node.Action.Starting", {"name": "other"})
            return SimpleNamespace(hit=True)

        context.run_recognition.side_effect = recognize
        context.run_task.return_value = SimpleNamespace(status=SimpleNamespace(succeeded=True))
        with patch.object(module.time, "monotonic", return_value=10):
            sink.on_raw_notification(context, "Node.Action.Starting", {"name": "business"})
            sink.on_raw_notification(context, "Node.Action.Starting", {"name": "business"})
        context.run_recognition.assert_called_once()
        context.run_task.assert_called_once_with(module.ENTRY)
        context.tasker.post_stop.assert_not_called()
        self.assertFalse(sink._busy)

    def test_handler_failure_stops_business_task(self):
        sink = module.RedPacketSink()
        context = self.context()
        context.run_recognition.return_value = SimpleNamespace(hit=True)
        context.run_task.return_value = SimpleNamespace(status=SimpleNamespace(succeeded=False))
        sink.on_raw_notification(context, "Node.Action.Starting", {"name": "business"})
        context.tasker.post_stop.assert_called_once_with()

    def test_negative_detection_uses_cached_frame_without_screencap(self):
        sink = module.RedPacketSink()
        context = self.context()
        context.run_recognition.return_value = SimpleNamespace(hit=False)
        sink.on_raw_notification(context, "Node.NextList.Starting", {"name": "business"})
        context.tasker.controller.post_screencap.assert_not_called()
        context.run_task.assert_not_called()


def native_resume():
    from maa.custom_action import CustomAction
    from maa.custom_recognition import CustomRecognition
    from maa.resource import Resource
    from maa.tasker import LoggingLevelEnum, Tasker
    from tools.test_make_target_clothes_readiness import ScreenController

    Tasker.set_stdout_level(LoggingLevelEnum.Off)
    events = []
    monitor = module.RedPacketSink()

    class Screen(ScreenController):
        phase = 5

        def click(self, x, y):
            events.append(f"click{self.phase}")
            self.phase += 1
            return True

    screen = Screen()
    states = {
        module.ICON: (0,), module.HEADER: (1, 2, 3, 4), module.REWARD: (1,),
        module.THANK: (1, 2), module.DIALOG: (3,), module.SEND: (3,), module.EXIT: (4,),
    }

    class Recognition(CustomRecognition):
        def analyze(self, context, argv):
            if screen.phase in states[argv.node_name]:
                return self.AnalyzeResult(box=(10, 20, 30, 40), detail={})
            return None

    class Business(CustomAction):
        def run(self, context, argv):
            events.append(argv.node_name)
            if argv.node_name == "Before":
                screen.phase = 0
                # 模拟下一次检测时刻到来，避免为了限频检查固定等待一秒。
                monitor._checked_at.clear()
            return True

    resource = Resource()
    assert resource.register_custom_recognition("screen", Recognition())
    assert resource.register_custom_action("business", Business())
    assert resource.register_custom_action("handle_red_packet", module.RedPacketAction())
    nodes = {name: {"recognition": {"type": "Custom", "param": {"custom_recognition": "screen"}}}
             for name in states}
    nodes[module.ENTRY] = {"action": {"type": "Custom", "param": {"custom_action": "handle_red_packet"}},
                           "post_delay": 0}
    nodes.update({name: {"action": {"type": "Custom", "param": {"custom_action": "business"}}, "post_delay": 0}
                  for name in ("Before", "After")})
    nodes["Before"]["next"] = ["After"]
    assert resource.override_pipeline(nodes)
    assert screen.post_connection().wait().succeeded
    tasker = Tasker()
    assert tasker.bind(resource, screen)
    assert tasker.add_context_sink(monitor) is not None
    assert tasker.post_task("Before").wait().succeeded
    assert events == ["Before", "click0", "click1", "click2", "click3", "click4", "After"], events


class NativeResumeTest(unittest.TestCase):
    def test_native_sink_handles_popup_and_resumes_without_restarting(self):
        code = (
            "from maa.library import Library; Library.version(); "
            "from tools.test_red_packet import native_resume; native_resume()"
        )
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=ROOT,
                                capture_output=True, encoding="utf-8", errors="replace", timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Exception ignored on calling ctypes callback", result.stderr)


if __name__ == "__main__":
    unittest.main()
