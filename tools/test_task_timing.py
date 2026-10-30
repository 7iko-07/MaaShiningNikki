"""Task timing lifecycle checks without a game or installed package."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from maa.event_sink import NotificationType as Event
import task_timing


class TaskTimingTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.log = self.enterContext(patch.object(task_timing, "logger"))
        self.clock = self.enterContext(patch.object(task_timing.time, "monotonic"))

    def interface(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"task": [{"name": "好友", "entry": "好友任务链入口"}]}), encoding="utf-8")

    def emit(self, sink, event, seconds, task_id=1, uuid="account", entry="好友任务链入口"):
        self.clock.return_value = seconds
        sink.on_tasker_task(None, event, SimpleNamespace(task_id=task_id, uuid=uuid, entry=entry))

    def test_source_and_release_names_and_whole_task_elapsed(self):
        for relative in ("assets/interface.json", "interface.json"):
            with self.subTest(relative=relative):
                self.interface(self.root / relative)
                sink = task_timing.TaskTimingSink(self.root)
                self.log.reset_mock()
                self.emit(sink, Event.Starting, 100)
                self.emit(sink, Event.Unknown, 102)
                self.log.info.assert_not_called()
                self.emit(sink, Event.Succeeded, 112.345)
                self.log.info.assert_called_once_with("任务「好友」完成，耗时 12.34 秒")
                self.log.error.assert_not_called()

    def test_failed_task_logs_error_and_clears_timing(self):
        sink = task_timing.TaskTimingSink(self.root)
        self.emit(sink, Event.Starting, 10, entry="未知入口")
        self.emit(sink, Event.Failed, 15, entry="未知入口")
        self.log.error.assert_called_once_with("任务「未知入口」失败，耗时 5.00 秒")
        self.log.info.assert_not_called()
        self.assertFalse(sink._started)

    def test_story_normal_end_uses_neutral_wording_but_real_failure_is_error(self):
        sink = task_timing.TaskTimingSink(self.root)
        entry = "剧情自动点击任务链入口"
        self.emit(sink, Event.Starting, 10, entry=entry)
        self.emit(sink, Event.Succeeded, 15, entry=entry)
        self.log.info.assert_called_once_with("任务「剧情自动点击任务链入口」结束，耗时 5.00 秒")
        self.log.error.assert_not_called()
        self.emit(sink, Event.Starting, 20, entry=entry)
        self.emit(sink, Event.Failed, 25, entry=entry)
        self.log.error.assert_called_once_with("任务「剧情自动点击任务链入口」失败，耗时 5.00 秒")

    def test_interleaved_tasks_accounts_and_reentry_are_independent(self):
        sink = task_timing.TaskTimingSink(self.root)
        self.emit(sink, Event.Starting, 10)
        self.emit(sink, Event.Starting, 11, task_id=2)
        self.emit(sink, Event.Starting, 12, uuid="other")
        self.emit(sink, Event.Succeeded, 15, task_id=2)
        self.emit(sink, Event.Failed, 16, uuid="other")
        self.emit(sink, Event.Succeeded, 20)
        self.emit(sink, Event.Starting, 30)
        self.emit(sink, Event.Succeeded, 32)
        self.assertEqual([call.args[0] for call in self.log.info.call_args_list], [
            "任务「好友任务链入口」完成，耗时 4.00 秒",
            "任务「好友任务链入口」完成，耗时 10.00 秒",
            "任务「好友任务链入口」完成，耗时 2.00 秒",
        ])
        self.log.error.assert_called_once_with("任务「好友任务链入口」失败，耗时 4.00 秒")
        self.assertFalse(sink._started)

    def test_unmatched_and_duplicate_events_do_not_fabricate_duration(self):
        sink = task_timing.TaskTimingSink(self.root)
        self.emit(sink, Event.Succeeded, 5)
        self.log.info.assert_not_called()
        self.emit(sink, Event.Starting, 10)
        self.emit(sink, Event.Starting, 12)
        self.emit(sink, Event.Succeeded, 15)
        self.emit(sink, Event.Succeeded, 16)
        self.log.info.assert_called_once_with("任务「好友任务链入口」完成，耗时 5.00 秒")

    def test_invalid_interface_does_not_break_task_timing(self):
        (self.root / "interface.json").write_text("{", encoding="utf-8")
        sink = task_timing.TaskTimingSink(self.root)
        self.log.warning.assert_called_once()
        self.emit(sink, Event.Starting, 1)
        self.emit(sink, Event.Succeeded, 2)
        self.log.info.assert_called_once_with("任务「好友任务链入口」完成，耗时 1.00 秒")


class NativeTaskTimingTest(unittest.TestCase):
    def test_real_complete_task_events(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--native"],
            cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", timeout=40,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def native_task_timing():
    import numpy as np
    from maa.controller import CustomController
    from maa.custom_action import CustomAction
    from maa.resource import Resource
    from maa.tasker import Tasker
    from maa.toolkit import Toolkit

    class Controller(CustomController):
        def connect(self):
            return True

        def request_uuid(self):
            return "task-timing-test"

        def get_features(self):
            return 0

        def screencap(self):
            return np.zeros((720, 720, 3), dtype=np.uint8)

    class FailedAction(CustomAction):
        def run(self, context, argv):
            return False

    Toolkit.init_option(ROOT)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "pipeline").mkdir()
        (root / "pipeline/test.json").write_text(json.dumps({
            "好友任务链入口": {"post_delay": 150, "next": ["计时子节点"]},
            "计时子节点": {"post_delay": 0},
            "邮件任务链入口": {"action": "Custom", "custom_action": "fail"},
        }, ensure_ascii=False), encoding="utf-8")
        resource = Resource()
        assert resource.register_custom_action("fail", FailedAction())
        assert resource.post_bundle(root).wait().status.succeeded
        controller = Controller()
        assert controller.post_connection().wait().status.succeeded
        tasker = Tasker()
        assert tasker.bind(resource, controller)
        with patch.object(task_timing, "logger") as log:
            assert tasker.add_sink(task_timing.TaskTimingSink(ROOT)) is not None
            assert tasker.post_task("好友任务链入口").wait().status.succeeded
            assert tasker.post_task("邮件任务链入口").wait().status.failed
            log.info.assert_called_once()
            message = log.info.call_args.args[0]
            assert message.startswith("任务「好友」完成，耗时 "), message
            assert float(message.split("耗时 ")[1].split()[0]) >= .15, message
            log.error.assert_called_once()
            assert log.error.call_args.args[0].startswith("任务「邮件」失败，耗时 ")
        del tasker
        del resource


if __name__ == "__main__":
    if "--native" in sys.argv:
        native_task_timing()
    else:
        unittest.main()
