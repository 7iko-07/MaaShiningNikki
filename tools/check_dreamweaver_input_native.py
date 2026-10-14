"""验证重命名输入经过真实原生回调；模拟外部输入进程，不操作模拟器。"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
from maa.library import Library

# 装饰器导入前初始化宿主库。
Library.version()

from maa.agent.agent_server import AgentServer
from maa.controller import Controller, CustomController
from maa.custom_action import CustomAction
from maa.resource import Resource
from maa.tasker import LoggingLevelEnum, Tasker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location("dream_input_native", ROOT / "agent/custom/dreamweaver_space.py")
dream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dream
with patch.object(AgentServer, "register_custom_action", return_value=True):
    spec.loader.exec_module(dream)


class MemoryController(CustomController):
    def connect(self):
        return True

    def request_uuid(self):
        return "dream-input-offline"

    def get_features(self):
        return 0

    def screencap(self):
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    def unsupported(self, *args):
        return False

    click = swipe = touch_down = touch_move = touch_up = unsupported
    start_app = stop_app = click_key = input_text = key_down = key_up = unsupported


class RecordingInput(dream.DreamweaverInputTextAction):
    def __init__(self):
        super().__init__()
        self.reco_ids = []

    def run(self, context, argv):
        assert argv.reco_detail.hit and argv.reco_detail.reco_id > 0
        self.reco_ids.append(argv.reco_detail.reco_id)
        return super().run(context, argv)


class Driver(CustomAction):
    def run(self, context, argv):
        previous = context.run_action("七日制衣输入临时")
        assert previous is None or not previous.success
        assert not action.reco_ids

        runner = dream.DreamRunner(context)
        runner.button = Mock()
        runner.click = Mock()
        runner.wait = Mock(return_value=(704, 429, 54, 31))
        info = {"type": "adb", "config": {"extras": {"mumu": {
            "enable": True, "path": "D:/MuMuPlayer", "index": 3,
        }}}}
        # 只替换设备配置和进程边界，识别、嵌套任务及 SDK 回调均走原生框架。
        with patch.object(Controller, "info", new_callable=PropertyMock, return_value=info), \
             patch.object(dream.Path, "is_file", return_value=True), \
             patch.object(dream.subprocess, "run", return_value=SimpleNamespace(
                 returncode=0, stdout='{"errcode":0}',
             )) as command:
            runner.seven_rename()
            assert len(action.reco_ids) == 1
            assert command.call_count == 1
            assert command.call_args.args[0][1:] == [
                "control", "-v", "3", "tool", "cmd", "--cmd", "input_text", "--text", "临时",
            ]
            runner.button.assert_any_call("七日制衣名称确定")
            runner.click.assert_called_once_with((704, 429, 54, 31))
            runner.button.reset_mock()
            runner.click.reset_mock()
            command.return_value = SimpleNamespace(returncode=0, stdout='{"errcode":1}')
            try:
                runner.seven_rename()
            except dream.DreamError:
                pass
            else:
                raise AssertionError("输入失败没有向上传播")
            assert len(action.reco_ids) == 2 and command.call_count == 2
            runner.button.assert_called_once_with("七日制衣点击名称框")
            runner.click.assert_not_called()
        return True


if __name__ == "__main__":
    Tasker.set_stdout_level(LoggingLevelEnum.Error)
    resource = Resource()
    assert resource.post_bundle(ROOT / "assets/resource").wait().succeeded
    action, driver = RecordingInput(), Driver()
    assert resource.register_custom_action("dreamweaver_input_text", action)
    assert resource.register_custom_action("input_native_driver", driver)
    controller = MemoryController()
    assert controller.post_connection().wait().succeeded
    tasker = Tasker()
    assert tasker.bind(resource, controller)
    job = tasker.post_task("InputNativeCheck", {
        "InputNativeCheck": {"action": {
            "type": "Custom", "param": {"custom_action": "input_native_driver"},
        }},
    }).wait()
    assert job.succeeded, "Native input callback check failed"
    print(f"PASS maafw={Library.version()}: old call rejected; "
          f"new callback reco_ids={action.reco_ids}; command failure blocks confirmation")
