"""用设计页截图验证真实 OCR / Custom 回调；只连接内存控制器，不操作模拟器。"""

import importlib.util
import sys
import struct
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from maa.library import Library

# 导入 Agent 装饰器前初始化宿主原生库，避免切换成 AgentServer 库。
Library.version()

from maa.agent.agent_server import AgentServer
from maa.controller import CustomController
from maa.custom_action import CustomAction
from maa.pipeline import JCustomAction
from maa.resource import Resource
from maa.tasker import Tasker, LoggingLevelEnum

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "agent/custom" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with patch.object(AgentServer, "register_custom_action", return_value=True):
        spec.loader.exec_module(module)
    return module


dream = load_module("dream_native_check", "dreamweaver_space.py")
click_module = load_module("click_native_check", "click_and_wait_change.py")


class ScreenshotController(CustomController):
    def __init__(self, frame):
        self.frame = frame
        self.clicks = []
        super().__init__()

    def connect(self):
        return True

    def request_uuid(self):
        return "dream-production-offline"

    def get_features(self):
        return 0

    def screencap(self):
        return self.frame.copy()

    def click(self, x, y):
        self.clicks.append((x, y))
        return True

    def unsupported(self, *args):
        return False

    start_app = stop_app = swipe = touch_down = touch_move = touch_up = unsupported
    click_key = input_text = key_down = key_up = unsupported


class RecordingClick(click_module.ClickAndWaitChangeAction):
    def __init__(self):
        super().__init__()
        self.recognitions = []

    def run(self, context, argv):
        assert argv.reco_detail.reco_id > 0
        assert argv.reco_detail.hit
        self.recognitions.append(argv.reco_detail)
        return super().run(context, argv)


class Driver(CustomAction):
    def run(self, context, argv):
        # 复现旧调用：SDK 拒绝 reco_id=0，不能进入点击函数。
        previous = context.run_action_direct(
            "Custom", JCustomAction(custom_action="click_and_wait_change"),
            box=(351, 621, 110, 38),
        )
        assert previous is None or not previous.success
        assert not controller.clicks
        runner = dream.DreamRunner(context)
        # 固定截图不会跳页；第二次点击后模拟工位页就绪，其余均走真实框架。
        runner.management_ready = lambda image, **kwargs: len(controller.clicks) >= 2
        quantity = runner.submit_production(deadline=time.monotonic() + 15)
        assert quantity == int(sys.argv[2])
        assert len(controller.clicks) == 2
        return True


if __name__ == "__main__":
    # 输入 24 位 BMP，避免为离线验证安装额外图像依赖。
    data = Path(sys.argv[1]).read_bytes()
    assert data[:2] == b"BM"
    offset = struct.unpack_from("<I", data, 10)[0]
    width, height = struct.unpack_from("<ii", data, 18)
    assert struct.unpack_from("<H", data, 28)[0] == 24
    assert struct.unpack_from("<I", data, 30)[0] == 0
    stride = (width * 3 + 3) // 4 * 4
    frame = np.frombuffer(data, dtype=np.uint8, offset=offset, count=abs(height) * stride)
    frame = frame.reshape(abs(height), stride)[:, :width * 3].reshape(abs(height), width, 3)
    if height > 0:
        frame = frame[::-1]
    frame = frame.copy()
    Tasker.set_stdout_level(LoggingLevelEnum.Error)
    resource = Resource()
    assert resource.post_bundle(ROOT / "assets/resource").wait().succeeded
    action, driver = RecordingClick(), Driver()
    assert resource.register_custom_action("click_and_wait_change", action)
    assert resource.register_custom_action("production_native_driver", driver)
    controller = ScreenshotController(frame)
    assert controller.post_connection().wait().succeeded
    tasker = Tasker()
    assert tasker.bind(resource, controller)
    job = tasker.post_task("ProductionNativeCheck", {
        "ProductionNativeCheck": {
            "action": {"type": "Custom", "param": {"custom_action": "production_native_driver"}},
        },
    }).wait()
    assert job.succeeded, "Native production callback check failed"
    assert len(action.recognitions) == 2
    for recognition, point in zip(action.recognitions, controller.clicks):
        box = recognition.box
        assert point == (int(box.x + box.w / 2), int(box.y + box.h / 2))
    print(f"PASS maafw={Library.version()}: old call rejected; OCR callbacks="
          f"{[r.reco_id for r in action.recognitions]}; clicks={controller.clicks}")
