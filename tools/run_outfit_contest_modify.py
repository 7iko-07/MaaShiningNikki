"""实机修改作品测试：手动进入显示“修改作品”的评选赛页面后运行。

会重新选择第一个套装并实际投稿；取消两项同步勾选，上传后返回确认。
直接测试投稿链，不执行主任务的导航与时段分流，每步沿用30秒超时。
仅在本脚本的 Resource 中覆盖识别和动作，不改写正式配置或 install/。
"""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = str(ROOT / "agent")
if AGENT_PATH not in sys.path:
    sys.path.insert(0, AGENT_PATH)
ENTRY = "搭配评选赛自动参赛"
ENTER_BUTTON = "搭配评选赛点击我要参赛"
SUBMITTED = "搭配评选赛已参赛"
FIRST_OUTFIT = "搭配评选赛选择首套"


def create_test_resource():
    from maa.library import Library

    # 先加载宿主库，再把源模块装饰器注册到本脚本的 Resource。
    Library.version()
    from maa.agent.agent_server import AgentServer
    from maa.resource import Resource

    resource = Resource()
    if not resource.post_bundle(ROOT / "assets/resource").wait().succeeded:
        raise RuntimeError("源资源加载失败")

    def load_source(name):
        spec = importlib.util.spec_from_file_location(
            f"outfit_modify_test_{name}", ROOT / "agent/custom" / f"{name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        with patch.object(AgentServer, "register_custom_action", side_effect=resource.register_custom_action):
            spec.loader.exec_module(module)
        return module

    contest = load_source("outfit_contest")
    load_source("click_and_wait_change")

    class ModifySubmission(contest.OutfitContestSubmit):
        def _wait(self, context, nodes, deadline=None, unchecked=None):
            # 仅替换首次进入的候选；末尾仍通过“修改作品”确认返回成功。
            if nodes == [SUBMITTED, ENTER_BUTTON]:
                nodes = [ENTER_BUTTON]
            return super()._wait(context, nodes, deadline, unchecked)

    test_action = "outfit_contest_modify_test"
    if not resource.register_custom_action(test_action, ModifySubmission()):
        raise RuntimeError("测试动作注册失败")
    first_action = resource.get_node_data(FIRST_OUTFIT)["action"]["param"]["custom_action_param"]
    if not resource.override_pipeline({
        ENTRY: {"action": {"type": "Custom", "param": {"custom_action": test_action}}},
        ENTER_BUTTON: {"recognition": resource.get_node_data(SUBMITTED)["recognition"]},
        FIRST_OUTFIT: {"action": {"type": "Click", "param": {"target": first_action["target"]}}},
    }):
        raise RuntimeError("测试配置覆盖失败")
    return resource


def run_test(resource, controller):
    from maa.tasker import Tasker

    tasker = Tasker()
    if not tasker.bind(resource, controller):
        raise RuntimeError("设备与资源绑定失败")
    try:
        job = tasker.post_task(ENTRY)
        while not job.done:
            time.sleep(.2)
        return job.succeeded
    finally:
        if tasker.running:
            tasker.post_stop().wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", type=Path, help="指定用于发现设备的 adb.exe 路径")
    parser.add_argument("--address", help="指定发现列表中的设备地址，多设备时必须指定")
    parser.add_argument("--list-devices", action="store_true", help="仅列出设备，不执行测试")
    args = parser.parse_args()
    os.chdir(ROOT)

    from maa.library import Library

    Library.version()
    from maa.controller import AdbController
    from maa.define import LoggingLevelEnum
    from maa.tasker import Tasker
    from maa.toolkit import Toolkit

    Tasker.set_log_dir(ROOT / "debug/outfit_contest_modify")
    Tasker.set_stdout_level(LoggingLevelEnum.Error)
    try:
        devices = Toolkit.find_adb_devices(args.adb)
        if args.list_devices:
            for device in devices:
                print(f"{device.name} | {device.address} | {device.adb_path}")
            if not devices:
                print("未发现设备，请启动模拟器或使用 --adb 指定路径")
            return 0 if devices else 1
        selected = [device for device in devices if not args.address or device.address == args.address]
        if len(selected) != 1:
            for device in devices:
                print(f"{device.name} | {device.address} | {device.adb_path}")
            raise RuntimeError("未能唯一确定设备，请使用 --address 指定地址，必要时同时指定 --adb")
        device = selected[0]
        resource = create_test_resource()
        controller = AdbController(
            device.adb_path, device.address, device.screencap_methods, device.input_methods, device.config,
        )
        if not controller.set_screenshot_target_short_side(720):
            raise RuntimeError("截图短边720设置失败")
        if not controller.post_connection().wait().succeeded:
            raise RuntimeError("设备连接失败")
        print(f"连接 {device.name}（{device.address}），开始修改作品并重新投稿；Ctrl+C 停止")
        succeeded = run_test(resource, controller)
        print("测试完成：已上传并返回修改作品页面" if succeeded else "测试失败，请查看日志和当前游戏页面")
        return 0 if succeeded else 1
    except KeyboardInterrupt:
        print("测试已停止")
        return 130
    except Exception as exc:
        from utils import debug_exception, logger

        debug_exception("修改作品实机测试异常")
        logger.error(f"修改作品测试：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
