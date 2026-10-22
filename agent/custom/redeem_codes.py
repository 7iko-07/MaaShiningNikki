"""兑换码批次：导航、输入、结果识别及本次结果回传。"""

import json
import re
import time
from dataclasses import dataclass
from uuid import uuid4

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.pipeline import JClick
from utils import logger
from utils.ocr import extract_ocr_text


@dataclass
class RedemptionResult:
    success: bool
    message: str


class RedemptionError(RuntimeError):
    pass


class RedeemRunner:
    def __init__(self, context):
        self.context = context
        self.needs_clear = False

    def check_cancel(self):
        if self.context.tasker.stopping:
            raise RedemptionError("任务已取消")

    def image(self):
        self.check_cancel()
        controller = self.context.tasker.controller
        if not controller.post_screencap().wait().succeeded:
            raise RedemptionError("兑换码截图失败")
        image = controller.cached_image
        if image is None:
            raise RedemptionError("兑换码截图为空")
        return image

    def reco(self, node, image, expected=None):
        override = None if expected is None else {
            node: {"recognition": {"param": {"expected": [expected]}}}
        }
        result = self.context.run_recognition(node, image, pipeline_override=override)
        if result is None:
            raise RedemptionError(f"无法执行识别：{node}")
        return result

    def hit(self, node, image):
        return self.reco(node, image).hit

    def box(self, node, image):
        result = self.reco(node, image)
        if not result.hit or result.box is None:
            return None
        x, y, w, h = result.box
        return (x, y, w, h) if w > 0 and h > 0 else None

    def wait(self, predicate, description):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            value = predicate(self.image())
            if value:
                return value
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))
        raise RedemptionError(f"等待超时：{description}")

    def click(self, target):
        self.check_cancel()
        result = self.context.run_action_direct("Click", JClick(target=target))
        if not result or not result.success:
            raise RedemptionError("兑换码点击失败")

    def button(self, node):
        self.click(self.wait(lambda image: self.box(node, image), node))

    def task(self, node, override=None):
        self.check_cancel()
        # 文本输入复用织梦空间 Custom 动作，run_task 提供有效的 reco_id。
        result = self.context.run_task(node, pipeline_override=override)
        if not result or not result.status.succeeded:
            raise RedemptionError(f"兑换码动作失败：{node}")

    def dialog_ready(self, image):
        return (self.hit("兑换码弹窗标题", image)
                and not self.hit("兑换码恭喜获得", image)
                and not self.hit("兑换码输入确定", image))

    def empty_dialog(self, image):
        return self.dialog_ready(image) and self.hit("兑换码空输入框", image)

    def settings_ready(self, image):
        return self.hit("兑换码设置页面", image) and not self.hit("兑换码弹窗标题", image)

    def open(self):
        self.task("兑换码前往名片")
        self.button("兑换码设置入口")
        self.wait(self.settings_ready, "设置页面")
        self.button("兑换码弹窗入口")
        self.wait(self.empty_dialog, "空兑换码输入框")

    def clear_previous(self):
        self.wait(self.dialog_ready, "兑换码弹窗")
        self.button("兑换码输入区域")
        self.wait(lambda image: self.hit("兑换码输入确定", image), "文字输入界面")
        self.task("兑换码全选旧码")
        self.button("兑换码输入确定")
        self.wait(self.empty_dialog, "清空后收起文字输入界面")
        self.needs_clear = False

    def redeem_one(self, code):
        if self.needs_clear:
            self.clear_previous()
        self.wait(self.empty_dialog, "空兑换码输入框")
        claim = self.wait(lambda image: self.box("兑换码领取", image), "领取按钮")
        self.button("兑换码空输入框")
        self.wait(lambda image: self.hit("兑换码输入确定", image), "文字输入界面")
        self.task("兑换码输入文本", {
            "兑换码输入文本": {"action": {"param": {"custom_action_param": {"input_text": code}}}}
        })
        # 第一次领取只收起文字输入界面；确认文字已落入游戏后，再真正提交。
        self.click(claim)
        self.wait(lambda image: self.dialog_ready(image) and self.reco(
            "兑换码输入区域", image, "^" + re.escape(code) + "$"
        ).hit, "兑换码已输入并收起文字输入界面")
        self.button("兑换码领取")
        result = self.wait_result()
        self.needs_clear = not result.success
        return result

    def wait_result(self):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            image = self.image()
            if self.hit("兑换码恭喜获得", image):
                return RedemptionResult(True, "兑换成功")
            reason = extract_ocr_text(self.reco("兑换码失败提示", image))
            if reason:
                return RedemptionResult(False, reason.strip())
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        return RedemptionResult(False, "未识别到兑换结果")

    def close_reward(self):
        self.wait(lambda image: self.hit("兑换码恭喜获得", image), "恭喜获得弹窗")
        self.click([343, 85, 37, 20])
        self.wait(self.empty_dialog, "奖励弹窗关闭后自动清空的兑换框")

    def close(self):
        self.wait(self.dialog_ready, "兑换码弹窗")
        self.click([343, 85, 37, 20])
        self.wait(self.settings_ready, "返回设置页面")
        self.click([343, 85, 37, 20])
        self.wait(lambda image: self.hit("导航在名片页面", image), "返回名片页面")
        self.click([343, 85, 37, 20])
        self.wait(lambda image: not self.hit("玩家名片页面点击esc", image)
                  and self.hit("是否是主页面", image), "返回主页面")


def report(run_id, kind, **values):
    # 独立的一行 JSON，经客户端已有 Agent 输出管道回传，不保存兑换历史。
    print("MFA_REDEEM:" + json.dumps(
        {"run_id": run_id, "kind": kind, **values}, ensure_ascii=False), flush=True)


@AgentServer.custom_action("redeem_codes")
class RedeemCodesAction(CustomAction):
    def run(self, context, argv):
        params = json.loads(argv.custom_action_param or "{}")
        run_id = params.get("run_id") or uuid4().hex
        try:
            codes = params.get("codes")
            if not isinstance(codes, list) or any(not isinstance(c, str) or not c.strip() for c in codes):
                raise ValueError("兑换码列表格式错误")
            codes = list(dict.fromkeys(codes))
            report(run_id, "begin", codes=codes)
            if not codes:
                logger.info("当前没有可兑换的兑换码")
                return True
            runner = RedeemRunner(context)
            runner.open()
            all_success = True
            for code in codes:
                if context.tasker.stopping:
                    return False
                report(run_id, "running", code=code)
                result = runner.redeem_one(code)
                report(run_id, "result", code=code, success=result.success, message=result.message)
                logger.info(f"兑换码 {code}：{result.message}")
                all_success = all_success and result.success
                if result.success:
                    runner.close_reward()
            runner.close()
            return all_success
        except Exception as exc:
            logger.error(f"兑换码任务失败：{exc}")
            report(run_id, "error", message=str(exc))
            return False
        finally:
            report(run_id, "end")
