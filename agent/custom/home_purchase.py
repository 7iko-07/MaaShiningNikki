"""日常采购限时查找并点击商品，购买仍由 Pipeline 执行。"""

import json
import time
from dataclasses import replace

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.pipeline import JOCR
from utils import debug_exception, logger

from .click_region import ClickCenterAction


@AgentServer.custom_action("home_purchase_find")
class HomePurchaseFind(CustomAction):
    def run(self, context, argv):
        try:
            return self._find(context, argv)
        except Exception:
            debug_exception("日常采购查找异常")
            logger.error("日常采购：查找商品失败，请检查当前页面")
            return False

    def _find(self, context, argv):
        params = json.loads(argv.custom_action_param)
        target = params["target_node"]
        skipped = params["skip_next"]
        target_node = context.get_node_data(target)
        purchase, = target_node["next"]
        if isinstance(purchase, dict):
            purchase = purchase["name"]
        # 上次超时可能改过 next；每次进入均恢复成功路径。
        if not context.override_next(argv.node_name, [purchase]):
            raise RuntimeError("无法恢复商品购买后继")
        config = target_node["recognition"]["param"]
        label = config["expected"][0].removeprefix("^").removesuffix("$")
        later_patterns = self._later_patterns(context, skipped)
        deadline = time.monotonic() + 30
        controller = context.tasker.controller

        def active():
            return not context.tasker.stopping and time.monotonic() < deadline

        while active():
            capture = controller.post_screencap().wait()
            if not capture.succeeded or controller.cached_image is None:
                raise RuntimeError("截图失败")
            if not active():
                break
            image = controller.cached_image
            result = context.run_recognition(target, image)
            if result is None:
                raise RuntimeError(f"无法执行商品识别：{target}")
            if not active():
                break
            if result.hit:
                return ClickCenterAction().run(
                    context, replace(argv, box=result.box, reco_detail=result),
                )
            # 后续商品已经在本屏时不再上划，避免找不到当前商品时把它们滑过去。
            later_visible = False
            if later_patterns:
                later = context.run_recognition_direct(
                    "OCR", JOCR(roi=config["roi"], expected=later_patterns), image,
                )
                if later is None:
                    raise RuntimeError("无法检查后续商品")
                later_visible = later.hit
            if not active():
                break
            if later_visible:
                time.sleep(min(0.25, max(0, deadline - time.monotonic())))
            else:
                result = context.run_task("上划日常采购")
                if not result or not result.status.succeeded:
                    raise RuntimeError("上划失败")

        if context.tasker.stopping:
            return False
        if not context.override_next(argv.node_name, [skipped]):
            raise RuntimeError("无法跳过未找到的商品")
        logger.warning(f"未找到{label}，已跳过")
        return True

    @staticmethod
    def _later_patterns(context, node_name):
        """沿本分类已选中的后继查找，切换分类前结束。"""
        patterns = []
        while node_name:
            node = context.get_node_data(node_name)
            action = node.get("action", {})
            action_type = action if isinstance(action, str) else action.get("type", "DoNothing")
            if action_type == "Custom" and action["param"].get("custom_action") == "home_purchase_find":
                params = action["param"]["custom_action_param"]
                target = context.get_node_data(params["target_node"])
                patterns.extend(target["recognition"]["param"]["expected"])
                node_name = params["skip_next"]
            elif action_type == "DoNothing":
                successors = node.get("next", [])
                edge = successors[0] if successors else ""
                node_name = edge.get("name", "") if isinstance(edge, dict) else edge
            else:
                break
        return patterns
