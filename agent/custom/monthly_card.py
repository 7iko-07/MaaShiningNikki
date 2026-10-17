import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger


@AgentServer.custom_action("monthly_card_popup")
class MonthlyCardPopupAction(CustomAction):
    """关闭特惠弹窗，确认消失后才交回领取流程。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            # 使用框架解析后的节点配置，不在 Agent 中重复定义默认值。
            node = context.get_node_data(argv.node_name)
            if not node or node["timeout"] <= 0 or node["rate_limit"] <= 0:
                logger.error("月卡弹窗检查：缺少有限超时或有效识别间隔")
                return False
            interval = node["rate_limit"] / 1000
            deadline = time.monotonic() + node["timeout"] / 1000
            controller = context.tasker.controller
            previously_absent = False

            while time.monotonic() < deadline:
                started = time.monotonic()
                if context.tasker.stopping:
                    return False
                if not controller.post_screencap().wait().succeeded:
                    logger.error("月卡弹窗检查：截图失败")
                    return False
                image = controller.cached_image
                popup = context.run_recognition("月卡特惠弹窗", image)
                tab = context.run_recognition("商城点击月卡", image)
                if popup is None or tab is None:
                    logger.error("月卡弹窗检查：识别执行失败")
                    return False
                if context.tasker.stopping:
                    return False
                if time.monotonic() >= deadline:
                    break

                # 背景月卡标签也必须可见；连续两帧未见弹窗才继续，避免一次漏识别。
                if not popup.hit and tab.hit:
                    if previously_absent:
                        return True
                    previously_absent = True
                else:
                    previously_absent = False
                    if popup.hit and tab.hit:
                        box = tab.box
                        if box is None or box.w <= 0 or box.h <= 0:
                            logger.error("月卡弹窗检查：月卡标签识别框无效")
                            return False
                        if not controller.post_click(
                            int(box.x + box.w / 2), int(box.y + box.h / 2)
                        ).wait().succeeded:
                            logger.error("月卡弹窗检查：点击月卡区域失败")
                            return False

                delay = min(
                    max(0, interval - (time.monotonic() - started)),
                    max(0, deadline - time.monotonic()),
                )
                if delay:
                    time.sleep(delay)
        except Exception as exc:
            logger.error(f"月卡弹窗检查失败：{exc}")
            return False

        logger.error("月卡特惠弹窗关闭失败：等待超时，停止领取")
        return False
