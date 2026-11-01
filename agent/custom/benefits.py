import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import debug_exception, logger


@AgentServer.custom_action("benefits_wait_ready")
class BenefitsWaitReadyAction(CustomAction):
    """连续两帧确认签到内容加载后，再交给奖励识别流程。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            node = context.get_node_data(argv.node_name)
            interval = node["rate_limit"] / 1000
            started = time.monotonic()
            deadline = started + node["timeout"] / 1000
            controller = context.tasker.controller
            hits = 0

            while time.monotonic() < deadline:
                if context.tasker.stopping:
                    return False
                frame_started = time.monotonic()
                capture = controller.post_screencap().wait()
                image = controller.cached_image if capture.succeeded else None
                if image is None:
                    logger.error("每日签到：截图失败")
                    return False
                if context.tasker.stopping:
                    return False
                if time.monotonic() >= deadline:
                    break

                result = context.run_recognition("签到页面就绪", image)
                if result is None:
                    logger.error("每日签到：页面识别执行失败")
                    return False
                if context.tasker.stopping:
                    return False
                if time.monotonic() >= deadline:
                    break
                hits = hits + 1 if result.hit else 0
                if hits >= 2:
                    logger.debug(f"每日签到：连续两帧确认页面就绪，耗时 {time.monotonic() - started:.3f}s")
                    return True

                delay = min(
                    max(0, interval - (time.monotonic() - frame_started)),
                    max(0, deadline - time.monotonic()),
                )
                if delay:
                    time.sleep(delay)
        except Exception:
            debug_exception("benefits_wait_ready: 签到页面就绪检查异常")
            logger.error("每日签到：无法确认签到页面")
            return False

        logger.error("每日签到：等待签到页面超时")
        return False
