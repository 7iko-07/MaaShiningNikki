"""只在投稿页面稳定且连续确认无红点时跳过话题领奖。"""

import json
import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.pipeline import JWaitFreezes
from utils import logger


@AgentServer.custom_action("manicure_topic_dispatch")
class ManicureTopicDispatchAction(CustomAction):
    def run(self, context, argv):
        params = json.loads(argv.custom_action_param or "{}") or {}
        deadline = time.monotonic() + params.get("timeout", 10000) / 1000
        controller = context.tasker.controller
        no_red = 0
        # Every invocation restores its own successor; a prior skip cannot leak.
        if not context.override_next(argv.node_name, ["点击话题任务"]):
            return False
        while not context.tasker.stopping and time.monotonic() < deadline:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms < 200:
                break
            if not context.wait_freezes(wait_freezes_param=JWaitFreezes(
                time=200, target=(0, 1133, 720, 147), rate_limit=100, timeout=remaining_ms,
            )):
                break
            if context.tasker.stopping or time.monotonic() >= deadline:
                break
            if not controller.post_screencap().wait().status.succeeded:
                return False
            image = controller.cached_image
            loading = context.run_recognition("在加载页面", image)
            popup = context.run_recognition("美甲周期奖励确认", image)
            page = context.run_recognition("美甲话题页面就绪", image)
            if (loading is not None and not loading.hit and popup is not None and not popup.hit
                    and page is not None and page.hit):
                red = context.run_recognition("点击话题任务", image)
                if context.tasker.stopping or time.monotonic() >= deadline:
                    break
                if red is None:
                    return False
                if red.hit:
                    return True
                no_red += 1
                if no_red >= 2:
                    logger.info("美甲话题任务无红点，跳过领奖")
                    return context.override_next(argv.node_name, [])
            else:
                no_red = 0
            time.sleep(min(.2, max(0, deadline - time.monotonic())))
        logger.error("美甲话题任务：页面未就绪或等待超时，停止任务")
        return False
