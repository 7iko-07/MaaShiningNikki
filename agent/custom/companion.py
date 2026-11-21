from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from utils import logger


@AgentServer.custom_action("companion_run")
class CompanionAction(CustomAction):
    def run(self, context, argv):
        for name in ("时光钟表铺", "不落的帷幕", "印象旅航"):
            if context.tasker.stopping:
                return False
            data = context.get_node_data("结伴选择" + name)
            if not data or not data.get("enabled"):
                continue
            logger.info(f"结伴：开始{name}")
            # 子任务返回后继续下一项，不执行原独立任务的 StopTask。
            result = context.run_task(
                name + "任务链入口",
                {"点击" + name + "返回": {"next": []}},
            )
            if not result or not result.status.succeeded:
                logger.error(f"结伴：{name}执行失败")
                return False
        return True
