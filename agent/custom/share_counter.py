import json

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils import logger


@AgentServer.custom_action("share_counter")
class ShareCounterAction(CustomAction):
    _task_share_counts = {}

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        operation = params.get("operation", "count")
        task_id = argv.task_detail.task_id

        if operation == "reset":
            self._task_share_counts.pop(task_id, None)
            logger.debug("share_counter: reset share count")
            return True

        if operation != "count":
            logger.debug(f"share_counter: invalid operation={operation!r}")
            logger.error("分享：操作配置无效")
            return False

        max_shares = self._as_int(params.get("max_shares"), 0)
        if not 0 <= max_shares <= 5:
            logger.debug(f"share_counter: max_shares must be between 0 and 5, got {max_shares}")
            logger.error("分享：次数须为 0～5，请检查设置")
            return False

        if max_shares == 0:
            logger.debug("share_counter: unlimited mode, continuing with reward detection")
            return True

        share_count = self._task_share_counts.get(task_id, 0) + 1
        self._task_share_counts[task_id] = share_count
        logger.debug(f"share_counter: share_count={share_count}, max_shares={max_shares}")
        logger.info(f"分享：已执行 {share_count} 次")

        if share_count < max_shares:
            return True

        self._task_share_counts.pop(task_id, None)
        if not context.override_next(argv.node_name, ["stop"]):
            logger.debug("share_counter: failed to stop after reaching share limit")
            logger.error("分享：达到次数上限后未能正常结束")
            return False

        logger.debug("share_counter: reached share limit, stopping task")
        logger.info("分享：已达到设置次数，任务结束")
        return True

    def _as_int(self, value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
