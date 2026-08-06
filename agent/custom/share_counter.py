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
            logger.info("share_counter: reset share count")
            return True

        if operation != "count":
            logger.error(f"share_counter: invalid operation={operation!r}")
            return False

        max_shares = self._as_int(params.get("max_shares"), 0)
        if not 0 <= max_shares <= 5:
            logger.error(f"share_counter: max_shares must be between 0 and 5, got {max_shares}")
            return False

        if max_shares == 0:
            logger.info("share_counter: unlimited mode, continuing with reward detection")
            return True

        share_count = self._task_share_counts.get(task_id, 0) + 1
        self._task_share_counts[task_id] = share_count
        logger.info(f"share_counter: share_count={share_count}, max_shares={max_shares}")

        if share_count < max_shares:
            return True

        self._task_share_counts.pop(task_id, None)
        if not context.override_next(argv.node_name, ["stop"]):
            logger.error("share_counter: failed to stop after reaching share limit")
            return False

        logger.info("share_counter: reached share limit, stopping task")
        return True

    def _as_int(self, value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
