import json
import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.custom_recognition import CustomRecognition
from utils import logger


_task_run_counts = {}
_wait_started_at = {}


@AgentServer.custom_action("unending_curtain_counter")
class UnendingCurtainCounterAction(CustomAction):
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        operation = params.get("operation", "count")
        task_id = argv.task_detail.task_id

        if operation == "reset":
            _task_run_counts.pop(task_id, None)
            _wait_started_at.pop(task_id, None)
            logger.info("unending_curtain_counter: reset solo run count")
            return True

        if operation == "start_wait":
            _wait_started_at[task_id] = time.monotonic()
            logger.info("unending_curtain_counter: start solo result wait timer")
            return True

        if operation != "count":
            logger.error(f"unending_curtain_counter: invalid operation={operation!r}")
            return False

        _wait_started_at.pop(task_id, None)
        configured_runs = self._as_int(params.get("max_runs"), 0)
        if not 0 <= configured_runs <= 3:
            logger.error(
                "unending_curtain_counter: max_runs must be between 0 and 3, "
                f"got {configured_runs}"
            )
            return False

        max_runs = 3 if configured_runs == 0 else configured_runs
        run_count = _task_run_counts.get(task_id, 0) + 1
        _task_run_counts[task_id] = run_count
        logger.info(
            "unending_curtain_counter: "
            f"solo_run_count={run_count}, max_runs={max_runs}"
        )

        if run_count < max_runs:
            next_nodes = ["点击第一幕"]
        else:
            next_nodes = ["stop"]
            _task_run_counts.pop(task_id, None)

        if not context.override_next(argv.node_name, next_nodes):
            logger.error(
                "unending_curtain_counter: failed to set next nodes "
                f"to {next_nodes!r}"
            )
            return False

        return True

    @staticmethod
    def _as_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default


@AgentServer.custom_recognition("unending_curtain_wait_timeout")
class UnendingCurtainWaitTimeoutRecognition(CustomRecognition):
    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult | None:
        params = (
            json.loads(argv.custom_recognition_param)
            if argv.custom_recognition_param
            else {}
        )
        timeout_seconds = self._as_float(params.get("timeout_seconds"), 180.0)
        if timeout_seconds <= 0:
            logger.error(
                "unending_curtain_wait_timeout: timeout_seconds must be positive, "
                f"got {timeout_seconds}"
            )
            return None

        task_id = argv.task_detail.task_id
        started_at = _wait_started_at.get(task_id)
        if started_at is None:
            started_at = time.monotonic()
            _wait_started_at[task_id] = started_at
            logger.warning(
                "unending_curtain_wait_timeout: wait timer was not initialized; "
                "starting it during recognition"
            )
            return None

        elapsed = time.monotonic() - started_at
        if elapsed < timeout_seconds:
            return None

        _wait_started_at.pop(task_id, None)
        _task_run_counts.pop(task_id, None)
        logger.warning(
            "unending_curtain_wait_timeout: "
            f"solo result wait timed out after {elapsed:.1f}s"
        )
        return CustomRecognition.AnalyzeResult(
            box=(0, 0, 1, 1),
            detail={
                "elapsed_seconds": elapsed,
                "timeout_seconds": timeout_seconds,
            },
        )

    @staticmethod
    def _as_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
