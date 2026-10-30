"""竞技场结算期拦截、搭配分流和完成确认。"""

import json
import sqlite3

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from utils import logger
from utils import arena_state as state


@AgentServer.custom_action("start_account_session")
class StartAccountSession(CustomAction):
    def run(self, context, argv):
        params = json.loads(argv.custom_action_param or "{}")
        state.begin_login(context, argv.task_detail.task_id, params.get("multi", False))
        return True


@AgentServer.custom_action("arena_weekly")
class ArenaWeekly(CustomAction):
    def __init__(self):
        super().__init__()
        self._runs = {}

    def run(self, context, argv):
        params = json.loads(argv.custom_action_param or "{}")
        operation = params.get("operation", "enter")
        device = context.tasker.controller.uuid
        now = state.now_beijing()

        if operation == "enter":
            # 即使本次在结算期，也消费 start 身份，防止下一批任务误用。
            account = state.take_account(context)
            self._runs[device] = {
                "task_id": argv.task_detail.task_id,
                "account": account,
                "week": state.week_key(now),
                "dressing": False,
            }
            next_nodes = ["竞技场任务链准备"]
        elif operation == "choose":
            run = self._runs.get(device)
            if not run or run["task_id"] != argv.task_detail.task_id:
                logger.debug("竞技场缺少本次任务状态，请从任务入口运行")
                logger.error("竞技场：任务状态缺失，请从任务列表重新运行")
                return False
            mode = params.get("mode", "never")
            if mode not in ("never", "always", "weekly"):
                logger.debug(f"未知竞技场搭配模式：{mode}")
                logger.error("竞技场：搭配模式配置无效")
                return False
            dress = mode != "never"
            if mode == "weekly" and run["account"]:
                try:
                    dress = not state.StateStore().completed(run["account"], run["week"])
                except (OSError, sqlite3.Error) as exc:
                    logger.debug(f"竞技场完成记录读取失败，本次重新搭配：{exc}")
                    logger.warning("竞技场：无法读取本周搭配记录，本次重新搭配")
            if mode != "never" and not run["account"]:
                logger.debug("账号身份未确认，本次搭配但不读写周记录；请先运行 start")
                logger.warning("竞技场：账号未确认，本次照常搭配；请先运行启动任务以使用每周记录")
            run["dressing"] = dress
            if dress:
                next_nodes = ["点击前往搭配", "我的搭配"]
            else:
                logger.info("竞技场本次跳过搭配，继续挑战")
                next_nodes = ["[JumpBack]点击OK-竞技场", "竞技场挑战策略分流"]
        elif operation == "complete":
            run = self._runs.get(device)
            if not run or run["task_id"] != argv.task_detail.task_id or not run["dressing"]:
                logger.debug("竞技场没有待确认的搭配，拒绝写入完成记录")
                logger.error("竞技场：搭配尚未确认完成，未保存本周记录")
                return False
            # 此节点只在完整搭配链返回、OCR 确认挑战页面之后执行。
            if run["account"] and run["week"] == state.week_key(now) and not state.in_settlement(now):
                try:
                    state.StateStore().mark_completed(run["account"], run["week"], now)
                    logger.info("竞技场本周搭配已完成并保存")
                except (OSError, sqlite3.Error) as exc:
                    logger.debug(f"竞技场完成记录保存失败，下次将重新搭配：{exc}")
                    logger.warning("竞技场：本周搭配记录保存失败，下次将重新搭配")
            run["dressing"] = False
            next_nodes = ["竞技场挑战策略分流"]
        else:
            logger.debug(f"未知竞技场操作：{operation}")
            logger.error("竞技场：操作配置无效")
            return False

        if state.in_settlement(now):
            logger.info("竞技场结算中（北京时间周六21:00至周日12:00），跳过任务")
            next_nodes = []
        return context.override_next(argv.node_name, next_nodes)
