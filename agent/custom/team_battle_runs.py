import json
import re
import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.pipeline import JActionType, JClick, JOCR, JRecognitionType

from utils import debug_exception, extract_ocr_text, logger


@AgentServer.custom_action("team_battle_set_runs")
class TeamBattleSetRunsAction(CustomAction):
    # 三个结伴关卡共用的 720 × 1280 次数控件。
    SELECTED = ([560, 1070, 70, 45], r"^\s*(\d{1,2})\s*$")
    FIELDS = {
        "选中次数": SELECTED,
        "剩余次数": ([0, 1205, 220, 60], r"(\d+)\s*[/／]\s*\d+"),
        "体力": ([95, 0, 120, 40], r"(\d+)\s*[/／]\s*\d+"),
        "每次体力消耗": ([280, 1205, 190, 60], r"(\d+)"),
    }

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        task_name = "结伴任务"
        try:
            params = json.loads(argv.custom_action_param)
            task_name = params["task_name"]
            runs = params.get("runs", 0)
            if type(runs) is not int or not 0 <= runs <= 99:
                raise RuntimeError("刷取次数须为 0～99 的整数")

            # 每次进入均恢复后继，避免无次数分支影响下一次执行。
            self._set_next(context, argv.node_name, f"点击{task_name}快速组队")
            if runs == 0:
                return True

            deadline = time.monotonic() + 60
            values = self._read_numbers(context, self.FIELDS, deadline)
            current = values["选中次数"]
            if values["每次体力消耗"] <= 0:
                raise RuntimeError("无法确认每次体力消耗")
            target = min(runs, values["剩余次数"],
                         values["体力"] // values["每次体力消耗"])
            if target == 0:
                logger.info(f"{task_name}：次数或体力不足，本次刷取 0 次")
                self._set_next(context, argv.node_name, f"点击{task_name}返回")
                return True
            if not 1 <= current <= 99:
                raise RuntimeError("无法确认当前选中次数")

            while current != target:
                if context.tasker.stopping:
                    return False
                if time.monotonic() >= deadline:
                    raise RuntimeError("调整刷取次数超时")
                step = 1 if current < target else -1
                point = (652, 1092) if step > 0 else (536, 1092)
                result = context.run_action_direct(JActionType.Click, JClick(target=point))
                if not result or not result.success:
                    raise RuntimeError("点击次数加减按钮失败")
                for _ in range(3):
                    time.sleep(0.3)
                    updated = self._read_numbers(
                        context, {"选中次数": self.SELECTED}, deadline
                    )["选中次数"]
                    if updated != current:
                        break
                if updated != current + step:
                    raise RuntimeError("未能确认刷取次数调整结果")
                logger.debug(f"{task_name}：选中次数 {current} → {updated}")
                current = updated

            if target < runs:
                logger.info(f"{task_name}：设定 {runs} 次，当前可刷 {target} 次")
            else:
                logger.info(f"{task_name}：本次刷取 {target} 次")
            return True
        except RuntimeError as exc:
            debug_exception(f"{task_name}：刷取次数调整失败")
            logger.error(f"{task_name}：{exc}")
            return False
        except Exception:
            debug_exception(f"{task_name}：刷取次数调整异常")
            logger.error(f"{task_name}：调整刷取次数时发生异常")
            return False

    @staticmethod
    def _set_next(context, node_name, next_node):
        if not context.override_next(node_name, [next_node]):
            raise RuntimeError("未能继续任务")

    @staticmethod
    def _read_numbers(context, fields, deadline):
        controller = context.tasker.controller
        values = {}
        for attempt in range(3):
            if context.tasker.stopping:
                raise RuntimeError("任务已停止")
            if time.monotonic() >= deadline:
                raise RuntimeError("调整刷取次数超时")
            if not controller.post_screencap().wait().succeeded:
                raise RuntimeError("截图失败")
            image = controller.cached_image
            for name, (roi, pattern) in fields.items():
                result = context.run_recognition_direct(
                    JRecognitionType.OCR, JOCR(roi=roi, expected=[pattern]), image
                )
                text = extract_ocr_text(result)
                match = re.search(pattern, text)
                logger.debug(f"结伴刷取：{name} OCR={text!r}")
                if match:
                    values[name] = int(match.group(1))
                else:
                    values.pop(name, None)
            if len(values) == len(fields):
                return values
            if attempt < 2:
                time.sleep(0.3)
        missing = "、".join(name for name in fields if name not in values)
        raise RuntimeError(f"无法识别{missing}")
