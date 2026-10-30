"""Log elapsed time for complete Tasker tasks, including failed tasks."""

import json
from pathlib import Path
from threading import Lock
import time

from maa.event_sink import NotificationType
from maa.tasker import TaskerEventSink
from utils import logger


class TaskTimingSink(TaskerEventSink):
    def __init__(self, root):
        self._started = {}
        self._lock = Lock()
        self._task_names = {}
        root = Path(root)
        # Release packages put interface.json beside agent/; source uses assets/.
        for path in (root / "interface.json", root / "assets/interface.json"):
            if not path.is_file():
                continue
            try:
                interface = json.loads(path.read_text(encoding="utf-8"))
                self._task_names = {
                    task["entry"]: task["name"] for task in interface.get("task", [])
                    if task.get("entry") and task.get("name")
                }
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                logger.debug(f"任务耗时日志：读取任务名称失败，将使用入口名称：{exc}")
                logger.warning("未能读取任务名称，本次使用任务入口名称显示耗时")
            break

    def on_tasker_task(self, tasker, noti_type, detail):
        key = (detail.uuid, detail.task_id)
        with self._lock:
            if noti_type == NotificationType.Starting:
                self._started.setdefault(key, time.monotonic())
                return
            if noti_type not in (NotificationType.Succeeded, NotificationType.Failed):
                return
            started = self._started.pop(key, None)
            elapsed = time.monotonic() - started if started is not None else None

        # Ignore unmatched/duplicate terminal events rather than invent a duration.
        if elapsed is None:
            logger.debug(f"任务耗时日志：未收到开始事件，跳过 task_id={detail.task_id}")
            return
        name = self._task_names.get(detail.entry, detail.entry)
        if noti_type == NotificationType.Succeeded:
            result = "结束" if detail.entry == "剧情自动点击任务链入口" else "完成"
            logger.info(f"任务「{name}」{result}，耗时 {elapsed:.2f} 秒")
        else:
            logger.error(f"任务「{name}」失败，耗时 {elapsed:.2f} 秒")
