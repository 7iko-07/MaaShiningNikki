"""剧情自动点击：按开关选择是否全屏查找章节。"""

import json
from contextlib import contextmanager
from threading import Lock

from maa.agent.agent_server import AgentServer
from maa.controller import ControllerEventSink
from maa.custom_action import CustomAction
from maa.custom_recognition import CustomRecognition
from maa.event_sink import NotificationType
from utils import logger


DISPATCH_NODE = "剧情自动点击调度"
SEARCH_NODE = "剧情自动点击查找章节"
SPECIAL_NODE = "剧情自动点击特殊互动识别"
ERROR_NODE = "剧情自动点击控制器异常"


def controller_key(controller):
    # SDK 未提供公共控制器 ID；回调与 Context 的借用对象共享此原生句柄。
    # 实时视图可以与任务使用相同 uuid，不能仅按设备 uuid 归属异常。
    handle = getattr(controller, "_handle", None)
    return getattr(handle, "value", handle)


class StoryControllerFailures(ControllerEventSink):
    """按控制器及本次调用隔离，只记录剧情运行期间的真实控制器异常。"""

    def __init__(self):
        self._active = {}
        self._lock = Lock()

    @contextmanager
    def watch(self, key):
        marker, failures = object(), set()
        with self._lock:
            self._active.setdefault(key, {})[marker] = failures
        try:
            yield failures
        finally:
            with self._lock:
                self._active[key].pop(marker)
                if not self._active[key]:
                    del self._active[key]

    def on_controller_action(self, controller, noti_type, detail):
        if noti_type == NotificationType.Failed:
            with self._lock:
                for failures in self._active.get(controller_key(controller), {}).values():
                    failures.add(detail.action)

    def has_failures(self, key):
        with self._lock:
            return any(self._active.get(key, {}).values())


controller_failures = StoryControllerFailures()
AgentServer.add_controller_sink(controller_failures)


def is_manual_end(task):
    if task is None:
        return False
    if task.status.succeeded:
        return True
    if not task.status.failed:
        return False
    # 只接管 next 识别超时的空占位结果；动作失败不能当作正常结束。
    nodes = task.nodes
    if not nodes or nodes[-1] is None:
        return False
    last = nodes[-1]
    if (last.name or last.completed or last.action is not None
            or last.recognition is not None):
        return False
    nodes = nodes[:-1]
    # JumpBack 只回到调度节点的 next，不会重新执行并记录调度动作。
    # 因此对话点击后的正常超时，最后一个成功节点可能是“剧情自动点击推进对话”。
    return bool(nodes
                and all(node is not None and node.completed
                        and node.action is not None and node.action.success for node in nodes)
                and any(node.name == DISPATCH_NODE for node in nodes)
                and nodes[-1].name in (DISPATCH_NODE, "剧情自动点击推进对话"))


@AgentServer.custom_action("story_auto_click")
class StoryAutoClickAction(CustomAction):
    def run(self, context, argv):
        try:
            params = json.loads(argv.custom_action_param or "{}")
            search = params.get("search_chapters", False)
            if not isinstance(search, bool):
                raise ValueError("search_chapters 必须为布尔值")

            # 沿用原候选顺序、JumpBack、识别、点击及等待配置。
            # 特殊互动只停止，不进入原活动专用的推理操作。
            candidates = [
                ERROR_NODE,
                "剧情自动点击结束识别",
                SPECIAL_NODE,
                "[JumpBack]剧情自动点击推进对话",
            ]
            if search:
                candidates.append("[Anchor]剧情自动点击查找章节")

            # 原 click_region 会动态修改 next；在独立 Context 中运行，
            # 避免修改原活动任务使用的流程或留下本任务的选项覆盖。
            key = controller_key(context.tasker.controller)
            if not key:
                logger.debug("剧情点击异常：无法确认控制器身份")
                logger.error("剧情自动点击：无法连接设备，请检查连接")
                return False
            with controller_failures.watch(key) as failures:
                task = context.clone().run_task(
                    SEARCH_NODE if search else DISPATCH_NODE,
                    pipeline_override={
                        DISPATCH_NODE: {"next": candidates},
                        "剧情自动点击结束识别": {"focus": {"Node.Action.Starting": []}},
                    },
                )
                if failures:
                    logger.debug(f"剧情点击控制器异常：{', '.join(sorted(failures))}")
                    logger.error("剧情自动点击：设备操作失败，请检查连接")
                    return False
            if context.tasker.stopping:
                return False
            return is_manual_end(task)
        except Exception as exc:
            logger.debug(f"剧情自动点击失败：{exc}")
            logger.error("剧情自动点击执行异常，请检查当前页面")
            return False


@AgentServer.custom_action("story_auto_click_manual_stop")
class StoryAutoClickManualStopAction(CustomAction):
    def run(self, context, argv):
        logger.debug("剧情点击遇到特殊互动，交给用户手动操作")
        return True


@AgentServer.custom_action("story_auto_click_controller_guard")
class StoryAutoClickControllerGuardAction(CustomAction):
    def run(self, context, argv):
        # 仅由已识别的控制器异常候选进入，不开始下一轮剧情点击。
        return not controller_failures.has_failures(controller_key(context.tasker.controller))


@AgentServer.custom_recognition("story_auto_click_controller_failed")
class StoryAutoClickControllerFailedRecognition(CustomRecognition):
    def analyze(self, context, argv):
        if controller_failures.has_failures(controller_key(context.tasker.controller)):
            return self.AnalyzeResult(box=(0, 0, 1, 1), detail={})
        return None
