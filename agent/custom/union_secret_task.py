import json
import re
import time
import unicodedata

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.custom_recognition import CustomRecognition
from maa.context import Context

from utils import logger
from utils.ocr import extract_text_from_raw_detail, iter_recognition_results


PROGRESS_PATTERN = re.compile(r"(?<!\d)(\d+)\s*[/／]\s*(\d+)(?!\d)")
PROGRESS_OCR_NODE = "机密任务今日执行次数OCR"
DEFAULT_SINGLE_BUTTON_NODE = "机密任务执行一次按钮OCR"
DEFAULT_MULTIPLE_BUTTON_NODE = "机密任务执行多次按钮OCR"


def parse_execution_progress(text):
    """Return (remaining, total) from text such as '今日执行次数：1/5'."""
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    match = PROGRESS_PATTERN.search(normalized)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def is_execution_complete(text):
    progress = parse_execution_progress(text)
    return bool(progress and progress[0] < 1)


def execution_button_node_for_remaining(
    remaining,
    single_button_node=DEFAULT_SINGLE_BUTTON_NODE,
    multiple_button_node=DEFAULT_MULTIPLE_BUTTON_NODE,
):
    if remaining < 1:
        return None
    if remaining == 1:
        return single_button_node
    return multiple_button_node


def _collect_ocr_text(result):
    texts = []
    for candidate in iter_recognition_results(result):
        text = getattr(candidate, "text", None)
        if text:
            texts.append(str(text))

    raw_text = extract_text_from_raw_detail(getattr(result, "raw_detail", None))
    if raw_text:
        texts.append(raw_text)
    return " ".join(texts)


@AgentServer.custom_action("union_secret_task_execute")
class UnionSecretTaskExecuteAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            return self._execute(context, argv)
        except Exception as e:
            logger.error(f"union_secret_task_execute: 执行失败: {e}")
            return False

    def _execute(self, context, argv):
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        progress_node = params.get("progress_node", PROGRESS_OCR_NODE)
        completed_next = params.get("completed_next", "点击返回联盟")
        single_button_node = params.get(
            "single_button_node",
            DEFAULT_SINGLE_BUTTON_NODE,
        )
        multiple_button_node = params.get(
            "multiple_button_node",
            DEFAULT_MULTIPLE_BUTTON_NODE,
        )
        purchase_popup_node = params.get(
            "purchase_popup_node",
            "机密任务购买次数弹窗",
        )
        cancel_purchase_node = params.get(
            "cancel_purchase_node",
            "取消购买机密任务次数",
        )
        reward_node = params.get("reward_node", "联盟执行多次奖励")
        weekly_node = params.get("weekly_popup_node", "关闭机密任务周奖励弹窗")
        close_reward_node = params.get("close_reward_node", "机密任务关闭奖励弹窗")
        deadline = time.monotonic() + max(0, int(params.get("timeout", 30000))) / 1000
        controller = context.tasker.controller
        stable_progress, stable_frames = None, 0
        clicked_remaining = None
        reward_seen = False
        next_click = 0.0
        first_poll = True

        def active():
            return not context.tasker.stopping and time.monotonic() < deadline

        while active():
            if not first_poll:
                time.sleep(min(0.25, max(0, deadline - time.monotonic())))
            first_poll = False
            if not active():
                break
            image = self._screencap(controller)
            if image is None or not active():
                return False
            # 弹窗优先于背景中的次数和按钮；仅执行关闭动作，不进入其 next 子链。
            if self._recognize(context, image, purchase_popup_node):
                if not active():
                    break
                logger.error("union_secret_task_execute: 出现购买次数弹窗，取消购买并终止任务")
                result = context.run_action(cancel_purchase_node)
                if not result or not result.success:
                    logger.error("union_secret_task_execute: 取消购买次数失败")
                return False
            weekly = self._recognize(context, image, weekly_node)
            reward = not weekly and self._recognize(context, image, reward_node)
            if weekly or reward:
                stable_progress, stable_frames = None, 0
                if reward:
                    reward_seen = True
                if active() and time.monotonic() >= next_click:
                    result = context.run_action(close_reward_node)
                    if not result or not result.success:
                        logger.error("union_secret_task_execute: 关闭奖励弹窗失败")
                        return False
                    next_click = time.monotonic() + 1
                continue

            result = self._run_recognition(context, image, progress_node)
            progress = (parse_execution_progress(_collect_ocr_text(result))
                        if result and result.hit else None)
            if not progress:
                stable_progress, stable_frames = None, 0
                continue
            remaining, total = progress
            target = None
            if remaining > 0:
                button_node = execution_button_node_for_remaining(
                    remaining, single_button_node, multiple_button_node,
                )
                button = self._run_recognition(context, image, button_node)
                target = self._execution_text_box(button)
                if not self._valid_target(target):
                    stable_progress, stable_frames = None, 0
                    continue
            if progress == stable_progress:
                stable_frames += 1
            else:
                stable_progress, stable_frames = progress, 1
            if stable_frames < 2 or not active():
                continue
            if remaining == 0:
                if not context.override_next(argv.node_name, [completed_next]):
                    logger.error("union_secret_task_execute: 完成后继设置失败")
                    return False
                logger.info(f"联盟机密任务: 今日剩余次数={remaining}/{total}，执行完成")
                return True
            if clicked_remaining is not None:
                if remaining > clicked_remaining:
                    logger.error("union_secret_task_execute: 剩余次数异常增加，停止执行")
                    return False
                if remaining < clicked_remaining:
                    logger.debug(f"union_secret_task_execute: 剩余次数 {clicked_remaining} -> {remaining}")
                    clicked_remaining = None
                    reward_seen = False
                elif reward_seen:
                    # 已确认执行奖励，次数尚未刷新时不能重复执行。
                    continue
            if active() and time.monotonic() >= next_click:
                logger.debug(f"union_secret_task_execute: remaining={remaining}/{total}, "
                             f"button_node={button_node!r}, recognized_box={target}")
                if not self._click_roi(controller, target):
                    logger.error("union_secret_task_execute: 点击执行失败")
                    return False
                clicked_remaining = remaining
                stable_progress, stable_frames = None, 0
                next_click = time.monotonic() + 1

        logger.error("union_secret_task_execute: 任务取消或等待执行结果超时，未确认完成")
        return False

    def _recognize(self, context, image, node_name):
        result = self._run_recognition(context, image, node_name)
        if result is None:
            raise RuntimeError(f"无法检查弹窗：{node_name}")
        return bool(result.hit)

    def _run_recognition(self, context, image, node_name):
        try:
            return context.run_recognition(node_name, image)
        except Exception as e:
            logger.warning(
                f"union_secret_task_execute: 识别 {node_name!r} 失败: {e}"
            )
            return None

    def _execution_text_box(self, result):
        if not result or not getattr(result, "hit", False):
            return None

        for candidate in iter_recognition_results(result):
            text = unicodedata.normalize(
                "NFKC",
                str(getattr(candidate, "text", "") or ""),
            ).replace(" ", "")
            if "执行" not in text:
                continue

            box = getattr(candidate, "box", None)
            if isinstance(box, (list, tuple)) and len(box) == 4:
                return list(box)
            if box is not None and all(
                hasattr(box, field) for field in ("x", "y", "w", "h")
            ):
                return [box.x, box.y, box.w, box.h]
        return None

    def _screencap(self, controller):
        capture = controller.post_screencap().wait()
        if not capture.succeeded or controller.cached_image is None:
            logger.error("union_secret_task_execute: 截图失败")
            return None
        return controller.cached_image.copy()

    def _click_roi(self, controller, roi):
        x, y, w, h = roi
        return controller.post_click(x + w // 2, y + h // 2).wait().succeeded

    def _valid_target(self, target):
        return (
            isinstance(target, (list, tuple))
            and len(target) == 4
            and all(isinstance(value, (int, float)) for value in target)
            and target[2] > 0
            and target[3] > 0
        )


@AgentServer.custom_recognition("union_secret_task_ready")
class UnionSecretTaskReadyRecognition(CustomRecognition):
    """Only enter the action when both the count and its button are visible.

    A miss lets the parent retry its entire candidate list within its timeout,
    including popups and the user-selected manual execution branch.
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult | None:
        helper = UnionSecretTaskExecuteAction()
        result = helper._run_recognition(context, argv.image, PROGRESS_OCR_NODE)
        if not result or not getattr(result, "hit", False):
            return None
        progress = parse_execution_progress(_collect_ocr_text(result))
        if not progress or progress[0] < 1:
            return None

        button_node = execution_button_node_for_remaining(progress[0])
        button = helper._run_recognition(context, argv.image, button_node)
        target = helper._execution_text_box(button)
        if not helper._valid_target(target):
            return None
        return CustomRecognition.AnalyzeResult(
            box=tuple(target),
            detail={"remaining": progress[0], "total": progress[1]},
        )


@AgentServer.custom_recognition("union_secret_task_complete")
class UnionSecretTaskCompleteRecognition(CustomRecognition):
    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult | None:
        try:
            result = context.run_recognition(PROGRESS_OCR_NODE, argv.image)
        except Exception as e:
            logger.warning(f"union_secret_task_complete: OCR error: {e}")
            return None

        text = _collect_ocr_text(result)
        progress = parse_execution_progress(text)
        if not progress:
            return None

        remaining, total = progress
        if remaining >= 1:
            return None

        logger.info(
            "联盟机密任务: "
            f"今日剩余次数={remaining}/{total}"
        )
        return CustomRecognition.AnalyzeResult(
            box=(0, 1210, 235, 70),
            detail={
                "text": text,
                "remaining": remaining,
                "total": total,
            },
        )
