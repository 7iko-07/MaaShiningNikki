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
        popup_timeout = max(0, int(params.get("popup_timeout", 2500)))
        popup_interval = max(50, int(params.get("popup_interval", 250)))

        controller = context.tasker.controller
        image = self._screencap(controller)
        try:
            result = context.run_recognition(progress_node, image)
        except Exception as e:
            logger.error(f"union_secret_task_execute: 今日执行次数 OCR 异常: {e}")
            return False

        text = _collect_ocr_text(result)
        progress = parse_execution_progress(text)
        if not progress:
            logger.error(
                "union_secret_task_execute: 无法读取今日剩余执行次数，"
                f"OCR={text!r}"
            )
            return False

        remaining, total = progress
        if remaining < 1:
            logger.info(
                "union_secret_task_execute: "
                f"today remaining progress={remaining}/{total}, completed"
            )
            context.override_next(argv.node_name, [completed_next])
            return True

        button_node = execution_button_node_for_remaining(
            remaining,
            single_button_node=single_button_node,
            multiple_button_node=multiple_button_node,
        )
        button_result = self._run_recognition(context, image, button_node)
        target = self._execution_text_box(button_result)
        if not self._valid_target(target):
            logger.error(
                "union_secret_task_execute: 未在对应区域识别到“执行”按钮，"
                f"remaining={remaining}, node={button_node!r}"
            )
            return False

        logger.info(
            "union_secret_task_execute: "
            f"today remaining progress={remaining}/{total}, "
            f"button_node={button_node!r}, recognized_box={target}"
        )
        self._click_roi(controller, target)

        popup_found = self._wait_for_result(
            context,
            controller,
            purchase_popup_node=purchase_popup_node,
            reward_node=reward_node,
            timeout=popup_timeout,
            interval=popup_interval,
        )
        if not popup_found:
            return True

        logger.error(
            "union_secret_task_execute: 点击执行按钮后出现购买联盟委托次数弹窗，"
            "将取消购买并使当前任务失败"
        )
        try:
            cancel_result = context.run_task(cancel_purchase_node)
        except Exception as e:
            logger.error(f"union_secret_task_execute: 取消购买次数失败: {e}")
            return False

        if not self._task_succeeded(cancel_result):
            logger.error("union_secret_task_execute: 取消购买次数节点执行失败")
        return False

    def _wait_for_result(
        self,
        context,
        controller,
        purchase_popup_node,
        reward_node,
        timeout,
        interval,
    ):
        deadline = time.monotonic() + timeout / 1000.0
        while True:
            image = self._screencap(controller)
            if self._recognize(context, image, purchase_popup_node):
                return True
            if self._recognize(context, image, reward_node):
                return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(interval / 1000.0)

    def _recognize(self, context, image, node_name):
        result = self._run_recognition(context, image, node_name)
        return bool(result and getattr(result, "hit", False))

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
        controller.post_screencap().wait()
        return controller.cached_image.copy()

    def _click_roi(self, controller, roi):
        x, y, w, h = roi
        controller.post_click(x + w // 2, y + h // 2).wait()

    def _valid_target(self, target):
        return (
            isinstance(target, (list, tuple))
            and len(target) == 4
            and all(isinstance(value, (int, float)) for value in target)
            and target[2] > 0
            and target[3] > 0
        )

    def _task_succeeded(self, result):
        if result is None:
            return False
        success = getattr(result, "success", None)
        if success is not None:
            return bool(success)
        return bool(getattr(getattr(result, "status", None), "succeeded", False))


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
            "union_secret_task_complete: "
            f"today remaining progress={remaining}/{total}, completed"
        )
        return CustomRecognition.AnalyzeResult(
            box=(0, 1210, 235, 70),
            detail={
                "text": text,
                "remaining": remaining,
                "total": total,
            },
        )
