import json
import re
import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JActionType, JRecognitionType, JOCR, JSwipe
from utils import logger


@AgentServer.custom_action("select_same_text_after_thumb")
class SelectSameTextAfterThumbAction(CustomAction):

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}

        source_roi = params.get("source_roi", [433, 215, 47, 24])
        thumb_roi = params.get("thumb_roi", [568, 739, 26, 26])
        search_button_roi = params.get("search_button_roi", [30, 721, 32, 27])
        search_roi = params.get("search_roi", [60, 354, 603, 381])
        confirm_roi = params.get("confirm_roi", [493, 943, 74, 35])

        max_swipes = self._as_int(params.get("max_swipes"), 4)
        swipe_distance = self._as_int(params.get("swipe_distance"), 200)
        swipe_duration = self._as_int(params.get("swipe_duration"), 900)
        swipe_end_hold = self._as_int(params.get("swipe_end_hold"), 200)
        ocr_threshold = self._as_float(params.get("ocr_threshold"), 0.3)
        click_delay = self._as_float(params.get("click_delay"), 2.0)
        search_delay = self._as_float(params.get("search_delay"), 0.8)
        swipe_delay = self._as_float(params.get("swipe_delay"), 0.8)

        for name, roi in (
            ("source_roi", source_roi),
            ("thumb_roi", thumb_roi),
            ("search_button_roi", search_button_roi),
            ("search_roi", search_roi),
            ("confirm_roi", confirm_roi),
        ):
            if not self._valid_roi(roi):
                logger.error(f"select_same_text_after_thumb: invalid {name}")
                return False

        controller = context.tasker.controller

        target_text = self._read_first_text(context, controller, source_roi, ocr_threshold)
        if not target_text:
            logger.error("select_same_text_after_thumb: failed to OCR source text")
            return False

        logger.info(f"select_same_text_after_thumb: target_text={target_text!r}")

        self._click_roi(controller, thumb_roi)
        time.sleep(click_delay)
        self._click_roi(controller, search_button_roi)
        time.sleep(search_delay)

        for attempt in range(max_swipes + 1):
            match = self._find_matching_text(context, controller, search_roi, target_text, ocr_threshold)
            if match:
                text, box = match
                logger.info(f"select_same_text_after_thumb: matched text={text!r}, box={box}")
                self._click_box(controller, box)
                time.sleep(click_delay)
                self._click_roi(controller, confirm_roi)
                return True

            if attempt >= max_swipes:
                break

            logger.info(
                "select_same_text_after_thumb: no match, swiping up "
                f"{attempt + 1}/{max_swipes}"
            )
            self._swipe_up(context, search_roi, swipe_distance, swipe_duration, swipe_end_hold)
            time.sleep(swipe_delay)

        logger.warning(f"select_same_text_after_thumb: no match for {target_text!r}")
        return False

    def _read_first_text(self, context, controller, roi, threshold):
        controller.post_screencap().wait()
        result = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(roi=roi, threshold=threshold),
            controller.cached_image,
        )
        for item in self._iter_ocr_results(result):
            text = self._normalize_text(getattr(item, "text", ""))
            if text:
                return text
        return ""

    def _find_matching_text(self, context, controller, roi, target_text, threshold):
        controller.post_screencap().wait()
        result = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(roi=roi, threshold=threshold, order_by="Vertical"),
            controller.cached_image,
        )

        for item in self._iter_ocr_results(result):
            text = self._normalize_text(getattr(item, "text", ""))
            box = getattr(item, "box", None)
            if text and box and self._text_matches(target_text, text):
                return text, list(box)

        return None

    def _iter_ocr_results(self, result):
        if not result or not getattr(result, "hit", False):
            return

        yielded = set()
        for attr in ("filtered_results", "all_results"):
            for item in getattr(result, attr, []) or []:
                key = id(item)
                if key in yielded:
                    continue
                yielded.add(key)
                if item is not None:
                    yield item

        best_result = getattr(result, "best_result", None)
        if best_result is not None and id(best_result) not in yielded:
            yield best_result

    def _text_matches(self, target_text, candidate_text):
        if target_text == candidate_text:
            return True
        return target_text in candidate_text or candidate_text in target_text

    def _normalize_text(self, text):
        text = str(text or "")
        return re.sub(r"\s+", "", text)

    def _click_roi(self, controller, roi):
        x, y, w, h = roi
        controller.post_click(x + w // 2, y + h // 2).wait()

    def _click_box(self, controller, box):
        x, y, w, h = box
        controller.post_click(x + w // 2, y + h // 2).wait()

    def _swipe_up(self, context, roi, distance, duration, end_hold):
        x, y, w, h = roi
        begin = [x + w // 2, y + h - 40]
        end = [begin[0], max(y + 20, begin[1] - distance)]
        result = context.run_action_direct(
            JActionType.Swipe,
            JSwipe(
                begin=begin,
                end=[end],
                duration=[duration],
                end_hold=[end_hold],
            ),
        )
        if not result or not result.success:
            raise RuntimeError("Swipe action failed")

    def _valid_roi(self, value):
        return isinstance(value, list) and len(value) == 4

    def _as_int(self, value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _as_float(self, value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


@AgentServer.custom_action("click_nail_tabs_sequence")
class ClickNailTabsSequenceAction(CustomAction):

    DEFAULT_ROIS = [
        [53, 992, 75, 47],
        [63, 1130, 61, 53],
        [232, 999, 74, 36],
        [63, 1140, 57, 50],
        [404, 1001, 86, 32],
        [63, 1140, 57, 50],
        [600, 1000, 71, 36],
        [63, 1140, 57, 50],
        [58, 149, 58, 23],
        [678, 953, 17, 12]
    ]

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        rois = params.get("rois", self.DEFAULT_ROIS)
        delay = self._as_float(params.get("delay"), 0.4)

        if not isinstance(rois, list) or not rois:
            logger.error("click_nail_tabs_sequence: invalid rois")
            return False

        for index, roi in enumerate(rois):
            if not self._valid_roi(roi):
                logger.error(f"click_nail_tabs_sequence: invalid roi at index {index}")
                return False

        controller = context.tasker.controller
        for roi in rois:
            self._click_roi(controller, roi)
            if delay > 0:
                time.sleep(delay)

        return True

    def _click_roi(self, controller, roi):
        x, y, w, h = roi
        controller.post_click(x + w // 2, y + h // 2).wait()

    def _valid_roi(self, value):
        return isinstance(value, list) and len(value) == 4

    def _as_float(self, value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
