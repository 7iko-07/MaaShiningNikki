import json
import re
import time

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JActionType, JRecognitionType, JOCR, JSwipe
from utils import logger


@AgentServer.custom_action("claim_visible_rewards")
class ClaimVisibleRewardsAction(CustomAction):
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}

        list_roi = params.get("list_roi", [40, 430, 640, 590])
        button_roi = params.get("button_roi", [455, 430, 220, 590])
        close_roi = params.get("close_roi", [315, 980, 90, 70])
        swipe_roi = params.get("swipe_roi", list_roi)
        expected = params.get("expected", ["^领取$", "领取"])
        exclude_pattern = params.get("exclude_pattern", "已领取|获得|获取")
        max_swipes = self._as_int(params.get("max_swipes"), 8)
        max_clicks_per_page = self._as_int(params.get("max_clicks_per_page"), 20)
        ocr_threshold = self._as_float(params.get("ocr_threshold"), 0.3)
        click_delay = self._as_float(params.get("click_delay"), 0.8)
        close_delay = self._as_float(params.get("close_delay"), 0.8)
        swipe_delay = self._as_float(params.get("swipe_delay"), 0.8)
        swipe_distance = self._as_int(params.get("swipe_distance"), 360)
        swipe_duration = self._as_int(params.get("swipe_duration"), 800)
        swipe_end_hold = self._as_int(params.get("swipe_end_hold"), 200)
        stop_on_unchanged = self._as_bool(params.get("stop_on_unchanged"), True)
        page_change_threshold = self._as_float(params.get("page_change_threshold"), 0.02)
        pixel_threshold = self._as_int(params.get("pixel_threshold"), 30)

        for name, roi in (
            ("list_roi", list_roi),
            ("button_roi", button_roi),
            ("close_roi", close_roi),
            ("swipe_roi", swipe_roi),
        ):
            if not self._valid_roi(roi):
                logger.error(f"claim_visible_rewards: invalid {name}")
                return False

        controller = context.tasker.controller
        claimed = 0

        for swipe_index in range(max_swipes + 1):
            page_claimed = self._claim_current_page(
                context,
                controller,
                button_roi,
                close_roi,
                expected,
                exclude_pattern,
                ocr_threshold,
                click_delay,
                close_delay,
                max_clicks_per_page,
            )
            claimed += page_claimed
            logger.info(
                "claim_visible_rewards: "
                f"page={swipe_index + 1}, page_claimed={page_claimed}, total_claimed={claimed}"
            )

            if swipe_index >= max_swipes:
                break

            before = self._screencap(controller).copy()
            self._swipe_up(context, swipe_roi, swipe_distance, swipe_duration, swipe_end_hold)
            time.sleep(swipe_delay)

            if stop_on_unchanged:
                after = self._screencap(controller)
                changed, ratio = self._image_changed(
                    before,
                    after,
                    list_roi,
                    pixel_threshold,
                    page_change_threshold,
                )
                logger.info(
                    "claim_visible_rewards: after swipe page changed="
                    f"{changed}, ratio={ratio:.6f}"
                )
                if not changed:
                    break

        logger.info(f"claim_visible_rewards: finished, claimed={claimed}")
        return True

    def _claim_current_page(
        self,
        context,
        controller,
        button_roi,
        close_roi,
        expected,
        exclude_pattern,
        threshold,
        click_delay,
        close_delay,
        max_clicks,
    ):
        claimed = 0
        clicked_boxes = []

        for _ in range(max_clicks):
            image = self._screencap(controller)
            boxes = self._find_claim_boxes(
                context,
                image,
                button_roi,
                expected,
                exclude_pattern,
                threshold,
                clicked_boxes,
            )
            if not boxes:
                break

            box = boxes[0]
            clicked_boxes.append(box)
            logger.info(f"claim_visible_rewards: clicking claim button box={box}")
            self._click_box(controller, box)
            claimed += 1
            time.sleep(click_delay)
            self._click_roi(controller, close_roi)
            time.sleep(close_delay)

        return claimed

    def _find_claim_boxes(
        self,
        context,
        image,
        roi,
        expected,
        exclude_pattern,
        threshold,
        clicked_boxes,
    ):
        result = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(roi=roi, expected=expected, threshold=threshold, order_by="Vertical"),
            image,
        )

        boxes = []
        for item in self._iter_ocr_results(result):
            text = self._normalize_text(getattr(item, "text", ""))
            box = getattr(item, "box", None)
            if not text or not box:
                continue
            if exclude_pattern and re.search(exclude_pattern, text):
                continue
            if "领取" not in text or "已领取" in text:
                continue

            box = list(box)
            if self._is_duplicate_box(box, boxes) or self._is_duplicate_box(box, clicked_boxes):
                continue
            boxes.append(box)

        boxes.sort(key=lambda item: (item[1], item[0]))
        return boxes

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
            raise RuntimeError("claim_visible_rewards: swipe action failed")

    def _screencap(self, controller):
        controller.post_screencap().wait()
        return controller.cached_image

    def _click_roi(self, controller, roi):
        x, y, w, h = roi
        controller.post_click(x + w // 2, y + h // 2).wait()

    def _click_box(self, controller, box):
        x, y, w, h = box
        controller.post_click(x + w // 2, y + h // 2).wait()

    def _image_changed(self, before, after, roi, pixel_threshold, change_threshold):
        x, y, w, h = roi
        before = before[y:y + h, x:x + w]
        after = after[y:y + h, x:x + w]
        if before.size == 0 or after.size == 0:
            return False, 0.0

        diff = np.abs(after.astype(np.int16) - before.astype(np.int16))
        changed_pixels = np.any(diff > pixel_threshold, axis=2)
        ratio = np.count_nonzero(changed_pixels) / changed_pixels.size
        return ratio >= change_threshold, ratio

    def _is_duplicate_box(self, box, boxes):
        cx, cy = self._center(box)
        for other in boxes:
            ox, oy = self._center(other)
            if abs(cx - ox) <= 20 and abs(cy - oy) <= 20:
                return True
        return False

    def _center(self, box):
        x, y, w, h = box
        return x + w / 2, y + h / 2

    def _normalize_text(self, text):
        return re.sub(r"\s+", "", str(text or ""))

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

    def _as_bool(self, value, default):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() not in ("0", "false", "no", "off")
        return bool(value)
