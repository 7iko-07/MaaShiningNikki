import json
import re
import time
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils import logger


@AgentServer.custom_action("arena_compare")
class ArenaCompareAction(CustomAction):
    DEFAULT_BLANK_ROI = [538, 133, 33, 19]

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        area_a = params.get("area_a")
        area_b_list = params.get("area_b", [])
        refresh_roi = params.get("refresh")
        remaining_count_roi = self._get_first_param(
            params,
            "remaining_count_roi",
            "remaining_roi",
            "剩余次数",
        )
        blank_roi = self._get_first_param(params, "blank_roi", "空白区域") or self.DEFAULT_BLANK_ROI

        max_rounds = self._as_int(params.get("max_rounds"), 5)
        blank_clicks = self._as_int(params.get("blank_clicks"), 10)
        blank_click_delay = self._as_float(params.get("blank_click_delay"), 1.0)

        if not area_a or not area_b_list or not refresh_roi:
            logger.error("arena_compare: missing required params (area_a, area_b, refresh)")
            return False
        if not self._valid_roi(area_a):
            logger.error("arena_compare: invalid area_a")
            return False
        if not self._valid_roi(refresh_roi):
            logger.error("arena_compare: invalid refresh")
            return False
        if not self._valid_roi_list(area_b_list):
            logger.error("arena_compare: invalid area_b")
            return False
        if remaining_count_roi and not self._valid_roi(remaining_count_roi):
            logger.error("arena_compare: invalid remaining_count_roi")
            return False
        if not self._valid_roi(blank_roi):
            logger.error("arena_compare: invalid blank_roi")
            return False

        controller = context.tasker.controller
        challenge_times = self._read_challenge_times(context, controller, remaining_count_roi)
        if challenge_times <= 0:
            logger.info(f"arena_compare: no remaining challenge times, challenge_times={challenge_times}")
            context.override_next(argv.node_name, [])
            return True

        for challenge_index in range(challenge_times):
            logger.info(f"arena_compare: challenge {challenge_index + 1}/{challenge_times}")
            found = self._challenge_once(context, controller, area_a, area_b_list, refresh_roi, max_rounds)
            if not found:
                context.override_next(argv.node_name, [])
                return True

            if challenge_index < challenge_times - 1:
                self._click_roi_repeated(controller, blank_roi, blank_clicks, blank_click_delay)

        return True

    def _read_challenge_times(self, context, controller, remaining_count_roi):
        if not remaining_count_roi:
            return 1

        job = controller.post_screencap()
        job.wait()
        remaining = self._ocr_remaining_count(context, controller.cached_image, remaining_count_roi)
        if remaining is None:
            logger.warning("arena_compare: failed to OCR remaining count, falling back to one challenge")
            return 1

        logger.info(f"arena_compare: remaining challenge times = {remaining}")
        return remaining

    def _challenge_once(self, context, controller, area_a, area_b_list, refresh_roi, max_rounds):
        for round_num in range(max_rounds):
            logger.info(f"arena_compare: refresh round {round_num + 1}/{max_rounds}")

            job = controller.post_screencap()
            job.wait()
            img = controller.cached_image

            player_power = self._ocr_number(context, img, area_a)
            if player_power is None:
                logger.warning("arena_compare: failed to OCR player power, retrying")
                time.sleep(0.5)
                continue

            logger.info(f"arena_compare: player power = {player_power}")

            for i, area_b in enumerate(area_b_list):
                opp_power = self._ocr_number(context, img, area_b)
                if opp_power is None:
                    logger.warning(f"arena_compare: failed to OCR opponent {i}")
                    continue
                logger.info(f"arena_compare: opponent {i} power = {opp_power}")
                if player_power > opp_power:
                    logger.info(f"arena_compare: player > opponent {i}, clicking opponent {i}")
                    self._click_opponent(controller, area_b)
                    return True

            if round_num < max_rounds - 1:
                logger.info("arena_compare: all opponents stronger, clicking refresh")
                self._click_roi(controller, refresh_roi)
                time.sleep(1.5)
            else:
                logger.warning(f"arena_compare: no weaker opponent after {max_rounds} rounds")

        return False

    def _ocr_remaining_count(self, context: Context, img, roi):
        try:
            result = context.run_recognition(
                "_arena_ocr_remainging_count",
                img,
                pipeline_override={
                    "_arena_ocr_remainging_count": {
                        "recognition": "OCR",
                        "roi": roi,
                        "expected": [r"\d+\s*/\s*\d+"],
                    }
                },
            )
            text = self._extract_ocr_text(result)
            if text:
                text = text.strip().replace(",", "").replace(" ", "")
                num_str = text.split("/", 1)[0]
                if num_str:
                    return int(num_str)
        except Exception as e:
            logger.warning(f"arena_compare: OCR error: {e}")
        return None

    def _ocr_number(self, context: Context, img, roi):
        try:
            result = context.run_recognition(
                "_arena_ocr",
                img,
                pipeline_override={
                    "_arena_ocr": {
                        "recognition": "OCR",
                        "roi": roi,
                        "expected": [r"\d[\d,]*"],
                    }
                },
            )
            text = self._extract_ocr_text(result)
            if text:
                text = text.strip().replace(",", "").replace(" ", "")
                num_str = re.sub(r"[^\d]", "", text)
                if num_str:
                    return int(num_str)
        except Exception as e:
            logger.warning(f"arena_compare: OCR error: {e}")
        return None

    def _click_opponent(self, controller, roi):
        x, y, w, h = roi
        controller.post_click(x + w // 2 - 60, y + h // 2 - 100).wait()

    def _click_roi(self, controller, roi):
        x, y, w, h = roi
        controller.post_click(x + w // 2, y + h // 2).wait()

    def _click_roi_repeated(self, controller, roi, count, delay):
        for _ in range(max(0, count)):
            self._click_roi(controller, roi)
            if delay > 0:
                time.sleep(delay)

    def _valid_roi(self, value):
        return isinstance(value, list) and len(value) == 4

    def _valid_roi_list(self, value):
        return (
            isinstance(value, list)
            and value
            and all(self._valid_roi(roi) for roi in value)
        )

    def _get_first_param(self, params, *names):
        for name in names:
            value = params.get(name)
            if value is not None:
                return value
        return None

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

    def _extract_ocr_text(self, result):
        if not result or not getattr(result, "hit", False):
            return ""

        for candidate in self._iter_recognition_results(result):
            text = getattr(candidate, "text", None)
            if text:
                return str(text)

            detail = getattr(candidate, "detail", None)
            if detail:
                return str(detail)

        return self._extract_text_from_raw_detail(getattr(result, "raw_detail", None))

    def _iter_recognition_results(self, result):
        best_result = getattr(result, "best_result", None)
        if best_result is not None:
            yield best_result

        for attr in ("filtered_results", "all_results"):
            for item in getattr(result, attr, []) or []:
                if item is not None:
                    yield item

    def _extract_text_from_raw_detail(self, raw_detail):
        if isinstance(raw_detail, dict):
            for key in ("text", "detail"):
                value = raw_detail.get(key)
                if value:
                    return str(value)

            for key in ("best", "filtered", "all"):
                value = raw_detail.get(key)
                text = self._extract_text_from_raw_detail(value)
                if text:
                    return text

        if isinstance(raw_detail, list):
            for item in raw_detail:
                text = self._extract_text_from_raw_detail(item)
                if text:
                    return text

        return ""
