import json
import re
import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils import logger


@AgentServer.custom_action("information_house_auto_investigate")
class InformationHouseAutoInvestigateAction(CustomAction):
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}

        stamina_roi = params.get("stamina_roi", [73, 64, 146, 40])
        add_button_roi = params.get("add_button_roi", [581, 604, 33, 33])
        cost_roi = params.get("cost_roi", [496, 838, 92, 35])
        minus_button_roi = params.get("minus_button_roi", params.get("confirm_button_roi", [350, 606, 35, 28]))
        start_button_roi = params.get("start_button_roi", [418, 777, 125, 40])

        cost_map = self._build_cost_map(params.get("cost_map"))
        max_adjust = self._as_int(params.get("max_adjust"), 8)
        retry = self._as_int(params.get("retry"), 3)
        retry_delay = self._as_float(params.get("retry_delay"), 0.5)
        click_delay = self._as_float(params.get("click_delay"), 0.8)

        if not all(self._valid_roi(roi) for roi in (stamina_roi, add_button_roi, cost_roi, minus_button_roi, start_button_roi)):
            logger.error("information_house_auto_investigate: invalid roi param")
            return False

        controller = context.tasker.controller
        stamina = self._read_number_before_slash(context, controller, stamina_roi, retry, retry_delay)
        if stamina is None:
            logger.error("information_house_auto_investigate: failed to read stamina")
            return False

        logger.info(f"information_house_auto_investigate: stamina={stamina}")
        target_clicks = self._target_clicks_for_stamina(stamina, cost_map)
        if target_clicks is None:
            logger.info(
                "information_house_auto_investigate: stamina is not enough for any cost, "
                f"stamina={stamina}, min_cost={min(cost_map.values())}"
            )
            return True

        target_cost = cost_map[target_clicks]
        logger.info(
            "information_house_auto_investigate: "
            f"target_clicks={target_clicks}, target_cost={target_cost}"
        )

        for _ in range(target_clicks):
            self._click_roi(controller, add_button_roi)
            time.sleep(click_delay)

        if not self._adjust_to_target_cost(
            context,
            controller,
            cost_roi=cost_roi,
            add_button_roi=add_button_roi,
            minus_button_roi=minus_button_roi,
            target_cost=target_cost,
            retry=retry,
            retry_delay=retry_delay,
            click_delay=click_delay,
            max_adjust=max_adjust,
        ):
            return False

        self._click_roi(controller, start_button_roi)
        return True

    def _build_cost_map(self, cost_map):
        default_cost_map = {
            0: 20,
            1: 32,
            2: 44,
            3: 59,
            4: 74,
            5: 92,
            6: 110,
            7: 135,
            8: 160,
        }
        if not isinstance(cost_map, dict):
            return default_cost_map

        normalized = {}
        for clicks, cost in cost_map.items():
            try:
                normalized[int(clicks)] = int(cost)
            except (TypeError, ValueError):
                logger.warning(
                    "information_house_auto_investigate: invalid cost_map item, "
                    f"clicks={clicks!r}, cost={cost!r}"
                )

        return normalized or default_cost_map

    def _target_clicks_for_stamina(self, stamina, cost_map):
        candidates = [clicks for clicks, cost in cost_map.items() if cost <= stamina]
        if not candidates:
            return None
        return max(candidates)

    def _adjust_to_target_cost(
        self,
        context,
        controller,
        cost_roi,
        add_button_roi,
        minus_button_roi,
        target_cost,
        retry,
        retry_delay,
        click_delay,
        max_adjust,
    ):
        for attempt in range(max(0, max_adjust) + 1):
            cost = self._read_number(context, controller, cost_roi, retry, retry_delay)
            if cost is None:
                logger.error("information_house_auto_investigate: failed to verify cost before start")
                return False

            logger.info(
                "information_house_auto_investigate: "
                f"adjust_attempt={attempt}, current_cost={cost}, target_cost={target_cost}"
            )
            if cost == target_cost:
                return True

            if attempt >= max_adjust:
                break

            self._click_roi(controller, add_button_roi if cost < target_cost else minus_button_roi)
            time.sleep(click_delay)

        logger.error(
            "information_house_auto_investigate: failed to adjust cost to target, "
            f"target_cost={target_cost}"
        )
        return False

    def _read_number_before_slash(self, context: Context, controller, roi, retry, retry_delay):
        text = self._read_ocr_text(context, controller, roi, [r"\d+\s*/"])
        if not text:
            for attempt in range(max(1, retry - 1)):
                time.sleep(retry_delay)
                text = self._read_ocr_text(context, controller, roi, [r"\d+\s*/"])
                if text:
                    break
                logger.warning(
                    "information_house_auto_investigate: failed to OCR stamina on "
                    f"retry {attempt + 1}/{max(1, retry - 1)}"
                )

        if not text:
            return None

        text = text.replace(",", "").replace(" ", "")
        match = re.search(r"(\d+)\s*/", text)
        if not match:
            match = re.search(r"\d+", text)
        return int(match.group(1) if match.lastindex else match.group(0)) if match else None

    def _read_number(self, context: Context, controller, roi, retry, retry_delay):
        for attempt in range(max(1, retry)):
            text = self._read_ocr_text(context, controller, roi, [r"\d+"])
            if text:
                text = text.replace(",", "").replace(" ", "")
                match = re.search(r"\d+", text)
                if match:
                    return int(match.group(0))

            if attempt < retry - 1:
                logger.warning(
                    "information_house_auto_investigate: failed to OCR number on "
                    f"attempt {attempt + 1}/{retry}"
                )
                time.sleep(retry_delay)

        return None

    def _read_ocr_text(self, context: Context, controller, roi, expected):
        controller.post_screencap().wait()
        image = controller.cached_image
        node_name = "_information_house_ocr"

        try:
            result = context.run_recognition(
                node_name,
                image,
                pipeline_override={
                    node_name: {
                        "recognition": "OCR",
                        "roi": roi,
                        "expected": expected,
                    }
                },
            )
            return self._extract_ocr_text(result)
        except Exception as e:
            logger.warning(f"information_house_auto_investigate: OCR error: {e}")
            return ""

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

    def _click_roi(self, controller, roi):
        x, y, w, h = roi
        controller.post_click(x + w // 2, y + h // 2).wait()

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
