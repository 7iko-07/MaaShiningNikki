import json
import re
import time
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils import logger


@AgentServer.custom_action("arena_compare")
class ArenaCompareAction(CustomAction):

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        area_a = params.get("area_a")
        area_b_list = params.get("area_b", [])
        refresh_roi = params.get("refresh")
        try:
            max_rounds = int(params.get("max_rounds", 5))
        except (ValueError, TypeError):
            max_rounds = 5

        if not area_a or not area_b_list or not refresh_roi:
            logger.error("arena_compare: missing required params (area_a, area_b, refresh)")
            return False

        controller = context.tasker.controller

        for round_num in range(max_rounds):
            logger.info(f"arena_compare: round {round_num + 1}/{max_rounds}")

            job = controller.post_screencap()
            job.wait()
            img = controller.cached_image

            player_power = self._ocr_number(context, img, area_a)
            if player_power is None:
                logger.warning("arena_compare: failed to OCR player power, retrying")
                time.sleep(0.5)
                continue

            logger.info(f"arena_compare: player power = {player_power}")

            found = False
            for i, area_b in enumerate(area_b_list):
                opp_power = self._ocr_number(context, img, area_b)
                if opp_power is None:
                    logger.warning(f"arena_compare: failed to OCR opponent {i}")
                    continue
                logger.info(f"arena_compare: opponent {i} power = {opp_power}")
                if player_power > opp_power:
                    bx, by, bw, bh = area_b
                    logger.info(f"arena_compare: player > opponent {i}, clicking opponent {i}")
                    controller.post_click(bx + bw // 2 - 60, by + bh // 2 - 100).wait()
                    found = True
                    break

            if found:
                return True

            if round_num < max_rounds - 1:
                rx, ry, rw, rh = refresh_roi
                logger.info("arena_compare: all opponents stronger, clicking refresh")
                controller.post_click(rx + rw // 2, ry + rh // 2).wait()
                time.sleep(1.5)
            else:
                logger.warning(f"arena_compare: no weaker opponent after {max_rounds} rounds")

        context.override_next(argv.node_name, [])
        return True

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
