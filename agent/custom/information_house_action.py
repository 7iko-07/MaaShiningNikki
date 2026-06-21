import json
import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils import logger, read_number_from_controller


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
        stamina = read_number_from_controller(
            context,
            controller,
            stamina_roi,
            "_information_house_ocr",
            [r"\d+\s*/"],
            retry=retry,
            retry_delay=retry_delay,
            log_prefix="information_house_auto_investigate",
            retry_label="stamina",
        )
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
            cost = read_number_from_controller(
                context,
                controller,
                cost_roi,
                "_information_house_ocr",
                [r"\d+"],
                retry=retry,
                retry_delay=retry_delay,
                log_prefix="information_house_auto_investigate",
                retry_label="number",
            )
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
