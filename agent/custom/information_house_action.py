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

        max_adjust = self._as_int(params.get("max_adjust"), 8)
        retry = self._as_int(params.get("retry"), 3)
        retry_delay = self._as_float(params.get("retry_delay"), 0.5)
        click_delay = self._as_float(params.get("click_delay"), 0.8)
        initial_count = self._as_int(params.get("initial_count"), 2)
        challenge_times = self._as_int(params.get("challenge_times"), 0)

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
        cost = self._read_cost(
            context,
            controller,
            cost_roi,
            retry,
            retry_delay,
        )
        if cost is None:
            logger.error("information_house_auto_investigate: failed to read initial cost")
            return False

        if cost > stamina:
            logger.info(
                "information_house_auto_investigate: stamina is not enough for current cost, "
                f"stamina={stamina}, cost={cost}"
            )
            return True

        if challenge_times > 0:
            target_cost = self._increase_to_target_count(
                context,
                controller,
                cost_roi,
                add_button_roi,
                minus_button_roi,
                stamina,
                cost,
                retry,
                retry_delay,
                click_delay,
                initial_count,
                challenge_times,
                max_adjust,
            )
        else:
            target_cost = self._increase_to_affordable_max(
                context,
                controller,
                cost_roi,
                add_button_roi,
                minus_button_roi,
                stamina,
                cost,
                retry,
                retry_delay,
                click_delay,
                max_adjust,
            )
        if target_cost is None:
            return False

        logger.info(
            "information_house_auto_investigate: "
            f"start investigate, stamina={stamina}, target_cost={target_cost}"
        )
        self._click_roi(controller, start_button_roi)
        return True

    def _increase_to_target_count(
        self,
        context,
        controller,
        cost_roi,
        add_button_roi,
        minus_button_roi,
        stamina,
        current_cost,
        retry,
        retry_delay,
        click_delay,
        initial_count,
        challenge_times,
        max_adjust,
    ):
        target_count = max(initial_count, challenge_times)
        add_count = min(max(0, target_count - initial_count), max(0, max_adjust))

        logger.info(
            "information_house_auto_investigate: "
            f"initial_count={initial_count}, challenge_times={challenge_times}, add_count={add_count}"
        )

        for attempt in range(add_count):
            logger.info(
                "information_house_auto_investigate: "
                f"target_increase_attempt={attempt + 1}/{add_count}, current_cost={current_cost}, stamina={stamina}"
            )

            self._click_roi(controller, add_button_roi)
            time.sleep(click_delay)

            next_cost = self._read_cost(
                context,
                controller,
                cost_roi,
                retry,
                retry_delay,
            )
            if next_cost is None:
                logger.error("information_house_auto_investigate: failed to read cost after target add")
                return None

            logger.info(
                "information_house_auto_investigate: "
                f"target_cost_after_add={next_cost}, previous_cost={current_cost}, stamina={stamina}"
            )
            if next_cost == current_cost:
                logger.info("information_house_auto_investigate: add did not change cost before target count")
                return current_cost

            if next_cost > stamina:
                logger.info(
                    "information_house_auto_investigate: target cost exceeds stamina after add, "
                    f"cost={next_cost}, stamina={stamina}, rollback_cost={current_cost}"
                )
                self._click_roi(controller, minus_button_roi)
                time.sleep(click_delay)

                rollback_cost = self._read_cost(
                    context,
                    controller,
                    cost_roi,
                    retry,
                    retry_delay,
                )
                if rollback_cost is None:
                    logger.error("information_house_auto_investigate: failed to read cost after target rollback")
                    return None

                if rollback_cost != current_cost:
                    logger.warning(
                        "information_house_auto_investigate: target rollback cost is different from expected, "
                        f"rollback_cost={rollback_cost}, expected={current_cost}"
                    )
                    if rollback_cost > stamina:
                        return None
                    return rollback_cost

                return current_cost

            current_cost = next_cost

        return current_cost

    def _increase_to_affordable_max(
        self,
        context,
        controller,
        cost_roi,
        add_button_roi,
        minus_button_roi,
        stamina,
        current_cost,
        retry,
        retry_delay,
        click_delay,
        max_adjust,
    ):
        for attempt in range(max(0, max_adjust)):
            logger.info(
                "information_house_auto_investigate: "
                f"increase_attempt={attempt + 1}, current_cost={current_cost}, stamina={stamina}"
            )

            self._click_roi(controller, add_button_roi)
            time.sleep(click_delay)

            next_cost = self._read_cost(
                context,
                controller,
                cost_roi,
                retry,
                retry_delay,
            )
            if next_cost is None:
                logger.error("information_house_auto_investigate: failed to read cost after add")
                return None

            logger.info(
                "information_house_auto_investigate: "
                f"cost_after_add={next_cost}, previous_cost={current_cost}, stamina={stamina}"
            )
            if next_cost == current_cost:
                logger.info("information_house_auto_investigate: add did not change cost, already max count")
                return current_cost

            if next_cost > stamina:
                logger.info(
                    "information_house_auto_investigate: cost exceeds stamina after add, "
                    f"cost={next_cost}, stamina={stamina}, rollback_cost={current_cost}"
                )
                self._click_roi(controller, minus_button_roi)
                time.sleep(click_delay)

                rollback_cost = self._read_cost(
                    context,
                    controller,
                    cost_roi,
                    retry,
                    retry_delay,
                )
                if rollback_cost is None:
                    logger.error("information_house_auto_investigate: failed to read cost after rollback")
                    return None

                if rollback_cost != current_cost:
                    logger.warning(
                        "information_house_auto_investigate: rollback cost is different from expected, "
                        f"rollback_cost={rollback_cost}, expected={current_cost}"
                    )
                    if rollback_cost > stamina:
                        return None
                    return rollback_cost

                return current_cost

            current_cost = next_cost

        logger.info(
            "information_house_auto_investigate: reached max_adjust while increasing, "
            f"cost={current_cost}, max_adjust={max_adjust}"
        )
        return current_cost

    def _read_cost(
        self,
        context,
        controller,
        cost_roi,
        retry,
        retry_delay,
    ):
        return read_number_from_controller(
            context,
            controller,
            cost_roi,
            "_information_house_ocr",
            [r"\d+"],
            retry=retry,
            retry_delay=retry_delay,
            log_prefix="information_house_auto_investigate",
            retry_label="cost",
        )

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
