import json
import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JWaitFreezes
from utils import debug_exception, logger, read_number_from_controller

from .click_and_wait_change import ClickAndWaitChangeAction


@AgentServer.custom_action("information_house_find")
class InformationHouseFindAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            return self._find(context)
        except Exception:
            debug_exception("情报屋查找关卡异常")
            logger.error("情报屋：查找关卡失败")
            return False

    def _find(self, context):
        controller = context.tasker.controller
        compare_roi = (0, 198, 295, 1082)
        deadline = time.monotonic() + 30
        before = None
        comparer = ClickAndWaitChangeAction()

        def active():
            return not context.tasker.stopping and time.monotonic() < deadline

        while active():
            stable = context.wait_freezes(wait_freezes_param=JWaitFreezes(
                time=200, target=compare_roi, rate_limit=100,
                timeout=min(3000, max(1, int((deadline - time.monotonic()) * 1000))),
            ))
            if not active():
                break
            if not stable:
                logger.error("情报屋：等待列表稳定超时")
                return False
            capture = controller.post_screencap().wait()
            image = controller.cached_image if capture.succeeded else None
            if image is None or image.size == 0:
                logger.error("情报屋：获取画面失败")
                return False
            image = image.copy()
            if not active():
                break
            result = context.run_recognition("点击情报屋关卡", image)
            if not active():
                break
            if result is None:
                logger.error("情报屋：无法识别所选关卡")
                return False
            # 先检查滑动后的目标，再判断到底，确保最后一屏也能进入。
            if result.hit:
                if result.box.w <= 0 or result.box.h <= 0:
                    logger.error("情报屋：无法定位所选关卡")
                    return False
                clicked = context.run_action("点击情报屋关卡", box=result.box)
                if not clicked or not clicked.success:
                    logger.error("情报屋：点击所选关卡失败")
                    return False
                return not context.tasker.stopping
            if before is not None:
                changed, ratio = comparer._image_changed(
                    before, image, compare_roi, 30, 0.01,
                )
                logger.debug(f"情报屋上划后画面变化比例：{ratio:.6f}")
                if not changed:
                    logger.error("情报屋：未找到所选关卡")
                    return False
            before = image
            if not active():
                break
            swipe = context.run_task("上滑情报屋")
            if not swipe or not swipe.status.succeeded:
                if not context.tasker.stopping:
                    logger.error("情报屋：上划失败")
                return False

        if not context.tasker.stopping:
            logger.error("情报屋：查找所选关卡超时")
        return False


@AgentServer.custom_action("information_house_auto_investigate")
class InformationHouseAutoInvestigateAction(CustomAction):
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}

        stamina_roi = params.get("stamina_roi", [0, 0, 215, 49])
        add_button_roi = params.get("add_button_roi", [581, 590, 34, 29])
        cost_roi = params.get("cost_roi", [496, 821, 108, 36])
        minus_button_roi = params.get("minus_button_roi", params.get("confirm_button_roi", [348, 593, 36, 24]))
        start_button_roi = params.get("start_button_roi", [412, 759, 132, 42])

        max_adjust = self._as_int(params.get("max_adjust"), 8)
        retry = self._as_int(params.get("retry"), 3)
        retry_delay = self._as_float(params.get("retry_delay"), 1.0)
        load_delay = self._as_float(params.get("load_delay"), 1.5)
        click_delay = self._as_float(params.get("click_delay"), 0.8)
        initial_count = self._as_int(params.get("initial_count"), 2)
        challenge_times = self._as_int(params.get("challenge_times"), 0)

        if not all(self._valid_roi(roi) for roi in (stamina_roi, add_button_roi, cost_roi, minus_button_roi, start_button_roi)):
            logger.debug("information_house_auto_investigate: invalid roi param")
            logger.error("情报屋：操作区域配置无效")
            return False

        controller = context.tasker.controller
        if load_delay > 0:
            logger.debug(f"information_house_auto_investigate: waiting {load_delay:.1f}s for page load")
            time.sleep(load_delay)

        stamina = read_number_from_controller(
            context,
            controller,
            stamina_roi,
            "_information_house_ocr",
            [r"\d+\s*/", r"\d+"],
            retry=retry,
            retry_delay=retry_delay,
            log_prefix="information_house_auto_investigate",
            retry_label="stamina",
        )
        if stamina is None:
            logger.debug("information_house_auto_investigate: failed to read stamina")
            logger.error("情报屋：无法识别当前体力")
            return False

        logger.debug(f"information_house_auto_investigate: stamina={stamina}")
        cost = self._read_cost(
            context,
            controller,
            cost_roi,
            retry,
            retry_delay,
        )
        if cost is None:
            logger.debug("information_house_auto_investigate: failed to read initial cost")
            logger.error("情报屋：无法识别体力消耗")
            return False

        if cost > stamina:
            logger.debug(
                "information_house_auto_investigate: stamina is not enough for current cost, "
                f"stamina={stamina}, cost={cost}"
            )
            logger.info("情报屋：体力不足，结束调查")
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

        logger.debug(
            "information_house_auto_investigate: "
            f"start investigate, stamina={stamina}, target_cost={target_cost}"
        )
        logger.info(f"情报屋：开始调查，预计消耗 {target_cost} 体力")
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

        logger.debug(
            "information_house_auto_investigate: "
            f"initial_count={initial_count}, challenge_times={challenge_times}, add_count={add_count}"
        )

        for attempt in range(add_count):
            logger.debug(
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
                logger.debug("information_house_auto_investigate: failed to read cost after target add")
                logger.error("情报屋：无法确认调查次数")
                return None

            logger.debug(
                "information_house_auto_investigate: "
                f"target_cost_after_add={next_cost}, previous_cost={current_cost}, stamina={stamina}"
            )
            if next_cost == current_cost:
                logger.debug("information_house_auto_investigate: add did not change cost before target count")
                return current_cost

            if next_cost > stamina:
                logger.debug(
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
                    logger.debug("information_house_auto_investigate: failed to read cost after target rollback")
                    logger.error("情报屋：无法确认调整后的体力消耗")
                    return None

                if rollback_cost != current_cost:
                    logger.debug(
                        "information_house_auto_investigate: target rollback cost is different from expected, "
                        f"rollback_cost={rollback_cost}, expected={current_cost}"
                    )
                    logger.warning("情报屋：体力消耗与预期不符")
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
            logger.debug(
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
                logger.debug("information_house_auto_investigate: failed to read cost after add")
                logger.error("情报屋：无法确认调查次数")
                return None

            logger.debug(
                "information_house_auto_investigate: "
                f"cost_after_add={next_cost}, previous_cost={current_cost}, stamina={stamina}"
            )
            if next_cost == current_cost:
                logger.debug("information_house_auto_investigate: add did not change cost, already max count")
                return current_cost

            if next_cost > stamina:
                logger.debug(
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
                    logger.debug("information_house_auto_investigate: failed to read cost after rollback")
                    logger.error("情报屋：无法确认调整后的体力消耗")
                    return None

                if rollback_cost != current_cost:
                    logger.debug(
                        "information_house_auto_investigate: rollback cost is different from expected, "
                        f"rollback_cost={rollback_cost}, expected={current_cost}"
                    )
                    logger.warning("情报屋：体力消耗与预期不符")
                    if rollback_cost > stamina:
                        return None
                    return rollback_cost

                return current_cost

            current_cost = next_cost

        logger.debug(
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
