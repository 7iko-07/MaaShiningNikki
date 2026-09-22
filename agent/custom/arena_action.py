import json
import time
from enum import Enum, auto
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils import logger, read_ocr_number
from utils.arena_state import in_settlement


class ChallengeOutcome(Enum):
    CHALLENGED = auto()
    STOPPED = auto()
    FAILED = auto()


class PopupOutcome(Enum):
    ABSENT = auto()
    HANDLED = auto()
    FAILED = auto()


@AgentServer.custom_action("arena_compare")
class ArenaCompareAction(CustomAction):
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        if in_settlement():
            logger.info("竞技场结算中，停止挑战")
            return context.override_next(argv.node_name, [])
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
        max_rounds = self._as_int(params.get("max_rounds"), 5)
        ok_node = params.get("ok_node", "点击OK-竞技场")
        result_node = params.get("result_node", "竞技场挑战结果处理")
        ok_retry_limit = self._as_int(params.get("ok_retry_limit"), 3)
        post_challenge_delay = self._as_float(params.get("post_challenge_delay"), 1.0)
        player_power_retry = self._as_int(params.get("player_power_retry"), 5)
        player_power_retry_delay = self._as_float(params.get("player_power_retry_delay"), 1.5)

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
        if not self._valid_roi(remaining_count_roi):
            logger.error("arena_compare: invalid remaining_count_roi")
            return False
        controller = context.tasker.controller
        challenge_times = self._read_challenge_times(context, controller, remaining_count_roi)
        if challenge_times is None:
            return False
        if challenge_times <= 0:
            logger.info(f"arena_compare: no remaining challenge times, challenge_times={challenge_times}")
            return context.override_next(argv.node_name, [])

        for challenge_index in range(challenge_times):
            logger.info(f"arena_compare: challenge {challenge_index + 1}/{challenge_times}")
            outcome = self._challenge_once(
                context,
                controller,
                area_a,
                area_b_list,
                refresh_roi,
                max_rounds,
                ok_node,
                ok_retry_limit,
                post_challenge_delay,
                player_power_retry,
                player_power_retry_delay,
            )
            if outcome is ChallengeOutcome.STOPPED:
                return context.override_next(argv.node_name, [])
            if outcome is not ChallengeOutcome.CHALLENGED:
                return False

            if not self._run_result_flow(context, result_node):
                logger.warning(f"arena_compare: challenge result flow {result_node!r} failed")
                return False

        return context.override_next(argv.node_name, [])

    def _read_challenge_times(self, context, controller, remaining_count_roi):
        if not remaining_count_roi:
            return None

        for attempt in range(3):
            if not controller.post_screencap().wait().succeeded:
                logger.error("arena_compare: remaining count screenshot failed")
                return None
            remaining = read_ocr_number(
                context, controller.cached_image, "_arena_ocr_remaining_count",
                remaining_count_roi, [r"\d+\s*/\s*\d+"], "arena_compare",
            )
            if remaining is not None:
                logger.info(f"arena_compare: remaining challenge times = {remaining}")
                return remaining
            if attempt < 2:
                time.sleep(1)
        logger.error("arena_compare: remaining count OCR failed after 3 attempts")
        return None

    def _challenge_once(
        self,
        context,
        controller,
        area_a,
        area_b_list,
        refresh_roi,
        max_rounds,
        ok_node,
        ok_retry_limit,
        post_challenge_delay,
        player_power_retry,
        player_power_retry_delay,
    ):
        # Keep the max_rounds parameter name for existing interface configurations.
        # The limit counts refresh clicks, excluding the initial recognition.
        refresh_count = 0
        ok_retry_count = 0

        while refresh_count <= max_rounds:
            if in_settlement():
                logger.info("竞技场已进入结算期，停止挑战")
                return ChallengeOutcome.STOPPED
            logger.info(f"arena_compare: recognizing opponents, refreshes used {refresh_count}/{max_rounds}")

            player_power, img = self._read_player_power(
                context,
                controller,
                area_a,
                player_power_retry,
                player_power_retry_delay,
            )
            if player_power is None:
                logger.warning("arena_compare: failed to OCR player power after retries")
                return ChallengeOutcome.FAILED

            logger.info(f"arena_compare: player power = {player_power}")

            ok_handled = False
            for i, area_b in enumerate(area_b_list):
                opp_power = self._read_opponent_power(context, controller, img, area_b)
                if opp_power is None:
                    logger.error(f"arena_compare: opponent {i} OCR failed after retries")
                    return ChallengeOutcome.FAILED
                logger.info(f"arena_compare: opponent {i} power = {opp_power}")
                if player_power > opp_power:
                    if in_settlement():
                        logger.info("竞技场已进入结算期，不再发起挑战")
                        return ChallengeOutcome.STOPPED
                    logger.info(f"arena_compare: player > opponent {i}, clicking opponent {i}")
                    if not self._click_opponent(controller, area_b):
                        logger.error("arena_compare: opponent click failed")
                        return ChallengeOutcome.FAILED
                    popup = self._handle_optional_ok_popup(context, controller, ok_node, post_challenge_delay)
                    if popup is PopupOutcome.FAILED:
                        return ChallengeOutcome.FAILED
                    if popup is PopupOutcome.HANDLED:
                        ok_retry_count += 1
                        ok_handled = True
                        if ok_retry_count > ok_retry_limit:
                            logger.warning(
                                f"arena_compare: OK popup appeared more than {ok_retry_limit} times"
                            )
                            return ChallengeOutcome.FAILED
                        logger.info("arena_compare: OK popup handled, retrying power recognition")
                        time.sleep(player_power_retry_delay)
                        break
                    return ChallengeOutcome.CHALLENGED

            if ok_handled:
                continue

            if refresh_count < max_rounds:
                if in_settlement():
                    logger.info("竞技场已进入结算期，不再刷新对手")
                    return ChallengeOutcome.STOPPED
                logger.info(f"arena_compare: no weaker opponent, clicking refresh {refresh_count + 1}/{max_rounds}")
                if not self._click_roi(controller, refresh_roi):
                    logger.error("arena_compare: refresh click failed")
                    return ChallengeOutcome.FAILED
                refresh_count += 1
                time.sleep(3)
            else:
                logger.warning(f"arena_compare: no weaker opponent after {refresh_count} refreshes")
                return ChallengeOutcome.STOPPED

        return ChallengeOutcome.FAILED

    def _read_opponent_power(self, context, controller, image, roi):
        for attempt in range(3):
            power = read_ocr_number(context, image, "_arena_ocr", roi,
                                    [r"\d[\d,]*"], "arena_compare")
            if power is not None:
                return power
            if attempt < 2:
                time.sleep(1)
                if not controller.post_screencap().wait().succeeded:
                    return None
                image = controller.cached_image
        return None

    def _read_player_power(
        self,
        context,
        controller,
        area_a,
        retry,
        retry_delay,
    ):
        retry = max(1, retry)
        last_img = None

        for attempt in range(retry):
            if not controller.post_screencap().wait().succeeded:
                logger.error("arena_compare: player power screenshot failed")
                return None, last_img
            last_img = controller.cached_image

            player_power = read_ocr_number(
                context,
                last_img,
                "_arena_ocr",
                area_a,
                [r"\d[\d,]*"],
                "arena_compare",
            )
            if player_power is not None:
                return player_power, last_img

            if attempt < retry - 1:
                logger.warning(
                    f"arena_compare: failed to OCR player power on "
                    f"attempt {attempt + 1}/{retry}"
                )
                time.sleep(retry_delay)

        return None, last_img

    def _handle_optional_ok_popup(self, context, controller, ok_node, delay):
        if delay > 0:
            time.sleep(delay)

        try:
            if not controller.post_screencap().wait().succeeded:
                return PopupOutcome.FAILED
            detail = context.run_recognition(ok_node, controller.cached_image)
            if not (detail and detail.hit):
                return PopupOutcome.ABSENT

            result = context.run_task(ok_node)
            if not (result and result.status.succeeded):
                logger.error(f"arena_compare: popup task {ok_node!r} failed")
                return PopupOutcome.FAILED
            logger.info(f"arena_compare: handled popup by task {ok_node!r}")
            return PopupOutcome.HANDLED
        except Exception as e:
            logger.warning(f"arena_compare: optional OK popup task {ok_node!r} failed: {e}")
            return PopupOutcome.FAILED

    def _run_result_flow(self, context, result_node):
        try:
            result = context.run_task(result_node)
            return bool(result and result.status.succeeded)
        except Exception as e:
            logger.warning(f"arena_compare: result flow {result_node!r} failed: {e}")
            return False

    def _click_opponent(self, controller, roi):
        x, y, w, h = roi
        return controller.post_click(x + w // 2 - 60, y + h // 2 - 100).wait().succeeded

    def _click_roi(self, controller, roi):
        x, y, w, h = roi
        return controller.post_click(x + w // 2, y + h // 2).wait().succeeded

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
