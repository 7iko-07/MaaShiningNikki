import json
import time
from enum import Enum, auto
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JRecognitionType, JTemplateMatch
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


@AgentServer.custom_action("arena_return_to_compare")
class ArenaReturnToCompareAction(CustomAction):
    """在同一总期限内重识别、点击返回，直到确认我的搭配。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        controller = context.tasker.controller
        deadline = time.monotonic() + 30
        next_click = 0.0
        try:
            while time.monotonic() < deadline and not context.tasker.stopping:
                if not controller.post_screencap().wait().succeeded:
                    logger.error("竞技场返回：截图失败")
                    return False
                image = controller.cached_image
                for node in ("竞技场搭配完成确认", "点击OK-竞技场", argv.node_name):
                    if context.tasker.stopping or time.monotonic() >= deadline:
                        return False
                    detail = context.run_recognition(node, image)
                    if detail is None:
                        logger.debug(f"竞技场返回：识别失败：{node}")
                        logger.error("竞技场：无法确认返回页面")
                        return False
                    if context.tasker.stopping or time.monotonic() >= deadline:
                        return False
                    if not detail.hit:
                        continue
                    if node == "竞技场搭配完成确认":
                        return True
                    # 弹窗命中时不再检查或点击背后的返回按钮。
                    if time.monotonic() >= next_click:
                        box = detail.box
                        if box is None or box.w <= 0 or box.h <= 0:
                            logger.debug(f"竞技场返回：点击框无效：{node}")
                            logger.error("竞技场：无法定位返回按钮")
                            return False
                        if not controller.post_click(
                            int(box.x + box.w / 2), int(box.y + box.h / 2)
                        ).wait().succeeded:
                            logger.debug(f"竞技场返回：点击失败：{node}")
                            logger.error("竞技场：点击返回按钮失败")
                            return False
                        next_click = time.monotonic() + 1
                    break
                time.sleep(min(0.3, max(0, deadline - time.monotonic())))
        except Exception as exc:
            logger.debug(f"竞技场返回失败：{exc}")
            logger.error("竞技场：返回搭配页面失败")
            return False
        logger.debug("竞技场返回未完成：任务停止或 30 秒内未识别到我的搭配")
        logger.error("竞技场：未能及时返回搭配页面，任务已停止")
        return False


@AgentServer.custom_action("arena_close_rank_popup")
class ArenaCloseRankPopupAction(CustomAction):
    """动画变化不代表关闭；只在文字仍可见时补点，限时确认消失。"""

    POPUP_NODES = ("竞技场排名提升页面", "竞技场段位晋级页面")

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        timeout = max(1, int(params.get("timeout", 10000))) / 1000
        total_timeout = max(1, int(params.get("total_timeout", 20000))) / 1000
        max_clicks = max(1, int(params.get("max_clicks", 3)))
        controller = context.tasker.controller
        started = time.monotonic()
        total_deadline = started + total_timeout
        deadline = min(total_deadline, started + timeout)
        current_node = argv.node_name
        if current_node not in self.POPUP_NODES:
            logger.debug(f"arena_close_rank_popup: unknown popup {current_node}")
            logger.error("竞技场：无法确认排名弹窗")
            return False
        next_click = 0.0
        clicks = dict.fromkeys(self.POPUP_NODES, 0)
        absent_count = 0
        try:
            while time.monotonic() < deadline:
                if context.tasker.stopping:
                    return False
                if not controller.post_screencap().wait().succeeded:
                    logger.debug("arena_close_rank_popup: screenshot failed")
                    logger.error("竞技场：获取画面失败")
                    return False
                image = controller.cached_image
                # Recognition only: never recursively execute this action/next.
                detail = context.run_recognition(current_node, image)
                if detail is None:
                    logger.debug("arena_close_rank_popup: recognition failed to start")
                    logger.error("竞技场：无法识别排名弹窗")
                    return False
                if context.tasker.stopping or time.monotonic() >= deadline:
                    return False
                if detail.hit:
                    absent_count = 0
                    if clicks[current_node] < max_clicks and time.monotonic() >= next_click:
                        box = detail.box
                        if box is None or box.w <= 0 or box.h <= 0:
                            logger.debug("arena_close_rank_popup: invalid text box")
                            logger.error("竞技场：无法定位弹窗按钮")
                            return False
                        if not controller.post_click(
                            int(box.x + box.w / 2), int(box.y + box.h / 2)
                        ).wait().succeeded:
                            logger.debug("arena_close_rank_popup: click failed")
                            logger.error("竞技场：关闭排名弹窗失败")
                            return False
                        clicks[current_node] += 1
                        next_click = time.monotonic() + 1.0
                        logger.debug(f"arena_close_rank_popup: node={current_node}, click {clicks[current_node]}/{max_clicks}")
                else:
                    absent_count += 1
                    if absent_count >= 3:
                        other_node = next(node for node in self.POPUP_NODES if node != current_node)
                        other = context.run_recognition(other_node, image)
                        if other is None:
                            logger.debug(f"arena_close_rank_popup: recognition failed to start: {other_node}")
                            logger.error("竞技场：无法识别排名弹窗")
                            return False
                        if context.tasker.stopping or time.monotonic() >= deadline:
                            return False
                        if other.hit:
                            logger.debug(f"arena_close_rank_popup: {current_node} disappeared, switching to {other_node}")
                            current_node = other_node
                            absent_count = 0
                            # Keep per-type click counts and a shared deadline even if animations alternate.
                            deadline = min(total_deadline, time.monotonic() + timeout)
                            time.sleep(min(0.3, max(0, deadline - time.monotonic())))
                            continue
                        # Hand rewards back to Pipeline; only recognize here, never run its next chain.
                        for exit_node in ("竞技场挑战奖励页面", "竞技场挑战返回页面"):
                            ready = context.run_recognition(exit_node, image)
                            if ready is None:
                                logger.debug(f"arena_close_rank_popup: recognition failed to start: {exit_node}")
                                logger.error("竞技场：无法确认弹窗已关闭")
                                return False
                            if context.tasker.stopping or time.monotonic() >= deadline:
                                return False
                            if ready.hit:
                                logger.debug(f"arena_close_rank_popup: text disappeared, next={exit_node}, clicks={clicks}")
                                return True
                # Exhausting clicks does not end the wait: the animation can finish itself.
                time.sleep(min(0.3, max(0, deadline - time.monotonic())))
        except Exception as exc:
            logger.debug(f"arena_close_rank_popup: failed: {exc}")
            logger.error("竞技场：关闭排名弹窗失败")
            return False
        logger.debug(f"arena_close_rank_popup: timeout, clicks={clicks}, absent_count={absent_count}")
        logger.error("竞技场：关闭排名弹窗超时")
        return False


@AgentServer.custom_action("arena_compare")
class ArenaCompareAction(CustomAction):
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        # Invocation-local counts: never share progress across tasks or accounts.
        counts = {">": 0, "<": 0, "=": 0}
        try:
            return self._run(context, argv, counts)
        except Exception as exc:
            logger.debug(f"arena_compare: failed: {exc}")
            logger.error("竞技场：挑战执行异常")
            return False
        finally:
            logger.info(
                f"竞技场挑战统计：自身战力大于对手 {counts['>']} 次，"
                f"小于 {counts['<']} 次，相等 {counts['=']} 次"
            )

    def _run(self, context, argv, counts):
        if context.tasker.stopping:
            return False
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
        node_data = context.get_node_data(argv.node_name) or {}
        challenge_after_refresh_limit = (
            node_data.get("attach", {}).get("challenge_after_refresh_limit") is True
        )

        if not area_a or not area_b_list or not refresh_roi:
            logger.debug("arena_compare: missing required params (area_a, area_b, refresh)")
            logger.error("竞技场：挑战配置不完整")
            return False
        if not self._valid_roi(area_a):
            logger.debug("arena_compare: invalid area_a")
            logger.error("竞技场：自身战力识别区域配置无效")
            return False
        if not self._valid_roi(refresh_roi):
            logger.debug("arena_compare: invalid refresh")
            logger.error("竞技场：刷新按钮配置无效")
            return False
        if not self._valid_roi_list(area_b_list):
            logger.debug("arena_compare: invalid area_b")
            logger.error("竞技场：对手识别区域配置无效")
            return False
        if not self._valid_roi(remaining_count_roi):
            logger.debug("arena_compare: invalid remaining_count_roi")
            logger.error("竞技场：剩余次数识别区域配置无效")
            return False
        controller = context.tasker.controller
        challenge_times = self._read_challenge_times(context, controller, remaining_count_roi)
        if challenge_times is None:
            return False
        if challenge_times <= 0:
            logger.debug(f"arena_compare: no remaining challenge times, challenge_times={challenge_times}")
            logger.info("本次挑战次数已用完")
            return context.override_next(argv.node_name, [])

        for _ in range(challenge_times):
            if context.tasker.stopping:
                return False
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
                challenge_after_refresh_limit,
                counts,
            )
            if outcome is ChallengeOutcome.STOPPED:
                return context.override_next(argv.node_name, [])
            if outcome is not ChallengeOutcome.CHALLENGED:
                return False

            if not self._run_result_flow(context, result_node):
                logger.debug(f"arena_compare: challenge result flow {result_node!r} failed")
                logger.warning("竞技场：未能确认挑战结果")
                return False

        return context.override_next(argv.node_name, [])

    def _read_challenge_times(self, context, controller, remaining_count_roi):
        if not remaining_count_roi:
            return None

        for attempt in range(3):
            if context.tasker.stopping:
                return None
            if not controller.post_screencap().wait().succeeded:
                logger.debug("arena_compare: remaining count screenshot failed")
                logger.error("竞技场：获取画面失败")
                return None
            remaining = read_ocr_number(
                context, controller.cached_image, "_arena_ocr_remaining_count",
                remaining_count_roi, [r"\d+\s*/\s*\d+"], "arena_compare",
            )
            if remaining is not None:
                logger.debug(f"arena_compare: remaining challenge times = {remaining}")
                return remaining
            if attempt < 2:
                time.sleep(1)
        logger.debug("arena_compare: remaining count OCR failed after 3 attempts")
        logger.error("竞技场：无法识别剩余挑战次数")
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
        challenge_after_refresh_limit=False,
        counts=None,
    ):
        # Keep the max_rounds parameter name for existing interface configurations.
        # The limit counts refresh clicks, excluding the initial recognition.
        refresh_count = 0
        ok_retry_count = 0

        while refresh_count <= max_rounds:
            if context.tasker.stopping:
                return ChallengeOutcome.FAILED
            if in_settlement():
                logger.info("竞技场已进入结算期，停止挑战")
                return ChallengeOutcome.STOPPED

            player_power, img = self._read_player_power(
                context,
                controller,
                area_a,
                player_power_retry,
                player_power_retry_delay,
            )
            if player_power is None:
                logger.debug("arena_compare: failed to OCR player power after retries")
                logger.warning("竞技场：无法识别自身战力")
                return ChallengeOutcome.FAILED

            ok_handled = False
            for i, area_b in enumerate(area_b_list):
                opp_power = self._read_opponent_power(context, controller, img, area_b)
                if opp_power is None:
                    logger.debug(f"arena_compare: opponent {i} OCR failed after retries")
                    logger.error("竞技场：无法识别对手战力")
                    return ChallengeOutcome.FAILED
                force_rightmost = (
                    challenge_after_refresh_limit
                    and refresh_count == max_rounds
                    and i == len(area_b_list) - 1
                )
                if player_power > opp_power or force_rightmost:
                    if context.tasker.stopping:
                        return ChallengeOutcome.FAILED
                    if in_settlement():
                        logger.info("竞技场已进入结算期，不再发起挑战")
                        return ChallengeOutcome.STOPPED
                    if not self._click_opponent(controller, area_b):
                        logger.debug("arena_compare: opponent click failed")
                        logger.error("竞技场：选择对手失败")
                        return ChallengeOutcome.FAILED
                    popup = self._handle_optional_ok_popup(context, controller, ok_node, post_challenge_delay)
                    if popup is PopupOutcome.FAILED:
                        return ChallengeOutcome.FAILED
                    if popup is PopupOutcome.HANDLED:
                        ok_retry_count += 1
                        ok_handled = True
                        if ok_retry_count > ok_retry_limit:
                            logger.debug(
                                f"arena_compare: OK popup appeared more than {ok_retry_limit} times"
                            )
                            logger.warning("竞技场：提示弹窗反复出现，停止挑战")
                            return ChallengeOutcome.FAILED
                        time.sleep(player_power_retry_delay)
                        break
                    relation = ">" if player_power > opp_power else "<" if player_power < opp_power else "="
                    if counts is not None:
                        counts[relation] += 1
                    position = "最右" if i == len(area_b_list) - 1 else "最左" if i == 0 else "中间"
                    logger.info(f"自身战力 {player_power} {relation} {position}选手战力 {opp_power}")
                    return ChallengeOutcome.CHALLENGED

            if ok_handled:
                continue

            if refresh_count < max_rounds:
                if context.tasker.stopping:
                    return ChallengeOutcome.FAILED
                if in_settlement():
                    logger.info("竞技场已进入结算期，不再刷新对手")
                    return ChallengeOutcome.STOPPED
                if not self._click_roi(controller, refresh_roi):
                    logger.debug("arena_compare: refresh click failed")
                    logger.error("竞技场：刷新对手失败")
                    return ChallengeOutcome.FAILED
                refresh_count += 1
                time.sleep(3)
            else:
                logger.debug(f"arena_compare: no weaker opponent after {refresh_count} refreshes")
                logger.warning(f"竞技场：已刷新 {refresh_count} 次，仍未找到较弱对手")
                return ChallengeOutcome.STOPPED

        return ChallengeOutcome.FAILED

    def _read_opponent_power(self, context, controller, image, roi):
        for attempt in range(3):
            if context.tasker.stopping:
                return None
            power = read_ocr_number(context, image, "_arena_ocr", roi,
                                    [r"\d[\d,]*"], "arena_compare", model="ppocr_v6")
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
            if context.tasker.stopping:
                return None, last_img
            if not controller.post_screencap().wait().succeeded:
                logger.debug("arena_compare: player power screenshot failed")
                logger.error("竞技场：获取画面失败")
                return None, last_img
            last_img = controller.cached_image

            player_power = read_ocr_number(
                context,
                self._mask_player_power_hint(context, last_img, area_a),
                "_arena_ocr",
                area_a,
                [r"\d[\d,]*"],
                "arena_compare",
                model="ppocr_v6",
            )
            if player_power is not None:
                return player_power, last_img

            if attempt < retry - 1:
                logger.debug(
                    f"arena_compare: failed to OCR player power on "
                    f"attempt {attempt + 1}/{retry}"
                )
                time.sleep(retry_delay)

        return None, last_img

    @staticmethod
    def _mask_player_power_hint(context, image, roi):
        if image is None:
            return image
        hint = context.run_recognition_direct(
            JRecognitionType.TemplateMatch,
            JTemplateMatch(roi=tuple(roi), template=["竞技场战力提示.png"],
                           green_mask=True, method=10001, threshold=[0.9]),
            image,
        )
        if not hint or not hint.hit or not hint.box:
            return image
        x, y, w, h = hint.box
        masked = image.copy()
        background_x = min(x + w, image.shape[1] - 1)
        masked[y:y + h, x:x + w] = image[y:y + h, background_x:background_x + 1]
        return masked

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
                logger.debug(f"arena_compare: popup task {ok_node!r} failed")
                logger.error("竞技场：关闭提示弹窗失败")
                return PopupOutcome.FAILED
            logger.debug(f"arena_compare: handled popup by task {ok_node!r}")
            return PopupOutcome.HANDLED
        except Exception as e:
            logger.debug(f"arena_compare: optional OK popup task {ok_node!r} failed: {e}")
            logger.warning("竞技场：关闭提示弹窗异常")
            return PopupOutcome.FAILED

    def _run_result_flow(self, context, result_node):
        try:
            result = context.run_task(result_node)
            return bool(result and result.status.succeeded)
        except Exception as e:
            logger.debug(f"arena_compare: result flow {result_node!r} failed: {e}")
            logger.warning("竞技场：确认挑战结果时发生异常")
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
