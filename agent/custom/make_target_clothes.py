import json
import time

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JActionType, JSwipe
from utils import logger, read_ocr_number, read_ocr_text


@AgentServer.custom_action("make_target_clothes_challenge")
class MakeTargetClothesChallengeAction(CustomAction):

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}

        challenge_count_roi = params.get("challenge_count_roi")
        stamina_roi = params.get("stamina_roi")
        once_button = params.get("once_button")
        multi_button = params.get("multi_button")

        if not challenge_count_roi or not stamina_roi or not once_button or not multi_button:
            logger.error(
                "make_target_clothes_challenge: 缺少必要参数 "
                "(challenge_count_roi, stamina_roi, once_button, multi_button)"
            )
            return False

        stamina_cost = self._as_int(params.get("stamina_cost"), 5)
        max_multi = self._as_int(params.get("max_multi"), 10)
        retry = self._as_int(params.get("retry"), 3)
        load_delay = self._as_float(params.get("load_delay"), 3.0)
        retry_delay = self._as_float(params.get("retry_delay"), 1.5)
        click_delay = self._as_float(params.get("click_delay"), 1.2)
        popup_close_roi = params.get("popup_close_roi", [221, 137, 137, 78])
        popup_close_delay = self._as_float(params.get("popup_close_delay"), 2.0)

        if stamina_cost <= 0 or max_multi <= 1:
            logger.error("make_target_clothes_challenge: 单次体力消耗或最大批量次数参数无效")
            return False

        controller = context.tasker.controller
        challenge_count = None
        stamina = None

        if load_delay > 0:
            logger.info(f"make_target_clothes_challenge: 等待页面加载 {load_delay:.1f} 秒")
            time.sleep(load_delay)

        for attempt in range(retry):
            controller.post_screencap().wait()
            img = controller.cached_image

            challenge_count = read_ocr_number(
                context,
                img,
                "_make_target_clothes_challenge_count_ocr",
                challenge_count_roi,
                [r"\d+"],
                "make_target_clothes_challenge",
            )
            stamina = read_ocr_number(
                context,
                img,
                "_make_target_clothes_stamina_ocr",
                stamina_roi,
                [r"\d+"],
                "make_target_clothes_challenge",
            )

            if challenge_count is not None and stamina is not None:
                break

            logger.warning(
                "make_target_clothes_challenge: OCR 识别失败，尝试次数 "
                f"{attempt + 1}/{retry}, 剩余挑战次数={challenge_count}，体力={stamina}"
            )
            if attempt < retry - 1 and retry_delay > 0:
                time.sleep(retry_delay)

        if challenge_count is None:
            logger.error("make_target_clothes_challenge: 无法识别剩余挑战次数")
            return False

        if stamina is None:
            logger.warning("make_target_clothes_challenge: 无法识别体力，按体力足够处理")
            challenge_times = challenge_count
        else:
            available_by_stamina = stamina // stamina_cost
            challenge_times = min(challenge_count, available_by_stamina)

        multi_clicks, once_clicks = self._calc_clicks(
            challenge_count=challenge_count,
            challenge_times=challenge_times,
            max_multi=max_multi,
        )

        logger.info(
            "make_target_clothes_challenge: "
            f"剩余挑战次数={challenge_count}，体力={stamina}, "
            f"计划挑战次数={challenge_times}，批量点击次数={multi_clicks}，单次点击次数={once_clicks}"
        )

        if challenge_times <= 0:
            # context.override_next(argv.node_name, [])
            return True

        for _ in range(multi_clicks):
            self._click_roi(controller, multi_button)
            time.sleep(5.0)
            self._click_roi(controller, popup_close_roi)
            time.sleep(popup_close_delay)

        for _ in range(once_clicks):
            self._click_roi(controller, once_button)
            time.sleep(click_delay)
            self._click_roi(controller, popup_close_roi)
            time.sleep(popup_close_delay)

        return True

    def _calc_clicks(self, challenge_count, challenge_times, max_multi):
        if challenge_times <= 0:
            return 0, 0

        multi_clicks = challenge_times // max_multi
        once_clicks = challenge_times % max_multi

        remaining_challenges = challenge_count - multi_clicks * max_multi
        if 1 < remaining_challenges <= max_multi and once_clicks == remaining_challenges:
            multi_clicks += 1
            once_clicks = 0

        return multi_clicks, once_clicks

    def _click_roi(self, controller, roi):
        x, y, w, h = roi
        controller.post_click(x + w // 2, y + h // 2).wait()

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


@AgentServer.custom_action("make_target_clothes_batch_challenge")
class MakeTargetClothesBatchChallengeAction(CustomAction):

    _task_challenge_counts = {}

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}

        mode = params.get("mode")
        stamina_roi = params.get("stamina_roi")
        stamina_cost_roi = params.get("stamina_cost_roi")
        once_button = params.get("once_button")
        multi_button = params.get("multi_button")
        remaining_count_roi = params.get("remaining_count_roi")
        sufficient_roi = params.get("sufficient_roi", [379, 355, 265, 63])
        sufficient_expected = params.get("sufficient_expected", ["数量已足够"])

        if mode not in ("heart_maze", "main_story"):
            logger.error(f"make_target_clothes_batch_challenge: 模式参数无效：{mode}")
            return False
        if not stamina_roi or not stamina_cost_roi or not once_button or not multi_button:
            logger.error(
                "make_target_clothes_batch_challenge: 缺少必要参数 "
                "(stamina_roi, stamina_cost_roi, once_button, multi_button)"
            )
            return False
        if mode == "heart_maze" and not remaining_count_roi:
            logger.error(
                "make_target_clothes_batch_challenge: "
                "心灵迷宫模式需要配置剩余挑战次数识别区域 remaining_count_roi"
            )
            return False
        if not sufficient_roi or not sufficient_expected:
            logger.error(
                "make_target_clothes_batch_challenge: 缺少必要参数 "
                "(sufficient_roi, sufficient_expected)"
            )
            return False

        max_challenges = self._as_int(params.get("max_challenges"), 0)
        max_multi = self._as_int(params.get("max_multi"), 10)
        retry = self._as_int(params.get("retry"), 3)
        retry_delay = self._as_float(params.get("retry_delay"), 1.5)
        load_delay = self._as_float(params.get("load_delay"), 3.0)
        result_delay = self._as_float(params.get("result_delay"), 5.0)
        popup_close_roi = params.get("popup_close_roi", [221, 137, 137, 78])
        popup_close_delay = self._as_float(params.get("popup_close_delay"), 2.0)
        max_rounds = self._as_int(params.get("max_rounds"), 100)
        insufficient_stamina_next = params.get("insufficient_stamina_next")
        default_limit_next = (
            ["返回制衣页面-心灵迷宫达到上限"]
            if mode == "heart_maze"
            else ["返回制衣页面-主线达到上限"]
        )
        limit_next = params.get("limit_next", default_limit_next)

        if max_challenges < 0 or max_multi <= 0 or retry <= 0 or max_rounds <= 0:
            logger.error("make_target_clothes_batch_challenge: 数值参数无效")
            return False

        controller = context.tasker.controller
        task_id = argv.task_detail.task_id
        state_key = (task_id, mode)
        challenged = self._task_challenge_counts.get(state_key, 0)

        if max_challenges > 0 and challenged >= max_challenges:
            return self._stop_for_limit(
                context,
                argv.node_name,
                limit_next,
                max_challenges,
                challenged,
            )

        if load_delay > 0:
            logger.info(
                f"make_target_clothes_batch_challenge: 等待页面加载 {load_delay:.1f} 秒"
            )
            time.sleep(load_delay)

        if self._has_sufficient_quantity(
            context=context,
            controller=controller,
            roi=sufficient_roi,
            expected=sufficient_expected,
            retry=retry,
            retry_delay=retry_delay,
        ):
            logger.info(
                "make_target_clothes_batch_challenge: 材料数量已足够，"
                "跳过挑战"
            )
            return True

        for round_index in range(max_rounds):
            values = self._read_round_values(
                context=context,
                controller=controller,
                mode=mode,
                stamina_roi=stamina_roi,
                stamina_cost_roi=stamina_cost_roi,
                remaining_count_roi=remaining_count_roi,
                retry=retry,
                retry_delay=retry_delay,
            )
            if values is None:
                return False

            stamina, stamina_cost, remaining_count = values
            possible_batch = min(max_multi, stamina // stamina_cost)
            if remaining_count is not None:
                possible_batch = min(possible_batch, remaining_count)

            allowance = None if max_challenges == 0 else max_challenges - challenged
            logger.info(
                "make_target_clothes_batch_challenge: "
                f"轮次={round_index + 1}，模式={'心灵迷宫' if mode == 'heart_maze' else '主线'}，体力={stamina}，"
                f"单次体力消耗={stamina_cost}，剩余挑战次数={remaining_count if remaining_count is not None else '不适用'}，"
                f"已挑战次数={challenged}，剩余允许次数={allowance if allowance is not None else '不限'}，"
                f"本轮最多可挑战次数={possible_batch}"
            )

            if mode == "heart_maze" and remaining_count == 0:
                logger.info(
                    "make_target_clothes_batch_challenge: 心灵迷宫剩余挑战"
                    "次数为 0，结束心灵迷宫材料阶段"
                )
                if not context.override_next(argv.node_name, limit_next):
                    logger.error(
                        "make_target_clothes_batch_challenge: 覆盖后继节点失败："
                        "心灵迷宫挑战次数已耗尽"
                    )
                    return False
                return True

            if stamina < stamina_cost and insufficient_stamina_next:
                logger.info(
                    "make_target_clothes_batch_challenge: 体力不足，"
                    f"体力={stamina}，单次体力消耗={stamina_cost}"
                )
                if not context.override_next(
                    argv.node_name,
                    insufficient_stamina_next,
                ):
                    logger.error(
                        "make_target_clothes_batch_challenge: 覆盖后继节点失败："
                        "体力不足"
                    )
                    return False
                return True

            if possible_batch <= 0:
                return True
            if allowance is not None and possible_batch > allowance:
                return self._stop_for_limit(
                    context,
                    argv.node_name,
                    limit_next,
                    max_challenges,
                    challenged,
                )

            button = once_button if possible_batch == 1 else multi_button
            logger.info(
                "make_target_clothes_batch_challenge: "
                f"点击{'挑战一次' if possible_batch == 1 else '批量挑战'}按钮"
            )
            self._click_roi(controller, button)
            time.sleep(result_delay)
            quantity_sufficient = self._has_sufficient_quantity(
                context=context,
                controller=controller,
                roi=sufficient_roi,
                expected=sufficient_expected,
                retry=retry,
                retry_delay=retry_delay,
            )
            self._click_roi(controller, popup_close_roi)
            time.sleep(popup_close_delay)

            new_stamina = self._read_required_number(
                context=context,
                controller=controller,
                roi=stamina_roi,
                node_name="_make_target_clothes_batch_stamina_after_ocr",
                expected=[r"\d+\s*/\s*\d+"],
                retry=retry,
                retry_delay=retry_delay,
                label="挑战后体力",
            )
            if new_stamina is None:
                return False

            consumed = stamina - new_stamina
            if consumed <= 0:
                if quantity_sufficient:
                    return True
                logger.warning(
                    "make_target_clothes_batch_challenge: 体力未减少，"
                    "未识别到材料数量已足够，结束当前材料阶段"
                )
                if not context.override_next(argv.node_name, limit_next):
                    logger.error(
                        "make_target_clothes_batch_challenge: 覆盖后继节点失败："
                        "挑战未取得进展"
                    )
                    return False
                return True
            if consumed % stamina_cost != 0:
                logger.error(
                    "make_target_clothes_batch_challenge: 体力变化量无法被单次消耗整除，"
                    f"挑战前={stamina}，挑战后={new_stamina}，单次消耗={stamina_cost}"
                )
                return False

            actual_batch = consumed // stamina_cost
            if actual_batch > possible_batch:
                logger.error(
                    "make_target_clothes_batch_challenge: 实际挑战次数超过安全预估："
                    f"实际次数={actual_batch}，预估次数={possible_batch}"
                )
                return False

            challenged += actual_batch
            self._task_challenge_counts[state_key] = challenged
            if max_challenges > 0 and challenged >= max_challenges:
                return self._stop_for_limit(
                    context,
                    argv.node_name,
                    limit_next,
                    max_challenges,
                    challenged,
                )

            if quantity_sufficient:
                logger.info(
                    "make_target_clothes_batch_challenge: 材料数量已足够，"
                    "返回复查当前服装材料"
                )
                return True

        logger.error(
            f"make_target_clothes_batch_challenge: 已超过最大循环轮数 {max_rounds}"
        )
        return False

    @classmethod
    def clear_task_state(cls, task_id, mode):
        cls._task_challenge_counts.pop((task_id, mode), None)

    def _stop_for_limit(
        self,
        context,
        node_name,
        limit_next,
        max_challenges,
        challenged,
    ):
        logger.info(
            "make_target_clothes_batch_challenge: 已达到挑战次数上限，停止挑战，"
            f"已挑战次数={challenged}，挑战次数上限={max_challenges}"
        )
        if not context.override_next(node_name, limit_next):
            logger.error(
                "make_target_clothes_batch_challenge: 覆盖后继节点失败："
                "已达到挑战次数上限"
            )
            return False
        return True

    def _has_sufficient_quantity(
        self,
        context,
        controller,
        roi,
        expected,
        retry,
        retry_delay,
    ):
        for attempt in range(retry):
            controller.post_screencap().wait()
            text = read_ocr_text(
                context,
                controller.cached_image,
                "_make_target_clothes_quantity_sufficient_ocr",
                roi,
                expected,
                "make_target_clothes_batch_challenge",
            )
            if text:
                return True

            logger.info(
                "make_target_clothes_batch_challenge: 未识别到“数量已足够”，"
                f"尝试次数 {attempt + 1}/{retry}"
            )
            if attempt < retry - 1 and retry_delay > 0:
                time.sleep(retry_delay)

        return False

    def _read_round_values(
        self,
        context,
        controller,
        mode,
        stamina_roi,
        stamina_cost_roi,
        remaining_count_roi,
        retry,
        retry_delay,
    ):
        stamina = self._read_required_number(
            context=context,
            controller=controller,
            roi=stamina_roi,
            node_name="_make_target_clothes_batch_stamina_ocr",
            expected=[r"\d+\s*/\s*\d+"],
            retry=retry,
            retry_delay=retry_delay,
            label="体力",
        )
        if stamina is None:
            return None

        stamina_cost = self._read_required_number(
            context=context,
            controller=controller,
            roi=stamina_cost_roi,
            node_name="_make_target_clothes_batch_cost_ocr",
            expected=[r".*\d+"],
            retry=retry,
            retry_delay=retry_delay,
            label="单次体力消耗",
        )
        if stamina_cost is None:
            return None
        if stamina_cost <= 0:
            logger.error(
                f"make_target_clothes_batch_challenge: 单次体力消耗无效：{stamina_cost}"
            )
            return None

        remaining_count = None
        if mode == "heart_maze":
            remaining_count = self._read_required_number(
                context=context,
                controller=controller,
                roi=remaining_count_roi,
                node_name="_make_target_clothes_heart_maze_remaining_ocr",
                expected=[r".*\d+\s*/\s*\d+"],
                retry=retry,
                retry_delay=retry_delay,
                label="心灵迷宫剩余挑战次数",
            )
            if remaining_count is None:
                return None

        return stamina, stamina_cost, remaining_count

    def _read_required_number(
        self,
        context,
        controller,
        roi,
        node_name,
        expected,
        retry,
        retry_delay,
        label,
    ):
        for attempt in range(retry):
            controller.post_screencap().wait()
            number = read_ocr_number(
                context,
                controller.cached_image,
                node_name,
                roi,
                expected,
                "make_target_clothes_batch_challenge",
            )
            if number is not None:
                return number

            logger.warning(
                "make_target_clothes_batch_challenge: OCR 识别失败："
                f"{label} 尝试次数 {attempt + 1}/{retry}"
            )
            if attempt < retry - 1 and retry_delay > 0:
                time.sleep(retry_delay)

        logger.error(
            f"make_target_clothes_batch_challenge: OCR 识别失败：{label}"
        )
        return None

    def _click_roi(self, controller, roi):
        x, y, w, h = roi
        controller.post_click(x + w // 2, y + h // 2).wait()

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


@AgentServer.custom_action("make_target_clothes_recheck_current")
class MakeTargetClothesRecheckCurrentAction(CustomAction):

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}

        blank_roi = params.get("blank_roi", [304, 144, 71, 36])
        click_interval = self._as_float(params.get("click_interval"), 0.5)
        post_delay = self._as_float(params.get("post_delay"), 1.0)
        if not isinstance(blank_roi, list) or len(blank_roi) != 4:
            logger.error("make_target_clothes_recheck_current: 空白点击区域 blank_roi 无效")
            return False
        if click_interval < 0 or post_delay < 0:
            logger.error("make_target_clothes_recheck_current: 延迟参数无效")
            return False

        controller = context.tasker.controller
        x, y, w, h = blank_roi
        click_x = x + w // 2
        click_y = y + h // 2
        controller.post_click(click_x, click_y).wait()
        if click_interval > 0:
            time.sleep(click_interval)
        controller.post_click(click_x, click_y).wait()
        if post_delay > 0:
            time.sleep(post_delay)
        return True

    def _as_float(self, value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


@AgentServer.custom_action("make_target_clothes_batch_finish")
class MakeTargetClothesBatchFinishAction(CustomAction):

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        mode = params.get("mode")
        if mode not in ("heart_maze", "main_story"):
            logger.error(f"make_target_clothes_batch_finish: 模式参数无效：{mode}")
            return False

        MakeTargetClothesBatchChallengeAction.clear_task_state(
            argv.task_detail.task_id,
            mode,
        )
        return True


@AgentServer.custom_action("make_target_clothes_reset_page")
class MakeTargetClothesResetPageAction(CustomAction):

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}

        begin = params.get("begin", [314, 823])
        end = params.get("end", [630, 823])
        duration = self._as_int(params.get("duration"), 500)
        end_hold = self._as_int(params.get("end_hold"), 200)
        wait_after_swipe = self._as_float(params.get("wait_after_swipe"), 0.5)
        pixel_threshold = self._as_int(params.get("pixel_threshold"), 25)
        change_threshold = self._as_float(params.get("change_threshold"), 0.01)
        compare_roi = params.get("compare_roi")
        max_swipes = self._as_int(params.get("max_swipes"), 30)

        if not self._valid_point(begin) or not self._valid_point(end):
            logger.error("make_target_clothes_reset_page: 滑动起点或终点无效")
            return False
        if max_swipes <= 0:
            logger.error("make_target_clothes_reset_page: 最大滑动次数参数无效")
            return False

        controller = context.tasker.controller
        for swipe_index in range(max_swipes):
            before = self._screencap(controller)
            self._swipe_with_end_hold(context, begin, end, duration, end_hold)
            time.sleep(wait_after_swipe)
            after = self._screencap(controller)

            changed = self._image_changed(
                before,
                after,
                compare_roi=compare_roi,
                pixel_threshold=pixel_threshold,
                change_threshold=change_threshold,
            )
            logger.info(
                "make_target_clothes_reset_page: "
                f"滑动次数={swipe_index + 1}，画面是否变化={'是' if changed else '否'}"
            )
            if changed:
                continue

            return True

        logger.error(
            f"make_target_clothes_reset_page: 已超过最大滑动次数 {max_swipes}"
        )
        return False

    def _screencap(self, controller):
        controller.post_screencap().wait()
        return controller.cached_image.copy()

    def _swipe_with_end_hold(self, context, begin, end, duration, end_hold):
        result = context.run_action_direct(
            JActionType.Swipe,
            JSwipe(
                begin=begin,
                end=end,
                duration=duration,
                end_hold=end_hold,
            ),
        )
        if not result or not result.success:
            raise RuntimeError("Swipe action failed")

    def _image_changed(self, before, after, compare_roi, pixel_threshold, change_threshold):
        if compare_roi:
            x, y, w, h = compare_roi
            before = before[y:y+h, x:x+w]
            after = after[y:y+h, x:x+w]

        if before.shape != after.shape or before.size == 0:
            return True

        diff = np.abs(after.astype(np.int16) - before.astype(np.int16))
        changed_pixels = np.any(diff > pixel_threshold, axis=2)
        changed_ratio = np.count_nonzero(changed_pixels) / changed_pixels.size

        logger.info(
            "make_target_clothes_reset_page: "
            f"画面变化比例={changed_ratio:.4f}，判定阈值={change_threshold}"
        )
        return changed_ratio >= change_threshold

    def _valid_point(self, value):
        return isinstance(value, list) and len(value) == 2

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


@AgentServer.custom_action("make_target_clothes_swipe_page")
class MakeTargetClothesSwipePageAction(CustomAction):

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}

        begin = params.get("begin", [630, 823])
        end = params.get("end", [314, 823])
        changed_next = params.get("changed_next", ["点击全部材料"])
        unchanged_next = params.get("unchanged_next", ["点击末项全部材料-时空回廊"])
        final_challenge_node = params.get("final_challenge_node", "计算并挑战时空回廊制衣材料")
        final_challenge_next = params.get("final_challenge_next", ["返回主页面"])

        duration = self._as_int(params.get("duration"), 300)
        end_hold = self._as_int(params.get("end_hold"), 400)
        wait_after_swipe = self._as_float(params.get("wait_after_swipe"), 1.0)
        pixel_threshold = self._as_int(params.get("pixel_threshold"), 25)
        change_threshold = self._as_float(params.get("change_threshold"), 0.01)
        compare_roi = params.get("compare_roi")

        if not self._valid_point(begin) or not self._valid_point(end):
            logger.error("make_target_clothes_swipe_page: 滑动起点或终点无效")
            return False
        controller = context.tasker.controller

        before = self._screencap(controller)
        self._swipe_with_end_hold(context, begin, end, duration, end_hold)
        time.sleep(wait_after_swipe)
        after = self._screencap(controller)

        changed = self._image_changed(
            before,
            after,
            compare_roi=compare_roi,
            pixel_threshold=pixel_threshold,
            change_threshold=change_threshold,
        )

        logger.info(f"make_target_clothes_swipe_page: 画面是否变化={'是' if changed else '否'}")

        if changed:
            context.override_next(argv.node_name, changed_next)
            return True

        context.override_next(argv.node_name, unchanged_next)
        context.override_next(final_challenge_node, final_challenge_next)
        return True

    def _screencap(self, controller):
        controller.post_screencap().wait()
        return controller.cached_image.copy()

    def _swipe_with_end_hold(self, context, begin, end, duration, end_hold):
        result = context.run_action_direct(
            JActionType.Swipe,
            JSwipe(
                begin=begin,
                end=end,
                duration=duration,
                end_hold=end_hold,
            ),
        )
        if not result or not result.success:
            raise RuntimeError("Swipe action failed")

    def _image_changed(self, before, after, compare_roi, pixel_threshold, change_threshold):
        if compare_roi:
            x, y, w, h = compare_roi
            before = before[y:y+h, x:x+w]
            after = after[y:y+h, x:x+w]

        if before.shape != after.shape or before.size == 0:
            return True

        diff = np.abs(after.astype(np.int16) - before.astype(np.int16))
        changed_pixels = np.any(diff > pixel_threshold, axis=2)
        changed_ratio = np.count_nonzero(changed_pixels) / changed_pixels.size

        logger.info(
            "make_target_clothes_swipe_page: "
            f"画面变化比例={changed_ratio:.4f}，判定阈值={change_threshold}"
        )
        return changed_ratio >= change_threshold

    def _valid_point(self, value):
        return isinstance(value, list) and len(value) == 2

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
