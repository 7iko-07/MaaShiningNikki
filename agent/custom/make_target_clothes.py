import json
import time

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JActionType, JSwipe
from utils import logger, read_ocr_number


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
                "make_target_clothes_challenge: missing required params "
                "(challenge_count_roi, stamina_roi, once_button, multi_button)"
            )
            return False

        stamina_cost = self._as_int(params.get("stamina_cost"), 5)
        max_multi = self._as_int(params.get("max_multi"), 10)
        retry = self._as_int(params.get("retry"), 3)
        retry_delay = self._as_float(params.get("retry_delay"), 0.5)
        click_delay = self._as_float(params.get("click_delay"), 1.2)
        popup_close_roi = params.get("popup_close_roi", [221, 137, 137, 78])
        popup_close_delay = self._as_float(params.get("popup_close_delay"), 2.0)

        if stamina_cost <= 0 or max_multi <= 1:
            logger.error("make_target_clothes_challenge: invalid stamina_cost or max_multi")
            return False

        controller = context.tasker.controller
        challenge_count = None
        stamina = None

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
                "make_target_clothes_challenge: OCR failed on attempt "
                f"{attempt + 1}/{retry}, challenge_count={challenge_count}, stamina={stamina}"
            )
            time.sleep(retry_delay)

        if challenge_count is None:
            logger.error("make_target_clothes_challenge: failed to OCR challenge count")
            return False

        if stamina is None:
            logger.warning("make_target_clothes_challenge: failed to OCR stamina, assuming stamina is enough")
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
            f"challenge_count={challenge_count}, stamina={stamina}, "
            f"challenge_times={challenge_times}, multi_clicks={multi_clicks}, once_clicks={once_clicks}"
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
        final_click_roi = params.get("final_click_roi", [539, 1106, 128, 39])
        changed_next = params.get("changed_next", ["点击全部材料"])
        unchanged_next = params.get("unchanged_next", ["点击前往获取时空回廊制衣材料"])
        final_challenge_node = params.get("final_challenge_node", "计算并挑战时空回廊制衣材料")
        final_challenge_next = params.get("final_challenge_next", ["返回主页面"])

        duration = self._as_int(params.get("duration"), 300)
        end_hold = self._as_int(params.get("end_hold"), 400)
        wait_after_swipe = self._as_float(params.get("wait_after_swipe"), 1.0)
        pixel_threshold = self._as_int(params.get("pixel_threshold"), 25)
        change_threshold = self._as_float(params.get("change_threshold"), 0.01)
        compare_roi = params.get("compare_roi")

        if not self._valid_point(begin) or not self._valid_point(end):
            logger.error("make_target_clothes_swipe_page: invalid begin or end")
            return False
        if not self._valid_roi(final_click_roi):
            logger.error("make_target_clothes_swipe_page: invalid final_click_roi")
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

        logger.info(f"make_target_clothes_swipe_page: changed={changed}")

        if changed:
            context.override_next(argv.node_name, changed_next)
            return True

        self._click_roi(controller, final_click_roi)
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
            f"changed_ratio={changed_ratio:.4f}, threshold={change_threshold}"
        )
        return changed_ratio >= change_threshold

    def _click_roi(self, controller, roi):
        x, y, w, h = roi
        controller.post_click(x + w // 2, y + h // 2).wait()

    def _valid_point(self, value):
        return isinstance(value, list) and len(value) == 2

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
