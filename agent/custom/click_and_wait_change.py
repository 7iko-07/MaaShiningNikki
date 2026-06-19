import json
import time

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils import logger


@AgentServer.custom_action("click_and_wait_change")
class ClickAndWaitChangeAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}

        target = self._get_target(params, argv)
        if target is None:
            logger.error("click_and_wait_change 缺少 target，且当前识别框无效")
            return False

        controller = context.tasker.controller
        before = self._screencap(controller)
        click_x, click_y = self._target_center(target)

        controller.post_click(click_x, click_y).wait()
        time.sleep(params.get("after_click_delay", 300) / 1000.0)

        compare_roi = params.get("compare_roi", [0, 0, 720, 1280])
        pixel_threshold = params.get("pixel_threshold", params.get("color_distance", 30))
        change_threshold = params.get("change_threshold", params.get("threshold", 0.3))
        timeout = params.get("timeout", params.get("wait_timeout", 5000))
        interval = params.get("interval", params.get("wait_interval", 300))
        max_retries = params.get("max_retries", params.get("max_clicks", 3))
        fail_on_timeout = params.get("fail_on_timeout", False)

        for retry_index in range(max_retries):
            deadline = time.monotonic() + timeout / 1000.0
            last_ratio = 0.0
            while True:
                after = self._screencap(controller)
                changed, last_ratio = self._image_changed(
                    before,
                    after,
                    compare_roi,
                    pixel_threshold,
                    change_threshold,
                )
                if changed:
                    logger.info(f"click_and_wait_change 检测到画面变化，变化比例: {last_ratio:.6f}")
                    return True

                if time.monotonic() >= deadline:
                    break

                time.sleep(interval / 1000.0)

            if retry_index >= max_retries - 1:
                logger.warning(
                    f"click_and_wait_change 重试 {max_retries} 次后画面仍未变化，变化比例: {last_ratio:.6f}"
                )
                return not fail_on_timeout

            logger.warning(
                f"click_and_wait_change 等待超时，画面未变化，准备第 {retry_index + 2} 次点击，"
                f"变化比例: {last_ratio:.6f}"
            )
            before = after.copy()
            controller.post_click(click_x, click_y).wait()
            time.sleep(params.get("after_click_delay", 300) / 1000.0)

        return not fail_on_timeout

    def _get_target(self, params, argv):
        target = params.get("target")
        if target is not None:
            return target

        box = argv.box
        if box.w > 0 and box.h > 0:
            return [box.x, box.y, box.w, box.h]
        return None

    def _target_center(self, target):
        if len(target) == 2:
            return int(target[0]), int(target[1])
        if len(target) >= 4:
            return int(target[0] + target[2] / 2), int(target[1] + target[3] / 2)
        raise ValueError("target 必须是 [x, y] 或 [x, y, w, h]")

    def _screencap(self, controller):
        controller.post_screencap().wait()
        return controller.cached_image.copy()

    def _image_changed(self, before, after, compare_roi, pixel_threshold, change_threshold):
        if compare_roi:
            x, y, w, h = compare_roi
            before = before[y:y + h, x:x + w]
            after = after[y:y + h, x:x + w]

        if before.size == 0 or after.size == 0:
            return False, 0.0

        diff = np.abs(after.astype(np.int16) - before.astype(np.int16))
        changed_pixels = np.any(diff > pixel_threshold, axis=2)
        ratio = np.count_nonzero(changed_pixels) / changed_pixels.size
        return ratio >= change_threshold, ratio
