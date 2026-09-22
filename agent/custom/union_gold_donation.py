import json
import time

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils import logger


@AgentServer.custom_action("union_gold_donation")
class UnionGoldDonationAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}

        target = params.get("target")
        if target is None:
            box = argv.box
            if box.w <= 0 or box.h <= 0:
                logger.error("union_gold_donation: 缺少有效的金币按钮点击区域")
                return False
            target = [box.x, box.y, box.w, box.h]

        max_clicks = max(1, int(params.get("max_clicks", 12)))
        compare_roi = params.get("compare_roi", [80, 540, 560, 180])
        pixel_threshold = int(params.get("pixel_threshold", 30))
        change_threshold = float(params.get("change_threshold", 0.12))
        timeout = max(100, int(params.get("timeout", 3000)))
        interval = max(50, int(params.get("interval", 150)))
        after_click_delay = max(0, int(params.get("after_click_delay", 100)))
        limit_node = params.get("limit_node", "联盟金币捐献达到上限")
        insufficient_node = params.get("insufficient_node", "点击体力用尽确定")

        controller = context.tasker.controller
        click_x, click_y = self._target_center(target)

        for click_index in range(max_clicks):
            before = self._screencap(controller)
            stop_reason = self._stop_reason(
                context,
                before,
                limit_node=limit_node,
                insufficient_node=insufficient_node,
            )
            if stop_reason:
                return self._finish_stop(context, stop_reason, insufficient_node)

            controller.post_click(click_x, click_y).wait()
            if after_click_delay:
                time.sleep(after_click_delay / 1000.0)

            after, changed, ratio = self._wait_change(
                controller,
                before,
                compare_roi=compare_roi,
                pixel_threshold=pixel_threshold,
                change_threshold=change_threshold,
                timeout=timeout,
                interval=interval,
            )

            stop_reason = self._stop_reason(
                context,
                after,
                limit_node=limit_node,
                insufficient_node=insufficient_node,
            )
            if stop_reason:
                return self._finish_stop(context, stop_reason, insufficient_node)

            if changed:
                logger.info(
                    "union_gold_donation: "
                    f"第 {click_index + 1}/{max_clicks} 次点击检测到画面变化，变化比例: {ratio:.6f}"
                )
            else:
                logger.warning(
                    "union_gold_donation: "
                    f"第 {click_index + 1}/{max_clicks} 次点击等待画面变化超时，变化比例: {ratio:.6f}"
                )

        final_image = self._screencap(controller)
        stop_reason = self._stop_reason(
            context,
            final_image,
            limit_node=limit_node,
            insufficient_node=insufficient_node,
        )
        if stop_reason:
            return self._finish_stop(context, stop_reason, insufficient_node)

        logger.warning(
            f"union_gold_donation: 已达到安全点击上限 {max_clicks}，"
            "仍未识别到捐献上限或体力/金币不足，停止捐献并继续联盟福利"
        )
        return True

    def _wait_change(
        self,
        controller,
        before,
        compare_roi,
        pixel_threshold,
        change_threshold,
        timeout,
        interval,
    ):
        deadline = time.monotonic() + timeout / 1000.0
        last_image = before
        last_ratio = 0.0

        while True:
            last_image = self._screencap(controller)
            changed, last_ratio = self._image_changed(
                before,
                last_image,
                compare_roi,
                pixel_threshold,
                change_threshold,
            )
            if changed or time.monotonic() >= deadline:
                return last_image, changed, last_ratio
            time.sleep(interval / 1000.0)

    def _stop_reason(self, context, image, limit_node, insufficient_node):
        if self._recognize(context, image, insufficient_node):
            return "insufficient"
        if self._recognize(context, image, limit_node):
            return "limit"
        return ""

    def _finish_stop(self, context, stop_reason, insufficient_node):
        if stop_reason == "insufficient":
            logger.info("union_gold_donation: 检测到体力或金币不足，点击取消并停止捐献")
            result = context.run_task(insufficient_node)
            if not self._task_succeeded(result):
                logger.warning("union_gold_donation: 体力或金币不足弹窗取消失败，仍停止捐献")
            return True

        logger.info("union_gold_donation: 检测到当前档位捐献已达上限，停止捐献")
        return True

    def _recognize(self, context: Context, image, node_name):
        try:
            result = context.run_recognition(node_name, image)
            return bool(result and getattr(result, "hit", False))
        except Exception as e:
            logger.warning(f"union_gold_donation: 识别 {node_name!r} 失败: {e}")
            return False

    def _task_succeeded(self, result):
        if result is None:
            return False
        success = getattr(result, "success", None)
        if success is not None:
            return bool(success)
        return bool(getattr(getattr(result, "status", None), "succeeded", False))

    def _screencap(self, controller):
        controller.post_screencap().wait()
        return controller.cached_image.copy()

    def _target_center(self, target):
        if len(target) == 2:
            return int(target[0]), int(target[1])
        if len(target) == 4:
            return int(target[0] + target[2] / 2), int(target[1] + target[3] / 2)
        raise ValueError("union_gold_donation: target 必须是 [x, y] 或 [x, y, w, h]")

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
