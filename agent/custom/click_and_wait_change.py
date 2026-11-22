import json
import time

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils import logger


@AgentServer.custom_action("click_and_wait_change")
class ClickAndWaitChangeAction(CustomAction):
    SELF_TARGET_NAMES = {
        "self",
        "current",
        "current_roi",
        "current_box",
        "roi",
        "box",
        "hit",
        "true",
        "自身",
        "本节点",
        "当前",
        "当前节点",
        "当前识别",
        "当前roi",
        "当前box",
    }

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}

        target = self._get_target(params, argv)
        if target is None:
            logger.debug("click_and_wait_change 缺少 target，且当前识别框无效")
            logger.error("无法定位点击目标")
            return False

        if params.get("ready_nodes"):
            return self._wait_ready(context, params, target)

        controller = context.tasker.controller
        before = self._screencap(controller)
        initial_image = before
        click_x, click_y = self._target_center(target)

        guarded = self._before_click(context, argv)
        if guarded is not None:
            return guarded
        controller.post_click(click_x, click_y).wait()
        time.sleep(params.get("after_click_delay", 300) / 1000.0)

        compare_roi = params.get("compare_roi", [0, 0, 720, 1280])
        pixel_threshold = params.get("pixel_threshold", params.get("color_distance", 30))
        change_threshold = params.get("change_threshold", params.get("threshold", 0.3))
        timeout = params.get("timeout", params.get("wait_timeout", 1000))
        interval = params.get("interval", params.get("wait_interval", 300))
        max_retries = params.get("max_retries", params.get("max_clicks", 8))
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
                    logger.debug(f"click_and_wait_change 检测到画面变化，变化比例: {last_ratio:.6f}")
                    return True
                if self._additional_image_change(context, initial_image, after):
                    return True

                if time.monotonic() >= deadline:
                    break

                time.sleep(interval / 1000.0)

            if retry_index >= max_retries - 1:
                logger.debug(
                    f"click_and_wait_change 重试 {max_retries} 次后画面仍未变化，变化比例: {last_ratio:.6f}"
                )
                logger.warning("多次点击后画面仍未变化")
                return not fail_on_timeout

            logger.debug(
                f"click_and_wait_change 等待超时，画面未变化，准备第 {retry_index + 2} 次点击，"
            )
            logger.debug(f"变化比例: {last_ratio:.6f}")
            before = after.copy()
            guarded = self._before_click(context, argv)
            if guarded is not None:
                return guarded
            controller.post_click(click_x, click_y).wait()
            time.sleep(params.get("after_click_delay", 300) / 1000.0)

        return not fail_on_timeout

    def _wait_ready(self, context, params, target):
        controller = context.tasker.controller
        probe = context.clone()
        popup_node = params.get("popup_node")
        click_interval = params.get("click_interval", 500) / 1000.0
        click_x, click_y = self._target_center(target)
        deadline = time.monotonic() + params.get("timeout", 30000) / 1000.0
        next_click = 0.0
        if not popup_node:
            if context.tasker.stopping:
                return False
            if not controller.post_click(click_x, click_y).wait().succeeded:
                logger.error("页面切换：点击按钮失败")
                return False
            next_click = time.monotonic() + click_interval

        while time.monotonic() < deadline:
            if context.tasker.stopping:
                return False
            capture = controller.post_screencap().wait()
            image = controller.cached_image if capture.succeeded else None
            if image is None:
                logger.error("页面切换：获取画面失败")
                return False
            popup_visible = False
            if popup_node:
                popup = probe.run_recognition(popup_node, image)
                if popup is None:
                    logger.error("页面切换：无法识别弹窗")
                    return False
                popup_visible = popup.hit
            ready = False
            if not popup_visible:
                for node in params["ready_nodes"]:
                    # 就绪检测不受任务勾选影响，覆盖仅留在识别用的独立 Context。
                    result = probe.run_recognition(node, image, pipeline_override={node: {"enabled": True}})
                    if result is None:
                        logger.error("页面切换：无法识别目标页面")
                        return False
                    if result.hit:
                        ready = True
                        break
            if context.tasker.stopping:
                return False
            if time.monotonic() >= deadline:
                break
            if ready:
                return True
            if (not popup_node or popup_visible) and time.monotonic() >= next_click:
                if not controller.post_click(click_x, click_y).wait().succeeded:
                    logger.error("页面切换：点击空白失败" if popup_node else "页面切换：点击按钮失败")
                    return False
                next_click = time.monotonic() + click_interval
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))
        logger.error("页面切换：等待无弹窗页面超时" if popup_node else "页面切换：等待目标页面超时")
        return False

    def _before_click(self, context, argv):
        return None

    def _additional_image_change(self, context, before, after):
        return False

    def _get_target(self, params, argv):
        target = params.get("target")
        if target is None:
            return self._argv_box_target(argv)

        if self._is_self_target(target):
            return self._argv_box_target(argv)

        if isinstance(target, str):
            logger.debug(
                "click_and_wait_change: custom_action_param.target 不支持节点名；"
                "如需引用当前节点或其他节点的识别框，请把 target 写在 Custom action 自身字段中"
            )
            logger.error("点击目标配置无效")
            return None

        if self._valid_target(target):
            return target

        logger.debug("click_and_wait_change: target 必须是 true/self、[x, y] 或 [x, y, w, h]")
        logger.error("点击目标配置无效")
        return None

    def _argv_box_target(self, argv):
        box = argv.box
        if box.w > 0 and box.h > 0:
            return [box.x, box.y, box.w, box.h]
        return None

    def _is_self_target(self, target):
        if target is True:
            return True
        if not isinstance(target, str):
            return False
        return target.strip().lower() in self.SELF_TARGET_NAMES

    def _valid_target(self, target):
        return isinstance(target, (list, tuple)) and len(target) in (2, 4)

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
