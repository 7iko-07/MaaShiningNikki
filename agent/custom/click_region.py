import json
import time
import random
import numpy as np
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


@AgentServer.custom_action("click_region")
class ClickRegionAction(CustomAction):

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}

        roi = params.get("roi")
        if roi is None:
            box = argv.box
            if box.w > 0 and box.h > 0:
                roi = [box.x, box.y, box.w, box.h]
            else:
                return False

        x, y, w, h = roi
        step = params.get("step", 0)
        step_x = params.get("step_x", step or 50)
        step_y = params.get("step_y", step or 50)
        if step_x <= 0 or step_y <= 0:
            return False

        # top_to_bottom: 逐列自上向下；left_to_right: 逐行从左向右。
        direction = params.get("direction", "top_to_bottom")
        if direction not in ("top_to_bottom", "left_to_right"):
            return False

        delay = params.get("delay", 200)
        use_random = params.get("random", False)
        max_points = params.get("max_points", 200)
        exclude = params.get("exclude", [])
        stop_on_no_change = params.get("stop_on_no_change", False)

        params_change = params.get("change_detection", None)
        if params_change is None:
            enable_change = False
            change_threshold = 0.3
            color_distance = 30
        elif isinstance(params_change, bool):
            enable_change = params_change
            change_threshold = 0.3
            color_distance = 30
        else:
            enable_change = True
            change_threshold = params_change.get("threshold", 0.3)
            color_distance = params_change.get("color_distance", 30)

        points = []
        cx = x
        while cx < x + w:
            cy = y
            while cy < y + h:
                skip = False
                for ex in exclude:
                    ex_x, ex_y, ex_w, ex_h = ex
                    if ex_x <= cx < ex_x + ex_w and ex_y <= cy < ex_y + ex_h:
                        skip = True
                        break
                if not skip:
                    points.append((cx, cy))
                cy += step_y
            cx += step_x

        if direction == "left_to_right":
            points.sort(key=lambda point: (point[1], point[0]))

        if use_random:
            random.shuffle(points)

        controller = context.tasker.controller

        ref_img = None
        any_change = False
        if enable_change:
            job = controller.post_screencap()
            job.wait()
            ref_img = controller.cached_image[y:y+h, x:x+w].copy()

        for i, (px, py) in enumerate(points):
            if i >= max_points:
                break

            job = controller.post_click(px, py)
            job.wait()
            if delay > 0:
                time.sleep(delay / 1000.0)

            if enable_change and ref_img is not None:
                job = controller.post_screencap()
                job.wait()
                cur_img = controller.cached_image[y:y+h, x:x+w]
                diff = np.abs(cur_img.astype(np.int16) - ref_img.astype(np.int16))
                changed = np.any(diff > color_distance, axis=2)
                ratio = np.count_nonzero(changed) / changed.size
                if ratio >= change_threshold:
                    any_change = True
                    break

        if stop_on_no_change and enable_change and not any_change:
            context.override_next(argv.node_name, [])

        return True
