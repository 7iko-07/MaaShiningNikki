import json
import re
import time

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JActionType, JOCR, JRecognitionType, JSwipe

from utils import logger, read_number_from_controller


RETURN_HOME_NEXT = ["主线挑战返回主页面"]


class _MainStorylineBase:
    def _params(self, argv):
        return json.loads(argv.custom_action_param) if argv.custom_action_param else {}

    def _screencap(self, controller):
        controller.post_screencap().wait()
        return controller.cached_image.copy()

    def _iter_ocr_results(self, result):
        if not result or not getattr(result, "hit", False):
            return

        yielded = set()
        for attr in ("filtered_results", "all_results"):
            for item in getattr(result, attr, []) or []:
                if item is None or id(item) in yielded:
                    continue
                yielded.add(id(item))
                yield item

        best_result = getattr(result, "best_result", None)
        if best_result is not None and id(best_result) not in yielded:
            yield best_result

    def _ocr_items(self, context, image, roi, threshold=0.3):
        result = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(roi=roi, threshold=threshold, order_by="Horizontal"),
            image,
        )
        return list(self._iter_ocr_results(result))

    def _box_list(self, box):
        if box is None:
            return None
        try:
            values = list(box)
        except TypeError:
            values = [box.x, box.y, box.w, box.h]
        if len(values) != 4:
            return None
        return [int(value) for value in values]

    def _click_box_and_wait_change(
        self,
        controller,
        box,
        compare_roi=None,
        pixel_threshold=30,
        change_threshold=0.08,
        timeout=5.0,
        interval=0.3,
        max_clicks=2,
    ):
        x, y, w, h = box
        click_x = x + w // 2
        click_y = y + h // 2
        before = self._screencap(controller)

        for click_index in range(max_clicks):
            controller.post_click(click_x, click_y).wait()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                time.sleep(interval)
                after = self._screencap(controller)
                changed, ratio = self._image_changed(
                    before,
                    after,
                    compare_roi,
                    pixel_threshold,
                    change_threshold,
                )
                if changed:
                    logger.info(
                        "main_storyline: click detected page change, "
                        f"ratio={ratio:.4f}"
                    )
                    return True
            logger.warning(
                "main_storyline: click did not change page, "
                f"attempt={click_index + 1}/{max_clicks}"
            )
            before = self._screencap(controller)

        return False

    def _swipe(self, context, begin, end, duration, end_hold):
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
            raise RuntimeError("main_storyline: swipe action failed")

    def _image_changed(
        self,
        before,
        after,
        compare_roi,
        pixel_threshold,
        change_threshold,
    ):
        if compare_roi:
            x, y, w, h = compare_roi
            before = before[y:y + h, x:x + w]
            after = after[y:y + h, x:x + w]

        if before.shape != after.shape or before.size == 0:
            return True, 1.0

        diff = np.abs(after.astype(np.int16) - before.astype(np.int16))
        changed_pixels = np.any(diff > pixel_threshold, axis=2)
        ratio = np.count_nonzero(changed_pixels) / changed_pixels.size
        return ratio >= change_threshold, ratio

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


@AgentServer.custom_action("main_storyline_select_chapter")
class MainStorylineSelectChapterAction(_MainStorylineBase, CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        params = self._params(argv)
        chapter_name = str(params.get("chapter_name", "")).strip()
        expected = params.get("expected") or []
        roi = params.get("roi", [190, 470, 500, 620])
        retry = max(1, self._as_int(params.get("retry"), 3))
        retry_delay = self._as_float(params.get("retry_delay"), 1.0)
        threshold = self._as_float(params.get("threshold"), 0.3)

        if isinstance(expected, str):
            expected = [expected]
        expected = [self._normalize_text(item) for item in expected if str(item).strip()]
        if not chapter_name or not expected:
            logger.error("main_storyline_select_chapter: missing chapter_name or expected")
            context.override_next(argv.node_name, RETURN_HOME_NEXT)
            return True

        controller = context.tasker.controller
        try:
            for attempt in range(retry):
                image = self._screencap(controller)
                for item in self._ocr_items(context, image, roi, threshold):
                    text = self._normalize_text(getattr(item, "text", ""))
                    box = self._box_list(getattr(item, "box", None))
                    if not text or not box:
                        continue
                    if not any(keyword in text for keyword in expected):
                        continue

                    logger.info(
                        "main_storyline_select_chapter: "
                        f"matched chapter={chapter_name!r}, text={text!r}, box={box}"
                    )
                    if self._click_box_and_wait_change(
                        controller,
                        box,
                        compare_roi=[0, 100, 720, 1100],
                        change_threshold=0.05,
                    ):
                        return True

                    logger.warning(
                        "main_storyline_select_chapter: chapter did not open; "
                        "it may still be locked"
                    )
                    context.override_next(argv.node_name, RETURN_HOME_NEXT)
                    return True

                if attempt < retry - 1:
                    time.sleep(retry_delay)

            logger.error(
                f"main_storyline_select_chapter: chapter not found: {chapter_name!r}"
            )
        except Exception as exc:
            logger.exception(f"main_storyline_select_chapter: {exc}")

        context.override_next(argv.node_name, RETURN_HOME_NEXT)
        return True

    def _normalize_text(self, text):
        return re.sub(r"[\s·•・,，。:：]", "", str(text or ""))


@AgentServer.custom_action("main_storyline_find_stage")
class MainStorylineFindStageAction(_MainStorylineBase, CustomAction):
    DASHES = "‐‑‒–—―﹘﹣－"

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        params = self._params(argv)
        chapter_number = self._as_int(params.get("chapter_number"), 0)
        stage_suffix = self._as_int(params.get("stage_suffix"), 0)
        roi = params.get("roi", [0, 105, 720, 1045])
        reset_begin = params.get("reset_begin", [61, 500])
        reset_end = params.get("reset_end", [698, 500])
        scan_begin = params.get("scan_begin", [698, 500])
        scan_end = params.get("scan_end", [61, 500])
        duration = self._as_int(params.get("duration"), 6000)
        end_hold = self._as_int(params.get("end_hold"), 2000)
        wait_after_swipe = self._as_float(params.get("wait_after_swipe"), 1.0)
        rounds = max(1, self._as_int(params.get("rounds"), 3))
        reset_swipes = max(1, self._as_int(params.get("reset_swipes"), 3))
        max_scan_swipes = max(1, self._as_int(params.get("max_scan_swipes"), 30))
        pixel_threshold = self._as_int(params.get("pixel_threshold"), 25)
        change_threshold = self._as_float(params.get("change_threshold"), 0.015)
        compare_roi = params.get("compare_roi", [0, 180, 720, 850])
        threshold = self._as_float(params.get("threshold"), 0.3)
        panel_title_roi = params.get("panel_title_roi", [26, 332, 271, 107])
        non_main_title = str(params.get("non_main_title", "挑战关卡")).strip()
        panel_retry = max(1, self._as_int(params.get("panel_retry"), 3))
        panel_retry_delay = self._as_float(params.get("panel_retry_delay"), 1.0)
        panel_load_delay = self._as_float(params.get("panel_load_delay"), 1.0)
        panel_close_roi = params.get("panel_close_roi", [320, 99, 32, 24])
        panel_close_clicks = max(1, self._as_int(params.get("panel_close_clicks"), 2))
        panel_close_delay = self._as_float(params.get("panel_close_delay"), 0.5)
        required_panel_title = str(params.get("required_panel_title", "")).strip()

        if not 1 <= chapter_number <= 99 or not 1 <= stage_suffix <= 99:
            logger.error(
                "main_storyline_find_stage: invalid stage, "
                f"chapter={chapter_number}, suffix={stage_suffix}"
            )
            context.override_next(argv.node_name, RETURN_HOME_NEXT)
            return True

        target = f"{chapter_number}-{stage_suffix}"
        controller = context.tasker.controller

        if required_panel_title:
            return self._scan_for_required_panel(
                context=context,
                controller=controller,
                roi=roi,
                target=target,
                threshold=threshold,
                scan_begin=scan_begin,
                scan_end=scan_end,
                duration=duration,
                end_hold=end_hold,
                wait_after_swipe=wait_after_swipe,
                max_scan_swipes=max_scan_swipes,
                compare_roi=compare_roi,
                pixel_threshold=pixel_threshold,
                change_threshold=change_threshold,
                panel_title_roi=panel_title_roi,
                required_panel_title=required_panel_title,
                panel_retry=panel_retry,
                panel_retry_delay=panel_retry_delay,
                panel_load_delay=panel_load_delay,
                panel_close_roi=panel_close_roi,
                panel_close_clicks=panel_close_clicks,
                panel_close_delay=panel_close_delay,
            )

        try:
            skip_match_after_reset = False
            list_is_at_front = False
            match = self._find_stage(context, controller, roi, target, threshold)
            if match and self._open_stage(controller, match, target):
                if not self._is_non_main_panel(
                    context,
                    controller,
                    panel_title_roi,
                    non_main_title,
                    threshold,
                    panel_retry,
                    panel_retry_delay,
                    panel_load_delay,
                ):
                    return True
                self._close_non_main_panel(
                    controller,
                    panel_close_roi,
                    panel_close_clicks,
                    panel_close_delay,
                    target,
                )
                self._reset_stage_list(
                    context,
                    reset_begin,
                    reset_end,
                    duration,
                    end_hold,
                    wait_after_swipe,
                    reset_swipes,
                )
                skip_match_after_reset = True
                list_is_at_front = True

            for round_index in range(rounds):
                logger.info(
                    "main_storyline_find_stage: "
                    f"starting search round {round_index + 1}/{rounds} for {target}"
                )
                if not list_is_at_front:
                    self._reset_stage_list(
                        context,
                        reset_begin,
                        reset_end,
                        duration,
                        end_hold,
                        wait_after_swipe,
                        reset_swipes,
                    )
                list_is_at_front = False

                if not skip_match_after_reset:
                    match = self._find_stage(context, controller, roi, target, threshold)
                    if match and self._open_stage(controller, match, target):
                        if not self._is_non_main_panel(
                            context,
                            controller,
                            panel_title_roi,
                            non_main_title,
                            threshold,
                            panel_retry,
                            panel_retry_delay,
                            panel_load_delay,
                        ):
                            return True
                        self._close_non_main_panel(
                            controller,
                            panel_close_roi,
                            panel_close_clicks,
                            panel_close_delay,
                            target,
                        )
                        self._reset_stage_list(
                            context,
                            reset_begin,
                            reset_end,
                            duration,
                            end_hold,
                            wait_after_swipe,
                            reset_swipes,
                        )
                        skip_match_after_reset = True
                        list_is_at_front = True
                        continue

                skip_match_after_reset = False

                for swipe_index in range(max_scan_swipes):
                    before = self._screencap(controller)
                    self._swipe(context, scan_begin, scan_end, duration, end_hold)
                    time.sleep(wait_after_swipe)
                    after = self._screencap(controller)

                    match = self._find_stage(
                        context,
                        controller,
                        roi,
                        target,
                        threshold,
                        image=after,
                    )
                    if match and self._open_stage(controller, match, target):
                        if not self._is_non_main_panel(
                            context,
                            controller,
                            panel_title_roi,
                            non_main_title,
                            threshold,
                            panel_retry,
                            panel_retry_delay,
                            panel_load_delay,
                        ):
                            return True
                        self._close_non_main_panel(
                            controller,
                            panel_close_roi,
                            panel_close_clicks,
                            panel_close_delay,
                            target,
                        )
                        self._reset_stage_list(
                            context,
                            reset_begin,
                            reset_end,
                            duration,
                            end_hold,
                            wait_after_swipe,
                            reset_swipes,
                        )
                        skip_match_after_reset = True
                        list_is_at_front = True
                        break

                    changed, ratio = self._image_changed(
                        before,
                        after,
                        compare_roi,
                        pixel_threshold,
                        change_threshold,
                    )
                    logger.info(
                        "main_storyline_find_stage: "
                        f"round={round_index + 1}, swipe={swipe_index + 1}, "
                        f"changed={changed}, ratio={ratio:.4f}"
                    )
                    if not changed:
                        break

            logger.error(
                "main_storyline_find_stage: target not found after three reset scans: "
                f"{target}"
            )
        except Exception as exc:
            logger.exception(f"main_storyline_find_stage: {exc}")

        context.override_next(argv.node_name, RETURN_HOME_NEXT)
        return True

    def _scan_for_required_panel(
        self,
        context,
        controller,
        roi,
        target,
        threshold,
        scan_begin,
        scan_end,
        duration,
        end_hold,
        wait_after_swipe,
        max_scan_swipes,
        compare_roi,
        pixel_threshold,
        change_threshold,
        panel_title_roi,
        required_panel_title,
        panel_retry,
        panel_retry_delay,
        panel_load_delay,
        panel_close_roi,
        panel_close_clicks,
        panel_close_delay,
    ):
        try:
            for swipe_index in range(max_scan_swipes):
                before = self._screencap(controller)
                self._swipe(context, scan_begin, scan_end, duration, end_hold)
                if wait_after_swipe > 0:
                    time.sleep(wait_after_swipe)
                after = self._screencap(controller)

                match = self._find_stage(
                    context,
                    controller,
                    roi,
                    target,
                    threshold,
                    image=after,
                )
                if match and self._open_stage(controller, match, target):
                    if self._panel_has_title(
                        context,
                        controller,
                        panel_title_roi,
                        required_panel_title,
                        threshold,
                        panel_retry,
                        panel_retry_delay,
                        panel_load_delay,
                    ):
                        logger.info(
                            "main_storyline_find_stage: required stage panel found, "
                            f"target={target}, title={required_panel_title!r}"
                        )
                        return True

                    logger.info(
                        "main_storyline_find_stage: candidate panel title did not match; "
                        f"target={target}, required={required_panel_title!r}"
                    )
                    self._close_non_main_panel(
                        controller,
                        panel_close_roi,
                        panel_close_clicks,
                        panel_close_delay,
                        target,
                    )
                    continue

                changed, ratio = self._image_changed(
                    before,
                    after,
                    compare_roi,
                    pixel_threshold,
                    change_threshold,
                )
                logger.info(
                    "main_storyline_find_stage: required-panel scan, "
                    f"swipe={swipe_index + 1}/{max_scan_swipes}, "
                    f"changed={changed}, ratio={ratio:.4f}"
                )
                if not changed:
                    logger.error(
                        "main_storyline_find_stage: stage list reached the end before "
                        f"finding title={required_panel_title!r} for target={target}"
                    )
                    return False

            logger.error(
                "main_storyline_find_stage: required stage panel not found after "
                f"{max_scan_swipes} swipes: target={target}, "
                f"title={required_panel_title!r}"
            )
        except Exception as exc:
            logger.exception(f"main_storyline_find_stage: required-panel scan failed: {exc}")

        return False

    def _find_stage(self, context, controller, roi, target, threshold, image=None):
        image = image if image is not None else self._screencap(controller)
        for item in self._ocr_items(context, image, roi, threshold):
            text = self._normalize_stage_text(getattr(item, "text", ""))
            box = self._box_list(getattr(item, "box", None))
            if not text or not box:
                continue
            candidates = [f"{left}-{right}" for left, right in re.findall(
                r"(?<!\d)(\d{1,2})-(\d{1,2})(?!\d)", text
            )]
            if target in candidates:
                logger.info(
                    "main_storyline_find_stage: "
                    f"matched target={target}, text={text!r}, box={box}"
                )
                return box
        return None

    def _open_stage(self, controller, box, target):
        if self._click_box_and_wait_change(
            controller,
            box,
            compare_roi=[0, 120, 720, 1080],
            change_threshold=0.06,
        ):
            return True
        logger.warning(f"main_storyline_find_stage: failed to open target {target}")
        return False

    def _is_non_main_panel(
        self,
        context,
        controller,
        roi,
        expected,
        threshold,
        retry,
        retry_delay,
        load_delay,
    ):
        matched = self._panel_has_title(
            context,
            controller,
            roi,
            expected,
            threshold,
            retry,
            retry_delay,
            load_delay,
        )
        if matched:
            logger.info("main_storyline_find_stage: non-main stage panel detected")
        else:
            logger.info(
                "main_storyline_find_stage: non-main title not detected; "
                "treating the current panel as the main-story quick-challenge panel"
            )
        return matched

    def _panel_has_title(
        self,
        context,
        controller,
        roi,
        expected,
        threshold,
        retry,
        retry_delay,
        load_delay,
    ):
        if load_delay > 0:
            time.sleep(load_delay)

        normalized_expected = self._normalize_panel_text(expected)
        for attempt in range(retry):
            image = self._screencap(controller)
            for item in self._ocr_items(context, image, roi, threshold):
                text = self._normalize_panel_text(getattr(item, "text", ""))
                if normalized_expected and normalized_expected in text:
                    logger.info(
                        "main_storyline_find_stage: stage panel title matched, "
                        f"text={text!r}, expected={normalized_expected!r}, roi={roi}"
                    )
                    return True
            if attempt < retry - 1 and retry_delay > 0:
                time.sleep(retry_delay)

        return False

    def _close_non_main_panel(self, controller, roi, clicks, delay, target):
        for click_index in range(clicks):
            self._click_roi(controller, roi)
            logger.info(
                "main_storyline_find_stage: closing non-main stage panel, "
                f"target={target}, click={click_index + 1}/{clicks}"
            )
            if click_index < clicks - 1 and delay > 0:
                time.sleep(delay)

    def _reset_stage_list(
        self,
        context,
        begin,
        end,
        duration,
        end_hold,
        wait_after_swipe,
        swipes,
    ):
        for swipe_index in range(swipes):
            self._swipe(context, begin, end, duration, end_hold)
            logger.info(
                "main_storyline_find_stage: resetting stage list to the front, "
                f"swipe={swipe_index + 1}/{swipes}"
            )
            if wait_after_swipe > 0:
                time.sleep(wait_after_swipe)

    def _normalize_panel_text(self, text):
        return re.sub(r"\s+", "", str(text or ""))

    def _normalize_stage_text(self, text):
        translation = str.maketrans({dash: "-" for dash in self.DASHES})
        return re.sub(r"\s+", "", str(text or "").translate(translation))


@AgentServer.custom_action("main_storyline_challenge")
class MainStorylineChallengeAction(_MainStorylineBase, CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        params = self._params(argv)
        max_challenges = self._as_int(params.get("max_challenges"), 0)
        stamina_roi = params.get("stamina_roi", [82, 0, 145, 40])
        stamina_cost_roi = params.get("stamina_cost_roi", [55, 755, 205, 52])
        once_button = params.get("once_button", [63, 806, 135, 67])
        multi_button = params.get("multi_button", [210, 806, 135, 67])
        popup_close_roi = params.get("popup_close_roi", [221, 137, 137, 78])
        max_multi = max(2, self._as_int(params.get("max_multi"), 10))
        retry = max(1, self._as_int(params.get("retry"), 3))
        retry_delay = self._as_float(params.get("retry_delay"), 1.5)
        load_delay = self._as_float(params.get("load_delay"), 3.0)
        result_delay = self._as_float(params.get("result_delay"), 5.0)
        popup_close_delay = self._as_float(params.get("popup_close_delay"), 2.0)
        max_rounds = max(1, self._as_int(params.get("max_rounds"), 200))

        if max_challenges < 0 or max_challenges > 100:
            logger.error(
                f"main_storyline_challenge: invalid max_challenges={max_challenges}"
            )
            return True

        if load_delay > 0:
            time.sleep(load_delay)

        controller = context.tasker.controller
        challenged = 0
        try:
            for round_index in range(max_rounds):
                stamina = read_number_from_controller(
                    context,
                    controller,
                    stamina_roi,
                    "_main_storyline_stamina_ocr",
                    [r"\d+\s*/\s*\d+"],
                    retry=retry,
                    retry_delay=retry_delay,
                    log_prefix="main_storyline_challenge",
                    retry_label="stamina",
                )
                stamina_cost = read_number_from_controller(
                    context,
                    controller,
                    stamina_cost_roi,
                    "_main_storyline_stamina_cost_ocr",
                    [r"\d+"],
                    retry=retry,
                    retry_delay=retry_delay,
                    log_prefix="main_storyline_challenge",
                    retry_label="stamina cost",
                )
                if stamina is None or stamina_cost is None or stamina_cost <= 0:
                    logger.error(
                        "main_storyline_challenge: quick-challenge panel was not recognized; "
                        "the stage may not support quick challenge"
                    )
                    return True

                possible_batch = min(max_multi, stamina // stamina_cost)
                allowance = None if max_challenges == 0 else max_challenges - challenged
                logger.info(
                    "main_storyline_challenge: "
                    f"round={round_index + 1}, stamina={stamina}, cost={stamina_cost}, "
                    f"challenged={challenged}, allowance={allowance}, "
                    f"possible_batch={possible_batch}"
                )

                if possible_batch <= 0 or (allowance is not None and allowance <= 0):
                    return True

                use_multi = possible_batch > 1 and (
                    allowance is None or allowance >= possible_batch
                )
                button = multi_button if use_multi else once_button
                before_stamina = stamina
                self._click_roi(controller, button)
                time.sleep(result_delay)
                self._click_roi(controller, popup_close_roi)
                time.sleep(popup_close_delay)

                new_stamina = read_number_from_controller(
                    context,
                    controller,
                    stamina_roi,
                    "_main_storyline_stamina_after_ocr",
                    [r"\d+\s*/\s*\d+"],
                    retry=retry,
                    retry_delay=retry_delay,
                    log_prefix="main_storyline_challenge",
                    retry_label="stamina after challenge",
                )
                if new_stamina is None:
                    return True

                consumed = before_stamina - new_stamina
                if consumed <= 0:
                    logger.info(
                        "main_storyline_challenge: stamina did not decrease; "
                        "stopping without purchasing stamina"
                    )
                    return True
                if consumed % stamina_cost != 0:
                    logger.error(
                        "main_storyline_challenge: unexpected stamina delta, "
                        f"before={before_stamina}, after={new_stamina}, cost={stamina_cost}"
                    )
                    return True

                actual = consumed // stamina_cost
                if allowance is not None and actual > allowance:
                    logger.error(
                        "main_storyline_challenge: game exceeded the requested hard limit, "
                        f"actual={actual}, allowance={allowance}"
                    )
                    return True
                challenged += actual

            logger.error(
                f"main_storyline_challenge: exceeded max_rounds={max_rounds}"
            )
        except Exception as exc:
            logger.exception(f"main_storyline_challenge: {exc}")

        return True
