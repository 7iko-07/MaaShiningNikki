import json
import re
import time

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JActionType, JRecognitionType, JOCR, JSwipe
from utils import logger
from utils.ocr import iter_recognition_results, read_ocr_fraction

from .stage_farming import CATALOG, OCR, FarmingCatalog, FarmingError, StageFarmingSession, normalized


class DailyMainStageSession(StageFarmingSession):
    KEY = ("main", 1, 1, "1-1", False)
    LIST = (20, 730, 680, 460)
    TOTAL = 10
    ACTIVITY_TARGET = 120

    def daily_page(self, image):
        return self.box(image, (452, 508, 161, 35), "今日活跃") and self.box(
            image, (53, 635, 235, 58), "每日任务"
        )

    def read_activity(self, image):
        for attempt in range(2):
            if attempt:
                time.sleep(0.25)
                image = self.frame()
                if not self.daily_page(image):
                    break
            value = read_ocr_fraction(self.context, image, (449, 566, 166, 39), model=OCR)
            if value is not None:
                return value, image
        return None, image

    def progress(self, image):
        result = self.ocr(image, (27, 730, 404, 451), [r"主\s*线\s*搭\s*配\s*关\s*卡\s*10"])
        if not result or not result.hit or not result.box:
            return None
        y = result.box[1]
        if y + 133 > 1190:
            return None
        button_roi = (446, y - 1, 201, 94)
        button = self.template(image, button_roi, "stage_farming/daily_main_stage_go.png", 0.88)
        state = "前往"
        if button is None:
            state_result = self.ocr(image, button_roi, [r"^\s*(已\s*领取|领\s*取)\s*$"])
            if not state_result or not state_result.hit or not state_result.best_result:
                return None
            item = state_result.best_result
            state = re.sub(r"\s+", "", item.text)
            button = tuple(item.box)
        count_result = self.ocr(image, (521, y + 87, 153, 46), [r"\d+\s*/\s*10"])
        counts = set()
        for entry in iter_recognition_results(count_result):
            if match := re.search(r"(?<!\d)(10|[0-9])\s*/\s*10(?!\d)", normalized(entry.text)):
                counts.add(int(match[1]))
        count = next(iter(counts)) if len(counts) == 1 else None
        return state, count, button

    def find_progress(self, image):
        upper, lower = (360, 733, 1, 1), (360, 1050, 1, 1)
        for start, end in ((lower, upper), (upper, lower)):
            for _ in range(14):
                if row := self.progress(image):
                    return row
                self._swipe(self.context, start, [end], end_hold=[500])
                after = self.frame()
                changed, _ = self._image_changed(image, after, self.LIST, 22, 0.02)
                image = after
                if not changed:
                    break
        return self.progress(image)

    def click_label(self, roi, text, timeout=10, *, ready=None):
        box = None

        def locate(frame):
            nonlocal box
            box = self.box(frame, roi, text) if ready is None or ready(frame) else None
            return box

        if self.wait(locate, timeout) is None:
            raise FarmingError(f"未识别到{text.strip('^$')}")
        self.click_box(box)

    def current_map(self, image):
        return (self.book(image) and not self.stage_panel(image)
                and self.box(image, (500, 49, 154, 45), r"卷\s*[IⅠ1]\s*第\s*[一1]\s*章"))

    def battle(self, *, first):
        if first:
            self.find_stage(self.KEY)
        else:
            image = self.frame()
            candidates = list(self.map_boxes(image, self.KEY)) if self.current_map(image) else []
            point = self.candidate_point(candidates[0], self.KEY) if candidates else None
            if point is None or self.probe_detail(self.KEY, point, phase=1)[0] is None:
                self.find_stage(self.KEY)
        self.click_label((448, 805, 235, 76), "开始故事")
        self.click_label((578, 52, 142, 65), "跳过")
        self.click_label((374, 670, 240, 78), "^确定$", ready=lambda frame: self.box(
            frame, (251, 563, 236, 65), "确定跳过对话"
        ))
        self.click_label((574, 1111, 146, 131), r"^套?\s*装$", 25)
        before = self.wait(lambda frame: self.box(frame, (550, 17, 170, 63), "推荐排序")
                           and self.box(frame, (582, 248, 138, 169), r"^[0-9]{3,}$"))
        if before is None:
            raise FarmingError("套装列表或第一套尚未就绪")
        self.click((645, 310))
        if self.wait(lambda frame: self._image_changed(
            before, frame, (10, 157, 178, 49), 24, 0.025
        )[0]) is None:
            raise FarmingError("选择第一套后未确认搭配变化")
        self.click_label((20, 1160, 250, 92), "下一步")
        self.click_label((220, 1057, 288, 135), "^确定$")
        score_roi = (247, 25, 473, 510)
        score_box = skip_box = None

        def find_battle_box(frame):
            nonlocal score_box, skip_box
            score_box = self.box(frame, score_roi, "战斗评分")
            if score_box is not None:
                return score_box
            skip_box = self.box(frame, (0, 0, 435, 164), "跳过")
            return skip_box

        self.wait(find_battle_box, 20)
        if skip_box is not None:
            self.click_box(skip_box)
        if score_box is not None:
            self.click_box(score_box)
        else:
            self.click_label(score_roi, "战斗评分", 30)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            image = self.frame()
            if self.current_map(image):
                return
            if score := self.box(image, score_roi, "战斗评分"):
                self.click_box(score)
            time.sleep(0.25)
        raise FarmingError("战斗评分关闭后未返回卷Ⅰ第一章")

    def supplement(self):
        image = self.wait(self.daily_page)
        if image is None:
            raise FarmingError("未确认每日任务页面")
        activity, image = self.read_activity(image)
        if activity is None:
            raise FarmingError("未识别到今日活跃度")
        if activity >= self.ACTIVITY_TARGET:
            return True
        row = self.find_progress(image)
        if row is None:
            raise FarmingError("未找到主线搭配关卡日常任务")
        state, completed, point_box = row
        if state != "前往" or completed == self.TOTAL:
            return True
        if completed is None:
            raise FarmingError("未识别到主线搭配任务的次数")
        pending = self.TOTAL - completed
        logger.info(f"一键领取：主线搭配已完成 {completed}/{self.TOTAL}，还需 {pending} 次")
        self.click_box(point_box)
        try:
            if self.wait(self.book, 15) is None:
                raise FarmingError("点击前往后未进入主线地图")
            for index in range(pending):
                self.battle(first=index == 0)
        except RuntimeError as error:
            if self.context.tasker.stopping:
                return False
            logger.warning(f"一键领取：补做中断，{error}")

        image = self.wait(self.current_map, 15)
        if image is None:
            logger.warning("一键领取：未确认回到卷Ⅰ第一章地图")
            return False
        back = self.template(image, (0, 30, 110, 110), "back1.png", 0.72)
        if back is None:
            logger.warning("一键领取：未识别到第一章地图的返回按钮")
            return False
        self.click_box(back)
        image = self.wait(self.daily_page, 15)
        if image is None:
            logger.warning("一键领取：点击返回后未确认每日任务页")
            return False
        result = self.context.run_task("点击一键领取", {"点击一键领取": {"next": []}})
        if not result or not result.status.succeeded:
            logger.warning("一键领取：补做后再次领取未完成")
        return True


@AgentServer.custom_action("claim_daily_main_stage")
class ClaimDailyMainStageAction(CustomAction):
    def run(self, context, argv):
        params = json.loads(argv.custom_action_param or "{}") or {}
        supplement = params.get("supplement", False)
        catalog = FarmingCatalog(json.loads(CATALOG.read_text(encoding="utf-8"))) if supplement else None
        session = DailyMainStageSession(context, catalog, False)
        try:
            if supplement and not session.supplement():
                return False
        except RuntimeError as error:
            if context.tasker.stopping:
                return False
            logger.warning(f"一键领取：{error}，跳过补做")

        try:
            result = context.run_task("点击今日活跃")
            if not result or not result.status.succeeded:
                raise FarmingError("未能点击活跃礼盒并返回")
            image = session.wait(session.daily_page)
            if image is None:
                raise FarmingError("未确认每日任务页面")
            activity, _ = session.read_activity(image)
        except RuntimeError as error:
            if not context.tasker.stopping:
                logger.error(f"一键领取：{error}")
            return False

        if activity is None:
            logger.warning("一键领取：未识别到今日活跃度")
        elif activity < session.ACTIVITY_TARGET:
            logger.info(f"今日活跃度 {activity}，未达 {session.ACTIVITY_TARGET}")
        else:
            logger.info(f"今日活跃度 {activity}")
        return True


@AgentServer.custom_action("claim_visible_rewards")
class ClaimVisibleRewardsAction(CustomAction):
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}

        list_roi = params.get("list_roi", [40, 430, 640, 590])
        button_roi = params.get("button_roi", [455, 430, 220, 590])
        close_roi = params.get("close_roi", [315, 980, 90, 70])
        swipe_roi = params.get("swipe_roi", list_roi)
        expected = params.get("expected", ["^领取$", "领取"])
        exclude_pattern = params.get("exclude_pattern", "已领取|获得|获取")
        gold_only = params.get("gold_only", False)
        max_swipes = self._as_int(params.get("max_swipes"), 8)
        max_clicks_per_page = self._as_int(params.get("max_clicks_per_page"), 20)
        ocr_threshold = self._as_float(params.get("ocr_threshold"), 0.3)
        click_delay = self._as_float(params.get("click_delay"), 0.8)
        close_delay = self._as_float(params.get("close_delay"), 0.8)
        swipe_delay = self._as_float(params.get("swipe_delay"), 0.8)
        swipe_distance = self._as_int(params.get("swipe_distance"), 360)
        swipe_duration = self._as_int(params.get("swipe_duration"), 800)
        swipe_end_hold = self._as_int(params.get("swipe_end_hold"), 200)
        stop_on_unchanged = self._as_bool(params.get("stop_on_unchanged"), True)
        page_change_threshold = self._as_float(params.get("page_change_threshold"), 0.02)
        pixel_threshold = self._as_int(params.get("pixel_threshold"), 30)

        for name, roi in (
            ("list_roi", list_roi),
            ("button_roi", button_roi),
            ("close_roi", close_roi),
            ("swipe_roi", swipe_roi),
        ):
            if not self._valid_roi(roi):
                logger.debug(f"claim_visible_rewards: invalid {name}")
                logger.error("领取奖励：点击区域配置无效")
                return False

        controller = context.tasker.controller
        claimed = 0

        for swipe_index in range(max_swipes + 1):
            page_claimed = self._claim_current_page(
                context,
                controller,
                button_roi,
                close_roi,
                expected,
                exclude_pattern,
                ocr_threshold,
                click_delay,
                close_delay,
                max_clicks_per_page,
                gold_only,
            )
            claimed += page_claimed
            logger.debug(
                "claim_visible_rewards: "
                f"page={swipe_index + 1}, page_claimed={page_claimed}, total_claimed={claimed}"
            )

            if swipe_index >= max_swipes:
                break

            before = self._screencap(controller).copy()
            self._swipe_up(context, swipe_roi, swipe_distance, swipe_duration, swipe_end_hold)
            time.sleep(swipe_delay)

            if stop_on_unchanged:
                after = self._screencap(controller)
                changed, ratio = self._image_changed(
                    before,
                    after,
                    list_roi,
                    pixel_threshold,
                    page_change_threshold,
                )
                logger.debug(
                    "claim_visible_rewards: after swipe page changed="
                    f"{changed}, ratio={ratio:.6f}"
                )
                if not changed:
                    break

        logger.debug(f"claim_visible_rewards: finished, claimed={claimed}")
        logger.info(f"已领取 {claimed} 项奖励")
        return True

    def _claim_current_page(
        self,
        context,
        controller,
        button_roi,
        close_roi,
        expected,
        exclude_pattern,
        threshold,
        click_delay,
        close_delay,
        max_clicks,
        gold_only,
    ):
        claimed = 0
        clicked_boxes = []

        for _ in range(max_clicks):
            image = self._screencap(controller)
            boxes = self._find_claim_boxes(
                context,
                image,
                button_roi,
                expected,
                exclude_pattern,
                threshold,
                clicked_boxes,
                gold_only,
            )
            if not boxes:
                break

            box = boxes[0]
            clicked_boxes.append(box)
            logger.debug(f"claim_visible_rewards: clicking claim button box={box}")
            self._click_box(controller, box)
            claimed += 1
            time.sleep(click_delay)
            self._click_roi(controller, close_roi)
            time.sleep(close_delay)

        return claimed

    def _find_claim_boxes(
        self,
        context,
        image,
        roi,
        expected,
        exclude_pattern,
        threshold,
        clicked_boxes,
        gold_only,
    ):
        result = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(roi=roi, expected=expected, threshold=threshold, order_by="Vertical"),
            image,
        )

        boxes = []
        for item in self._iter_ocr_results(result):
            text = self._normalize_text(getattr(item, "text", ""))
            box = getattr(item, "box", None)
            if not text or not box:
                continue
            if exclude_pattern and re.search(exclude_pattern, text):
                continue
            if "领取" not in text or "已领取" in text:
                continue

            box = list(box)
            if gold_only and (text != "领取" or not self._is_gold_button(image, box)):
                continue
            if self._is_duplicate_box(box, boxes) or self._is_duplicate_box(box, clicked_boxes):
                continue
            boxes.append(box)

        boxes.sort(key=lambda item: (item[1], item[0]))
        return boxes

    def _is_gold_button(self, image, box):
        x, y, w, h = box
        background = image[y:y + h, max(0, x - 12):x + w + 12]
        if not background.size:
            return False
        # BGR 范围取自话题任务和周好评礼包的黄色按钮底色。
        gold = np.all((background >= [105, 150, 210]) & (background <= [155, 195, 245]), axis=2)
        return gold.mean() >= 0.25

    def _iter_ocr_results(self, result):
        if not result or not getattr(result, "hit", False):
            return

        yielded = set()
        for attr in ("filtered_results", "all_results"):
            for item in getattr(result, attr, []) or []:
                key = id(item)
                if key in yielded:
                    continue
                yielded.add(key)
                if item is not None:
                    yield item

        best_result = getattr(result, "best_result", None)
        if best_result is not None and id(best_result) not in yielded:
            yield best_result

    def _swipe_up(self, context, roi, distance, duration, end_hold):
        x, y, w, h = roi
        begin = [x + w // 2, y + h - 40]
        end = [begin[0], max(y + 20, begin[1] - distance)]
        result = context.run_action_direct(
            JActionType.Swipe,
            JSwipe(
                begin=begin,
                end=[end],
                duration=[duration],
                end_hold=[end_hold],
            ),
        )
        if not result or not result.success:
            raise RuntimeError("claim_visible_rewards: swipe action failed")

    def _screencap(self, controller):
        controller.post_screencap().wait()
        return controller.cached_image

    def _click_roi(self, controller, roi):
        x, y, w, h = roi
        controller.post_click(x + w // 2, y + h // 2).wait()

    def _click_box(self, controller, box):
        x, y, w, h = box
        controller.post_click(x + w // 2, y + h // 2).wait()

    def _image_changed(self, before, after, roi, pixel_threshold, change_threshold):
        x, y, w, h = roi
        before = before[y:y + h, x:x + w]
        after = after[y:y + h, x:x + w]
        if before.size == 0 or after.size == 0:
            return False, 0.0

        diff = np.abs(after.astype(np.int16) - before.astype(np.int16))
        changed_pixels = np.any(diff > pixel_threshold, axis=2)
        ratio = np.count_nonzero(changed_pixels) / changed_pixels.size
        return ratio >= change_threshold, ratio

    def _is_duplicate_box(self, box, boxes):
        cx, cy = self._center(box)
        for other in boxes:
            ox, oy = self._center(other)
            if abs(cx - ox) <= 20 and abs(cy - oy) <= 20:
                return True
        return False

    def _center(self, box):
        x, y, w, h = box
        return x + w / 2, y + h / 2

    def _normalize_text(self, text):
        return re.sub(r"\s+", "", str(text or ""))

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

    def _as_bool(self, value, default):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() not in ("0", "false", "no", "off")
        return bool(value)
