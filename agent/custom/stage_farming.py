import json
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.pipeline import JOCR, JRecognitionType, JTemplateMatch

from utils import logger
from utils.ocr import _fraction_value, iter_recognition_results

from .main_storyline import _MainStorylineBase


CATALOG = Path(__file__).resolve().parents[1] / "data/stage_farming.json"
OCR = "ppocr_v6"
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}
CN_NUMBERS = "零一二三四五六七八九"
STAGE = re.compile(r"[1-9]\d?-[1-9]\d?[ABαβ]?")
DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―﹘﹣－"})
COMMON = frozenset("的一是在不了有和人这中大为上个国我以要他时来用们到作去地出就分对下能可也之与及心如新天日月色影风花梦")
GENERIC = ("时间之诗", "金币", "记忆钥匙")
BOOK = (616, 38, 104, 133)
PICKER = (285, 273, 150, 55)
TITLE = (44, 346, 420, 80)
STAMINA = (82, 0, 145, 40)
MAP = (0, 132, 720, 1018)
INV_TITLE = (254, 263, 208, 59)
INV_ENTRY = (92, 809, 150, 86)
INV_LIST = (64, 326, 592, 599)
INV_X = (119, 240, 361, 482, 602)
CHAPTER_GIFT = (285, 249, 150, 53)


class FarmingError(RuntimeError):
    pass


def normalized(value):
    return unicodedata.normalize("NFKC", value).strip()


def name_text(value):
    return re.sub(r"[^\w\u3400-\u9fff]|_", "", normalized(value)).casefold()


def stage_number(value, volume=None):
    text = re.sub(r"\s+", "", normalized(value).translate(DASHES))
    text = re.sub(r"^完美", "", text)
    text = text.translate(str.maketrans("aAbBΑΒ", "ααββαβ"))
    if volume == 2 and text.startswith("13-"):
        text = text.translate(str.maketrans("αβ", "AB"))
    return text if STAGE.fullmatch(text) else ""


def chapter_number(text):
    text = normalized(text)
    if text.isdecimal():
        return int(text)
    if text in CN_NUMBERS:
        return CN_NUMBERS.index(text)
    if "十" in text:
        tens, units = text.split("十", 1)
        if (not tens or tens in CN_NUMBERS) and (not units or units in CN_NUMBERS):
            return (CN_NUMBERS.index(tens) if tens else 1) * 10 + (CN_NUMBERS.index(units) if units else 0)
    return 0


def title_match(text, target, other_titles):
    text, target = name_text(text), name_text(target)
    if not text or not target or target in other_titles:
        return False
    if text == target:
        return True
    # Reject another full title before accepting a short fragment.
    if text in other_titles:
        return False
    for size in range(1, min(4, len(target)) + 1):
        for index in range(len(target) - size + 1):
            fragment = target[index:index + size]
            if size == 1 and (fragment in COMMON or target.count(fragment) != 1 or not fragment.isalpha()):
                continue
            if fragment in text and not any(fragment in title for title in other_titles):
                return True
    return False


class FarmingCatalog:
    def __init__(self, rows):
        self.chapters = {}
        self.stages = {}
        self.materials = defaultdict(list)
        for row in rows:
            chapter_key = row["mode"], row.get("volume"), row["chapter"]
            self.chapters[chapter_key] = row
            for stage in row["stages"]:
                key = *chapter_key, stage["id"], stage.get("challenge", False)
                self.stages[key] = stage
                for material in stage["drops"]:
                    self.materials[name_text(material)].append(key)

    def lookup(self, raw):
        raw = normalized(raw)
        main = re.fullmatch(r"(?:主线\s*)?卷\s*([12]|II|I)\s*(普通|挑战)?(?:关卡)?\s*(\S+)", raw, re.I)
        maze = re.fullmatch(r"迷宫\s*(\S+)", raw)
        investigation = re.fullmatch(r"调查\s*([34])-([1-9]\d?)-([1-5])", raw)
        if main:
            volume = ROMAN.get(main[1].upper()) or int(main[1])
            stage = stage_number(main[3], volume)
            if stage:
                return "main", volume, int(stage.split("-")[0]), stage, main[2] == "挑战"
        if maze and (stage := stage_number(maze[1])):
            return "maze", None, int(stage.split("-")[0]), stage, False
        if investigation:
            return "investigation", int(investigation[1]), int(investigation[2]), f"{investigation[2]}-{investigation[3]}", False
        raise FarmingError(f"关卡格式无效：{raw}")

    @staticmethod
    def label(key):
        mode, volume, _, stage, challenge = key
        if mode == "maze":
            return f"迷宫 {stage}"
        if mode == "investigation":
            return f"调查 {volume}-{stage}"
        return f"卷{volume} {'挑战 ' if challenge else ''}{stage}"

    def plan(self, text, by_material):
        for row in re.split(r"[,，;；\n]+", text):
            if not row.strip():
                continue
            lhs, separator, count = row.partition("=")
            if not separator:
                lhs, separator, count = row.partition("＝")
            count = normalized(count)
            if not count:
                continue
            if not re.fullmatch(r"[0-9]+", count):
                yield lhs, None, (), "数量需为非负整数"
                continue
            amount = int(count)
            material, source = lhs.strip(), lhs.strip()
            try:
                if by_material:
                    material, _, source = lhs.partition("@") if "@" in lhs else lhs.partition("＠")
                    choices = self.materials.get(name_text(material), ())
                    if source.strip() and source.strip() != "自动":
                        selected = self.lookup(source)
                        if selected not in choices:
                            raise FarmingError("指定关卡没有该材料的掉落记录")
                        choices = (selected,)
                    if not choices:
                        raise FarmingError("未找到材料的掉落来源")
                else:
                    selected = self.lookup(source)
                    if selected not in self.stages:
                        raise FarmingError("目录没有该关的验证数据")
                    choices = (selected,)
                yield material.strip(), amount, choices, None
            except FarmingError as error:
                yield lhs.strip(), None, (), str(error)


class StageFarmingSession(_MainStorylineBase):
    def __init__(self, context, catalog, auto_clear):
        self.context = context
        self.controller = context.tasker.controller
        self.catalog = catalog
        self.auto_clear = auto_clear

    def frame(self):
        if self.context.tasker.stopping:
            raise FarmingError("任务已停止")
        if not self.controller.post_screencap().wait().succeeded:
            raise FarmingError("截图失败")
        image = self.controller.cached_image
        if image is None or image.shape[:2] != (1280, 720):
            raise FarmingError("关卡定位需要 720×1280 的画面")
        return image

    def ocr(self, image, roi, expected=None, *, single=False):
        params = JOCR(roi=tuple(roi), model=OCR, expected=expected or [], only_rec=single)
        return self.context.run_recognition_direct(JRecognitionType.OCR, params, image)

    def texts(self, image, roi, *, single=False):
        return [str(item.text) for item in iter_recognition_results(self.ocr(image, roi, single=single))]

    def box(self, image, roi, pattern):
        result = self.ocr(image, roi, [pattern])
        return tuple(result.box) if result and result.hit and result.box else None

    def template(self, image, roi, filename, threshold=0.8):
        if not filename:
            return None
        result = self.context.run_recognition_direct(
            JRecognitionType.TemplateMatch,
            JTemplateMatch(roi=tuple(roi), template=[filename], threshold=[threshold]),
            image,
        )
        return tuple(result.box) if result and result.hit and result.box else None

    def wait(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            image = self.frame()
            if predicate(image):
                return image
            time.sleep(0.25)
        return None

    def click(self, point):
        if not self.controller.post_click(*point).wait().succeeded:
            raise FarmingError("点击失败")

    def click_box(self, box):
        x, y, w, h = box
        self.click((x + w // 2, y + h // 2))

    def swipe(self, start, end, *, reset=False):
        self._swipe(self.context, (*start, 1, 1), [(*end, 1, 1)],
                    None if reset else [550], None if reset else [1500])
        time.sleep(0.25 if reset else 0.6)

    def navigate(self, target, *, investigation_entry=False):
        navigation = self.context.clone()
        overrides = {}
        extra = {"是否是主页面", "主线特殊过渡页面", "点击返回", "关闭退出游戏确认弹窗"}
        for name in navigation.tasker.resource.node_list:
            if name.startswith("导航") or name in extra:
                node = navigation.get_node_data(name) or {}
                recognition = node.get("recognition")
                if isinstance(recognition, dict) and recognition.get("type") == "OCR":
                    overrides[name] = {"recognition": {"param": {"model": OCR}}}
                elif recognition == "OCR":
                    overrides[name] = {"model": OCR}
        params = {"target": target}
        if investigation_entry:
            entry = "导航在调查行动页面"
            story_map = "导航在主线故事页面"
            overrides[entry] = {"recognition": {"param": {
                "roi": list(INV_ENTRY), "expected": ["^调查行动$"], "model": OCR,
            }}}
            overrides.setdefault("导航点击主线故事", {})["next"] = [
                entry, "主线特殊过渡页面", story_map,
            ]
            params["pages"] = {target: [entry, story_map]}
        if overrides and not navigation.override_pipeline(overrides):
            raise FarmingError("未能设置导航识别模型")
        result = navigation.run_task("主线故事任务链准备", {
            "主线故事任务链准备": {
                "action": {
                    "type": "Custom",
                    "param": {
                        "custom_action": "page_navigate",
                        "custom_action_param": params,
                    },
                },
                "next": [],
            },
        })
        return bool(result and result.status.succeeded)

    def book(self, image):
        return self.template(image, BOOK, "主线章节.png", 0.7)

    def picker(self, image):
        return self.box(image, PICKER, "切换期数")

    def inv_list(self, image):
        return self.box(image, INV_TITLE, "^调查行动$")

    def stage_panel(self, image):
        return (self.box(image, (43, 530, 420, 275), "本关掉落|每次消耗")
                or self.box(image, (55, 403, 134, 146), "最高评分|评审风格|必须穿戴")
                or self.box(image, TITLE, "挑战关[卡关]"))

    def close_stage(self, image):
        if self.box(image, CHAPTER_GIFT, "^章节礼包$") or self.stage_panel(image):
            self.click((335, 111))
        elif self.box(image, (552, 33, 168, 91), "跳过"):
            self.click((46, 69))
        elif np.mean(image[140:1100].max(axis=2) < 34) > 0.9:
            back = self.template(image, (0, 30, 110, 110), "back1.png", 0.72)
            if back:
                self.click_box(back)
        return self.wait(lambda frame: self.book(frame) and not self.stage_panel(frame), 3)

    def map_ready(self):
        image = self.frame()
        if self.inv_list(image):
            self.click((355, 190))
            image = self.wait(lambda f: not self.inv_list(f))
        if image is not None and self.book(image):
            return self.close_stage(image)
        if self.navigate("主线故事"):
            return self.wait(lambda frame: self.book(frame))
        return None

    def chapter_picker(self):
        image = self.frame()
        if self.picker(image):
            return image
        image = self.map_ready()
        if image is None:
            raise FarmingError("未进入主线地图")
        self.click_box(self.book(image))
        image = self.wait(self.picker, 3)
        if image is None:
            image = self.close_stage(self.frame())
            if image is not None:
                self.click_box(self.book(image))
                image = self.wait(self.picker, 3)
        if image is None:
            raise FarmingError("未打开章节切换")
        return image

    def main_volume(self, volume, image):
        tab = self.box(image, (205, 344, 145, 67), r"^主\s*线$")
        if not tab:
            raise FarmingError("未识别到主线标签")
        self.click_box(tab)
        image = self.wait(lambda f: self.picker(f) and self.box(f, (50, 424, 580, 59), "卷"))
        if image is None:
            raise FarmingError("未打开主线章节列表")
        x = (125, 270, 415, 560)[volume - 1]
        self.click((x, 454))
        return self.wait(lambda f: self.picker(f) and np.median(f[431:440, x - 26:x + 26]) >= 235)

    def enter_chapter(self, key):
        mode, volume, chapter = key[:3]
        image = self.chapter_picker()
        if mode == "maze":
            def maze_selected(frame):
                colors = frame[358:390, 365:383].astype(np.int16)
                return np.median(colors[:, :, 2] - colors[:, :, 0]) >= 65
            if not maze_selected(image):
                tab = self.box(image, (350, 345, 142, 68), r"迷\s*宫")
                if not tab:
                    raise FarmingError("未识别到迷宫标签")
                self.click_box(tab)
                image = self.wait(lambda frame: self.picker(frame) and maze_selected(frame))
            roi, start, end = (78, 432, 111, 568), (350, 961), (350, 580)
        else:
            image = self.main_volume(volume, image)
            roi, start, end = (218, 497, 336, 576), (350, 1064), (350, 498)
        if image is None:
            raise FarmingError("章节列表未就绪")

        def rows(frame):
            result = self.ocr(frame, roi, [r"^第\s*[一二三四五六七八九十\d]+\s*章"])
            boxes = {}
            for item in iter_recognition_results(result):
                match = re.match(r"^第\s*([一二三四五六七八九十\d]+)\s*章", str(item.text).strip())
                if match:
                    boxes[chapter_number(match[1])] = tuple(item.box)
            return boxes

        moved = set()
        for _ in range(24):
            visible = rows(image)
            if chapter in visible:
                self.click_box(visible[chapter])
                def entered(frame):
                    if not self.book(frame) or self.picker(frame):
                        return False
                    heading = "".join(self.texts(frame, (480, 49, 167, 46)))
                    match = re.search(r"第\s*([一二三四五六七八九十\d]+)\s*章", heading)
                    if not match or chapter_number(match[1]) != chapter:
                        return False
                    if volume is None:
                        return True
                    prefix = normalized(heading.split("第", 1)[0]).upper()
                    vol = re.search(r"卷\s*(II|I|[12])", prefix)
                    return bool(vol and (ROMAN.get(vol[1]) or int(vol[1])) == volume)
                if self.wait(entered, 8) is not None:
                    return
                raise FarmingError("切章后未确认目标章节")
            marker = tuple(sorted(visible))
            if marker and marker in moved:
                break
            moved.add(marker)
            up = not visible or chapter > min(visible)
            self.swipe(start if up else end, end if up else start)
            image = self.frame()
        raise FarmingError("章节列表里没找到目标")

    def title_ok(self, image, key):
        stage = self.catalog.stages[key]
        competitors = {
            name_text(s["title"]) for s in self.catalog.chapters[key[:3]]["stages"] if s is not stage
        }
        lines = self.texts(image, TITLE)
        challenge = any(re.search(r"挑战关[卡关]", text) for text in lines)
        if challenge != key[-1]:
            return False
        clean = [re.sub(r"^\s*(?:挑战关[卡关]\s*)?(?:[0-9]+\s*[-－—]\s*[0-9]+[αβABab]?)?\s*", "", text)
                 for text in lines]
        return any(title_match(line, stage["title"], competitors) for line in clean)

    def map_boxes(self, image, key):
        _, volume, _, target, _ = key
        result = self.ocr(image, MAP)
        used = set()
        for item in iter_recognition_results(result):
            box = self._box_list(item.box)
            if box and min(box[2:]) > 0 and stage_number(item.text, volume) == target and tuple(box) not in used:
                used.add(tuple(box))
                yield box
        for candidate in self._stage_candidates(image, list(MAP)):
            if stage_number(candidate.text, volume) == target and tuple(candidate.box) not in used:
                if self.zoom_matches(image, candidate.box, key):
                    yield candidate.box
            if (volume, target) != (2, "3-11") or not candidate.rejection:
                continue
            if candidate.left not in ((None,), ("3",)) or candidate.right != ("1", "1", None):
                continue
            x, y, w, h = candidate.box
            left, top = max(0, x - 28), max(132, y - 40)
            roi = (left, top, min(720, x + w + 25) - left, min(1150, y + h + 39) - top)
            box = self.box(image, roi, "^3-11$")
            if box and max(abs(a - b) for a, b in zip(box, candidate.box)) <= 18:
                yield box

    def zoom_matches(self, image, roi, key):
        x, y, w, h = roi
        crop = image[max(0, y - 4):min(1280, y + h + 4), max(0, x - 4):min(720, x + w + 4)]
        if not crop.size:
            return False
        zoom = np.repeat(np.repeat(crop, 3, axis=0), 3, axis=1)
        return any(stage_number(text, key[1]) == key[3]
                   for text in self.texts(zoom, (0, 0, zoom.shape[1], zoom.shape[0]), single=True))

    def candidate_point(self, box, key):
        x, y, w, h = box
        cx, cy = x + w // 2, y + h // 2
        if key == ("main", 2, 16, "16-4", True) and x < 261 and y < 1015 < y + h:
            return cx, 1015
        if self._boxes_overlap(box, (0, 894, 261, 105)):
            return None
        if self._boxes_overlap(box, (580, 916, 140, 79)):
            return None
        return cx, cy

    @staticmethod
    def badge(image, box):
        x, y, w, h = box
        patch = image[max(0, y - 16):y + h + 16, max(0, x - 8):x + w + 8]
        b, g, r = patch.astype(np.int16).transpose(2, 0, 1)
        blue = (b >= 205) & (b - g >= 50) & (g - r >= 5) & (r <= 165)
        pink = (r >= 160) & (r - g >= 35) & (r - b >= 12)
        for kind, mask, threshold in (("story", blue, 0.45), ("challenge", pink, 0.55)):
            if all(strip.size and strip.mean() >= threshold for strip in (mask[:14], mask[14:-14], mask[-14:])):
                return kind
        return None

    def probe_detail(self, key, point, *, phase):
        self.click(point)
        dismissed_rules = False

        def detail(frame):
            nonlocal dismissed_rules
            if key[0] == "maze" and not dismissed_rules and self.maze_rules(frame):
                self.click((335, 111))
                dismissed_rules = True
                return False
            return self.stage_panel(frame) and self.title_ok(frame, key)

        result = self.wait(detail, 4)
        if result is not None:
            return result, False
        logger.debug(f"刷取：{self.catalog.label(key)} 第{phase}轮点击 {point} 后标题核对失败")
        image = self.frame()
        gift = key == ("main", 1, 12, "12-5", True) and self.box(image, CHAPTER_GIFT, "^章节礼包$")
        if self.close_stage(image) is None:
            raise FarmingError("点击后未能返回章节地图")
        return None, bool(gift)

    def maze_rules(self, image):
        return self.box(image, (215, 306, 292, 65), "心灵迷宫次数规则")

    def reset_map(self):
        for _ in range(3):
            self.swipe((61, 500), (698, 500), reset=True)

    def left(self):
        self.swipe((614, 602), (339, 602))

    def find_stage(self, key):
        stage = self.catalog.stages[key]
        self.enter_chapter(key)
        self.reset_map()
        image = self.frame()
        still = 0
        for _ in range(30):
            tried = set()
            for _ in range(5):
                candidates = []
                for box in self.map_boxes(image, key):
                    badge = self.badge(image, box)
                    if badge == "story" or (not key[-1] and badge == "challenge"):
                        continue
                    point = self.candidate_point(box, key)
                    if point is not None and not any(abs(x - point[0]) < 12 and abs(y - point[1]) < 12 for x, y in tried):
                        candidates.append((badge != "challenge", point))
                if not candidates:
                    break
                _, point = min(candidates)
                tried.add(point)
                result, gift = self.probe_detail(key, point, phase=1)
                if result is not None:
                    return result
                image = self.frame()
                if gift:
                    break
            self.left()
            after = self.frame()
            changed, _ = self._image_changed(image, after, (90, 220, 495, 640), 25, 0.025)
            still = 0 if changed else still + 1
            if still >= 2:
                break
            image = after

        anchor = stage.get("anchor")
        if anchor:
            for phase in (2, 3):
                self.reset_map()
                for _ in range(anchor["left"]):
                    self.left()
                image = self.frame()
                point = tuple(anchor["point"])
                if phase == 2:
                    x, y, w, h = anchor["roi"]
                    lx, ty = max(0, x - 28), max(0, y - 29)
                    search = (lx, ty, min(720, x + w + 28) - lx, min(1280, y + h + 29) - ty)
                    if not (self.zoom_matches(image, anchor["roi"], key)
                            or self.template(image, search, anchor["image"], 0.85)):
                        continue
                result, gift = self.probe_detail(key, point, phase=phase)
                if result is not None:
                    return result
                if gift:
                    # 礼包旁的挑战节点移到下一屏后才能正常点击。
                    self.left()
                    image = self.frame()
                    for box in self.map_boxes(image, key):
                        if (point := self.candidate_point(box, key)) is None or self.badge(image, box) == "story":
                            continue
                        result, _ = self.probe_detail(key, point, phase=phase)
                        if result is not None:
                            return result
                        break
        raise FarmingError("三轮后仍未通过关卡标题核对")

    def open_investigation(self):
        image = self.frame()
        if self.inv_list(image):
            return image
        entry = self.box(image, INV_ENTRY, "^调查行动$")
        if not entry and not self.picker(image) and not self.book(image):
            if not self.navigate("主线故事", investigation_entry=True):
                raise FarmingError("未进入主线故事")
            image = self.frame()
            entry = self.box(image, INV_ENTRY, "^调查行动$")
        if not entry:
            image = self.chapter_picker()
            picker = self.main_volume(3, image)
            if picker is not None:
                first = self.box(picker, (218, 497, 336, 576), r"^第\s*[一1]\s*章")
                if not first:
                    raise FarmingError("未找到卷Ⅲ第一章")
                self.click_box(first)
            image = self.wait(lambda f: self.box(f, INV_ENTRY, "^调查行动$"), 5)
            if image is None:
                # 卷Ⅲ地图的上一页是带调查行动入口的过渡页。
                current = self.frame()
                back = self.template(current, (0, 30, 110, 110), "back1.png", 0.72)
                if self.book(current) and back:
                    self.click_box(back)
                    image = self.wait(lambda f: self.box(f, INV_ENTRY, "^调查行动$"), 5)
            if image is None:
                raise FarmingError("未进入调查行动入口页")
            entry = self.box(image, INV_ENTRY, "^调查行动$")
        self.click_box(entry)
        image = self.wait(self.inv_list)
        if image is None:
            raise FarmingError("未打开调查行动列表")
        return image

    def investigation_rows(self, image):
        b, g, r = image[324:926, 97:108].astype(np.int16).transpose(2, 0, 1)
        line = ((b - r >= 16) & (b - g >= 12) & (r <= 209)).mean(axis=1) >= 0.7
        edges = np.diff(np.r_[False, line, False].astype(int))
        rows = []
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if not 26 <= stop - start <= 43:
                continue
            y = int((start + stop) // 2) + 324
            chapter = None
            for roi in ((104, y - 21, 146, 43), (96, y - 24, 173, 47)):
                label = "".join(self.texts(image, roi, single=True))
                match = re.search(r"卷\s*(IV|III|II|I|[1-4])\s*第\s*([一二三四五六七八九十\d]+)\s*章",
                                  normalized(label).upper())
                if match and (number := chapter_number(match[2])):
                    chapter = (ROMAN.get(match[1]) or int(match[1]), number)
                    break
            rows.append((y, chapter))
        return rows

    def outfit_y(self, image, key, rows):
        expected = key[1:3]
        info = self.catalog.chapters[key[:3]]
        other_names = {name_text(c["name"]) for k, c in self.catalog.chapters.items()
                       if k[0] == "investigation" and k != key[:3]}
        for y, label in rows:
            if label is not None and label != expected:
                continue
            names = self.texts(image, (485, y - 21, 151, 28), single=True)
            if any(title_match(line, info["name"], other_names) for line in names):
                return y
            if self.template(image, (412, y - 36, 246, 78), info["image"], 0.83):
                return y
        return None

    @staticmethod
    def card_selected(image, x, y):
        b, g, r = np.median(image[y + 5:y + 28, x - 26:x + 25], axis=(0, 1))
        return b > r + 6 and b > g + 10

    def investigation_reward_y(self, image, y):
        start = y + 151
        if start + 45 >= 930:
            return None
        box = self.box(image, (175, start, 215, min(73, 930 - start)), "挑战奖励")
        if not box:
            return None
        reward_y = box[1] + box[3] // 2
        return reward_y if reward_y + 170 < 930 else None

    def investigation_state(self, image, key):
        if not self.inv_list(image):
            return None
        y = self.outfit_y(image, key, self.investigation_rows(image))
        stage = self.catalog.stages[key]
        if y is None or y + 125 > 930 or not self.card_selected(image, INV_X[stage["section"] - 1], y + 96):
            return None
        return self.investigation_reward_y(image, y)

    def swipe_investigation(self, start, end):
        self._swipe(self.context, (*start, 1, 1), [(*end, 1, 1)], [200], [500])
        time.sleep(0.25)

    def align_investigation(self, y):
        start = 864 if y > 552 else 480
        self.swipe_investigation((397, start), (397, start - (y - 552)))

    def find_investigation(self, key):
        image = self.open_investigation()
        stage = self.catalog.stages[key]
        expected = key[1:3]
        unchanged = 0
        for _ in range(40):
            rows = self.investigation_rows(image)
            y = self.outfit_y(image, key, rows)
            if y is not None:
                x = INV_X[stage["section"] - 1]
                if not 360 <= y <= 573 and not (expected in ((3, 1), (3, 2)) and y + 300 < 930):
                    self.align_investigation(y)
                    image = self.frame()
                    continue
                if not self.card_selected(image, x, y + 96):
                    card = image[y + 101:y + 124, x - 26:x + 25]
                    b, _, r = np.median(card, axis=(0, 1))
                    if r < b + 18:
                        raise FarmingError("调查关卡未解锁或卡片未确认")
                    self.click((x, y + 96))
                result = self.wait(lambda f: self.investigation_state(f, key) is not None)
                if result is not None:
                    return result
                image = self.frame()
                new_y = self.outfit_y(image, key, self.investigation_rows(image))
                if new_y is None:
                    raise FarmingError("点选后无法确认调查章节")
                self.align_investigation(new_y)
                image = self.frame()
                continue
            known = [label for _, label in rows if label is not None]
            up = not known or expected >= min(known)
            self.swipe_investigation((395, 901) if up else (395, 470), (395, 611) if up else (395, 850))
            after = self.frame()
            changed, _ = self._image_changed(image, after, INV_LIST, 20, 0.015)
            if not changed and not known and unchanged == 0:
                self.swipe_investigation((395, 470), (395, 880))
                after = self.frame()
            unchanged = 0 if changed else unchanged + 1
            if unchanged >= 2:
                break
            image = after
        raise FarmingError("未确认目标调查章节及序号")

    def first_reward(self, image, key, y, *, after_clear=False):
        stage = self.catalog.stages[key]
        name = stage["title"]
        texts = self.texts(image, (62, y + 127, 138, 30), single=True)
        names = [name_text(text) for text in texts]
        if any(any(name_text(g) in text for g in GENERIC) for text in names):
            return "generic"
        peers = {name_text(s["title"]) for k, s in self.catalog.stages.items()
                 if k[0] == "investigation" and k != key}
        expected = [name, *stage["drops"]] if after_clear else [name]
        if any(title_match(text, item, peers - ({name_text(item)} if after_clear else set()))
               for text in texts for item in expected):
            return "confirmed"
        if self.template(image, (84, y + 34, 98, 98), stage["image"], 0.85):
            return "generic" if name.startswith(GENERIC) else "confirmed"
        return None

    def challenge_button(self, image, key, reward_y=None):
        roi = (480, reward_y + 65, 175, 90) if reward_y is not None else (43, 800, 153, 76)
        return self.box(image, roi, r"挑战\s*[一1]\s*次")

    @staticmethod
    def enabled(image, box):
        if not box:
            return False
        x, y, w, h = box
        b, g, r = np.median(image[y:y + h, x:x + w].astype(np.int16), axis=(0, 1))
        return b - r >= 25 and b - g >= 12

    @staticmethod
    def pink(image, roi):
        x, y, w, h = roi
        b, g, r = image[y:y + h, x:x + w].astype(np.int16).transpose(2, 0, 1)
        return ((r >= 200) & (r - g >= 45) & (r - b >= 25)).sum() >= 100

    def auto_first_clear(self, image, key, y):
        start = self.box(image, (475, y - 17, 186, 72), "^进行调查$")
        if not start:
            raise FarmingError("未找到进行调查按钮")
        recommendation = (515, 1132, 205, 148)
        auto = (48, 1098, 47, 52)
        self.click_box(start)
        image = self.wait(lambda f: self.box(f, recommendation, "推荐搭配")
                          and self.box(f, (88, 1094, 120, 62), "代理评选"), 20)
        if image is None:
            raise FarmingError("调查首通未进入搭配页")
        if not self.pink(image, auto):
            self.click_box(self.box(image, (88, 1094, 120, 62), "代理评选"))
            image = self.wait(lambda f: self.pink(f, auto))
        for roi, label in (
            (recommendation, "推荐搭配"),
            ((0, 1090, 310, 190), "下一步"),
            ((220, 1057, 288, 135), "^确定$"),
            ((480, 0, 240, 220), "跳过"),
        ):
            image = self.wait(lambda f: self.box(f, roi, label), 15) if image is not None else None
            if image is None:
                raise FarmingError(f"调查首通未确认{label.strip('^$')}")
            self.click_box(self.box(image, roi, label))
        if self.wait(lambda f: self.box(f, (247, 25, 473, 510), "战斗评分"), 30) is None:
            raise FarmingError("未确认战斗评分")
        self.click((350, 120))
        image = self.wait(lambda f: self.investigation_state(f, key) is not None, 8)
        if image is None:
            image = self.find_investigation(key)
        y = self.investigation_state(image, key)
        if y is None or not self.enabled(image, self.challenge_button(image, key, y)):
            raise FarmingError("首通后挑战一次仍不可用")
        reward = self.first_reward(image, key, y, after_clear=True)
        if not reward:
            raise FarmingError("首通后未通过首项奖励核对")
        return image

    def stage_details(self, key):
        if key[0] != "investigation":
            image = self.find_stage(key)
            return image, None, False
        image = self.find_investigation(key)
        y = self.investigation_state(image, key)
        if y is None:
            raise FarmingError("未找到调查奖励区")
        reward = self.first_reward(image, key, y)
        button = self.challenge_button(image, key, y)
        if reward == "generic" or (button and not self.enabled(image, button)):
            return image, y, True
        if not reward:
            raise FarmingError("第一项挑战奖励未核对成功")
        return image, y, False

    def stamina(self, image, roi):
        for single in (False, True):
            value = _fraction_value(self.ocr(image, roi, [".*"], single=single))
            if value is not None:
                return value
        return None

    def cost(self, image, key):
        roi = (595, 939, 30, 53) if key[0] == "investigation" else (193, 766, 43, 27)
        matches = {int(text) for s in self.texts(image, roi, single=True)
                   if (text := normalized(s)).isdecimal()}
        return next(iter(matches)) if len(matches) == 1 else None

    def remaining(self, image, key):
        if key[0] == "maze" or key[-1]:
            return self.stamina(image, (503, 754, 104, 55))
        return None

    def owned(self, image, key, material, reward_y):
        if reward_y is not None:
            names_y, values_y = reward_y + 127, reward_y + 153
            boxes = [(62, 138), (216, 112), (353, 112)]
        else:
            header = self.box(image, (285, 510, 150, 80), "本关掉落")
            if not header:
                return None
            y = header[1] + header[3] // 2
            names_y, values_y = y + 138, y + 174
            boxes = [(118, 145), (290, 145), (462, 145)]
        for x, width in boxes:
            names = self.texts(image, (x, names_y, width, 30), single=True)
            if not any(name_text(name) == name_text(material) for name in names):
                continue
            for single in (True, False):
                values = self.texts(image, (x, values_y, width, 32), single=single)
                for value in values:
                    if match := re.search(r"已\s*拥\s*有\s*[:：]?\s*([0-9][0-9,]*)\s*$", normalized(value)):
                        return int(match[1].replace(",", ""))
                if any("已拥有" in s for s in values):
                    digits = {int(s) for s in values if s.isdecimal()}
                    if len(digits) == 1:
                        return digits.pop()
        return None

    def dropped(self, image, material):
        result = self.ocr(image, (60, 215, 600, 680))
        items = list(iter_recognition_results(result))
        found = set()
        for item in items:
            raw = normalized(item.text)
            match = re.fullmatch(r"(.+?)\s*[xX×*]\s*(\d+)", raw)
            if match and name_text(match[1]) == name_text(material):
                found.add(int(match[2]))
        return found.pop() if len(found) == 1 else None

    def active_details(self, image, key):
        if key[0] == "investigation":
            y = self.investigation_state(image, key)
            return y is not None and self.first_reward(image, key, y, after_clear=True)
        return self.stage_panel(image) and self.title_ok(image, key)

    def challenge_once(self, key, button, material):
        self.click_box(button)
        result = self.wait(lambda f: self.box(f, (70, 198, 580, 982), r"过\s*关\s*完\s*成|恭喜获得"), 12)
        if result is None:
            raise FarmingError("未确认挑战结算")
        gained = self.dropped(result, material) if material else None
        self.click((289, 176))
        image = self.wait(lambda f: self.active_details(f, key), 8)
        if image is None:
            raise FarmingError("结算后未返回目标关")
        return image, gained

    def farm(self, key, image, reward_y, first, amount, material):
        stock = self.owned(image, key, material, reward_y) if material else 0
        if material and stock is None:
            raise FarmingError("未读到目标材料的持有量")
        if material and amount and stock >= amount:
            return stock, None
        if first:
            if not self.auto_clear:
                raise FarmingError("需要先挑战通关才能刷取")
            cost, stamina = self.cost(image, key), self.stamina(image, STAMINA)
            if cost is None or stamina is None:
                raise FarmingError("未确认首通体力消耗")
            if stamina < cost:
                return stock, "体力不足"
            image = self.auto_first_clear(image, key, reward_y)
            reward_y = self.investigation_state(image, key)
            if material:
                stock = self.owned(image, key, material, reward_y)
                if stock is None:
                    raise FarmingError("首通后未读到目标材料持有量")
        progress = stock
        while not amount or progress < amount:
            button = self.challenge_button(image, key, reward_y)
            remaining = self.remaining(image, key)
            if remaining == 0:
                return progress, "可用挑战次数用完"
            cost, stamina = self.cost(image, key), self.stamina(image, STAMINA)
            if cost is None or cost <= 0 or stamina is None:
                raise FarmingError("未确认体力或单次消耗")
            if stamina < cost:
                return progress, "体力不足"
            if not self.enabled(image, button):
                raise FarmingError("挑战一次不可用")
            image, gained = self.challenge_once(key, button, material)
            stamina_after = self.stamina(image, STAMINA)
            if stamina_after is None or stamina_after >= stamina:
                raise FarmingError("结算后未确认体力消耗")
            if reward_y is not None:
                reward_y = self.investigation_state(image, key)
            if material:
                if gained is None:
                    current = self.owned(image, key, material, reward_y)
                    if current is None or current < progress:
                        raise FarmingError("结算和库存均未确认材料数量")
                    gained = current - progress
                progress += gained
            else:
                progress += 1
        return progress, None

    def candidates(self, choices):
        if len(choices) == 1:
            return [(choices[0], None, None)]
        ranked, failed = [], []
        for index, key in enumerate(choices):
            try:
                image, y, first = self.stage_details(key)
                if first and not self.auto_clear:
                    raise FarmingError("需要先挑战通关才能刷取")
                button = self.challenge_button(image, key, y)
                if not first and not self.enabled(image, button):
                    raise FarmingError("挑战一次不可用")
                if self.remaining(image, key) == 0:
                    raise FarmingError("可用挑战次数用完")
                cost = self.cost(image, key)
                if cost is None:
                    raise FarmingError("未确认单次消耗")
                ranked.append((first, cost, index, key))
            except RuntimeError as error:
                failed.append((key, None, str(error)))
        return [(key, cost, None) for _, cost, _, key in sorted(ranked)] + failed

    def recover(self):
        image = self.frame()
        if self.inv_list(image):
            return
        if self.picker(image):
            self.click((355, 190))
            return
        if self.book(image):
            self.close_stage(image)

    def run_item(self, amount, choices, material):
        errors = []
        for key, _, known_error in self.candidates(choices):
            if known_error:
                errors.append(f"{self.catalog.label(key)} {known_error}")
                continue
            try:
                image, y, first = self.stage_details(key)
            except RuntimeError as error:
                errors.append(f"{self.catalog.label(key)} {error}")
                self.recover()
                continue
            progress, reason = self.farm(key, image, y, first, amount, material)
            if reason and amount and progress < amount:
                errors.append(f"{self.catalog.label(key)} {reason}（已{'拥有' if material else '刷'} {progress}/{amount}）")
                continue
            return progress, reason
        raise FarmingError("；".join(errors) or "没有可用来源")


@AgentServer.custom_action("stage_farming")
class StageFarmingAction(CustomAction):
    def run(self, context, argv):
        params = json.loads(argv.custom_action_param)
        catalog = FarmingCatalog(json.loads(CATALOG.read_text(encoding="utf-8")))
        by_material = params.get("mode") == "material"
        plan = list(catalog.plan(params.get("list", ""), by_material))
        if not plan:
            logger.info("关卡刷取：未填写刷取项目")
            return True
        policy = context.get_node_data("关卡刷取允许调查首通") or {}
        session = StageFarmingSession(context, catalog, policy.get("enabled", False))
        failures = []
        for label, amount, choices, error in plan:
            if context.tasker.stopping:
                break
            if error:
                failures.append((label, error))
                continue
            try:
                progress, reason = session.run_item(amount, choices, label if by_material else None)
                result = f"持有 {progress} 件" if by_material else f"刷取 {progress} 次"
                logger.info(f"关卡刷取：{label}，{result}" + (f"，{reason}" if reason else ""))
            except RuntimeError as error:
                failures.append((label, str(error)))
                if not context.tasker.stopping:
                    try:
                        session.recover()
                    except RuntimeError as recovery_error:
                        logger.debug(f"关卡刷取：页面恢复失败，{recovery_error}")
        if failures:
            message = "；".join(f"{name}：{reason}" for name, reason in failures)
            logger.warning(f"关卡刷取：未完成 {len(failures)} 项。{message}")
        return not context.tasker.stopping
