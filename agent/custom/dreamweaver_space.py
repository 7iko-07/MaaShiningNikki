"""织梦空间：签到、经营和七日妙想的识别驱动流程。"""

import json
import math
import re
import subprocess
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.pipeline import JClick, JInputText, JOCR, JSwipe
from utils import extract_ocr_text, logger


TYPES = ("连衣裙", "上装", "下装")
MATERIALS = ("布艺", "皮革", "丝绒", "绸缎", "特殊")
COLORS = ("蓝色系", "黄色系", "红色系", "绿色系", "棕色系", "灰色系")
ORDER_IMAGES = ("一星订单.png", "二星订单.png", "三星订单.png", "订单完成对号.png")
NATURAL_WAIT_SECONDS = 600
PRODUCTION_CONFIRM_SECONDS = 30
BEIJING = timezone(timedelta(hours=8))
SALE_MODES = {
    "stock": "攒1.5倍",
    "current": "仅售卖当期产品",
    "tomorrow": "非1.5倍攒明天",
}


class DreamError(RuntimeError):
    pass


# 灵感拾趣：棋盘计算和单局状态均在 Agent 中管理。
COLS, ROWS = 13, 8
PITCH = 84.5
ORIGIN = (206.0, 63.5)
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
# 摇杆遮挡的格子不作为主动收集路线；紧急避碰仍可使用。
OBSCURED = frozenset((x, y) for x in (11, 12) for y in (6, 7))


class TrackingError(RuntimeError):
    pass


def step(cell, direction):
    return ((cell[0] + direction[0]) % COLS, (cell[1] + direction[1]) % ROWS)


def cell_at(point):
    return (int((point[0] - ORIGIN[0]) / PITCH + 0.5) % COLS,
            int((point[1] - ORIGIN[1]) / PITCH + 0.5) % ROWS)


def center(cell):
    return tuple(ORIGIN[i] + PITCH * cell[i] for i in (0, 1))


def displacement(before, after):
    # 跨边界的像素位移也取最短环形距离。
    return tuple((after[i] - before[i] + size * PITCH / 2) % (size * PITCH)
                 - size * PITCH / 2 for i, size in enumerate((COLS, ROWS)))


def motion(before, after):
    dx, dy = displacement(before, after)
    if max(abs(dx), abs(dy)) < 5:
        return None
    if min(abs(dx), abs(dy)) > 12:
        return None  # 一帧内转过弯，不能只按较大的坐标差猜方向。
    return (1 if dx > 0 else -1, 0) if abs(dx) > abs(dy) else (0, 1 if dy > 0 else -1)


def advance(body, target, food):
    """返回新身体和剩余食物；不吃东西时允许进入即将移走的尾格。"""
    grow = target in food
    occupied = body if grow else body[:-1]
    if target in occupied:
        return None
    new_body = (target,) + (body if grow else body[:-1])
    return new_body, food - {target}


def reachable(start, blocked):
    seen = {start}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for direction in DIRECTIONS:
            target = step(cell, direction)
            if target not in blocked and target not in seen:
                seen.add(target)
                queue.append(target)
    return seen


def survival(body):
    # 新头仍能通向尾巴，且有容纳身体的空间，才主动追逐食物。
    area = reachable(body[0], set(body[1:-1]))
    return body[-1] in area and len(area) >= min(len(body) + 2, COLS * ROWS - len(body) + 1)


def choose_direction(body, heading, food):
    """模拟移动尾巴和途中增长，选择可安全到达的小物；无食物时保持安全移动。"""
    food = frozenset(food) - set(body)
    queue = deque([(tuple(body), food, heading, ())]) if food else deque()
    seen = {(tuple(body), food, heading)}
    # 一局只有 60 秒，限制搜索规模；每次只执行第一步并用下一帧重新规划。
    expanded = 0
    while queue and expanded < 1800:
        current, remaining, direction, route = queue.popleft()
        expanded += 1
        if len(route) >= 18:
            continue
        choices = (direction,) + tuple(d for d in DIRECTIONS if d != direction)
        for candidate in choices:
            if candidate == (-direction[0], -direction[1]):
                continue
            target = step(current[0], candidate)
            if target in OBSCURED:
                continue
            moved = advance(current, target, remaining)
            if moved is None:
                continue
            new_body, new_food = moved
            path = route + (candidate,)
            if target in remaining and survival(new_body):
                return path[0]
            key = (new_body, new_food, candidate)
            if key not in seen:
                seen.add(key)
                queue.append((new_body, new_food, candidate, path))
    safe = []
    for candidate in DIRECTIONS:
        if candidate == (-heading[0], -heading[1]):
            continue
        target = step(body[0], candidate)
        moved = advance(tuple(body), target, food)
        if moved:
            new_body, _ = moved
            area = len(reachable(target, set(new_body[1:])))
            safe.append(((survival(new_body), target not in OBSCURED, area, candidate == heading), candidate))
    return max(safe, key=lambda item: item[0])[1] if safe else None


@dataclass
class Tracker:
    """只允许已确认的直走或一次转向，拒绝存在多条身体轨迹的观测。"""
    point: tuple
    observed_at: float
    body: tuple = field(init=False)
    heading: tuple | None = None
    pending: tuple | None = None
    food: set = field(default_factory=set)
    eaten: set = field(default_factory=set)
    period: float | None = None
    last_sample_at: float = field(init=False)

    def __post_init__(self):
        self.body = (cell_at(self.point),)
        self.last_sample_at = self.observed_at

    def see_food(self, food):
        # 暂时被头部遮住的小物不能从记录中消失；已吃过的格子可再次生成。
        observed = set(food) - set(self.body)
        self.eaten.difference_update(observed)
        self.food.update(observed)

    def request(self, direction):
        if self.pending or self.heading is None or direction == self.heading:
            raise TrackingError("不能在转向未确认时叠加指令")
        if direction == (-self.heading[0], -self.heading[1]):
            raise TrackingError("不能直接反向移动")
        self.pending = (self.heading, direction)

    def observe(self, point, now):
        target = cell_at(point)
        old_cell = self.body[0]
        observed_motion = motion(self.point, point)
        if self.heading is None:
            self.heading = observed_motion
        # 用实际像素位移估计速度，静止倒计时不计入移动周期。
        dt = now - self.last_sample_at
        if observed_motion == self.heading and self.heading is not None and not self.pending and dt > 0:
            distance = sum(abs(value) for value in displacement(self.point, point))
            if distance >= 5:
                sample = dt * PITCH / distance
                if 0.03 <= sample <= 2:
                    self.period = sample if self.period is None else 0.7 * self.period + 0.3 * sample
        if target != old_cell:
            if self.heading is None:
                raise TrackingError("尚未确认开局方向")
            elapsed = now - self.observed_at
            # 跳过过多格或绕行半个棋盘后，最短距离不再足以还原真实路径。
            limit = min(5, max(1, int(elapsed / self.period) + 2)) if self.period else 4
            paths = set()
            if self.pending:
                old, new = self.pending
                # 摇杆立即转向或下一格转向，均以实际目标位置验证。
                sequences = [((old,) * lead + (new,) * (length - lead))
                             for length in range(1, limit + 1) for lead in (0, 1) if lead < length]
                sequences.append((old,))  # 下一格才转向时，先看到一格原方向位移。
            else:
                sequences = [(self.heading,) * length for length in range(1, limit + 1)]
            for sequence in sequences:
                cell = old_cell
                route = []
                for direction in sequence:
                    cell = step(cell, direction)
                    route.append(cell)
                if cell == target:
                    paths.add((tuple(route), sequence[-1]))
            if len(paths) != 1:
                raise TrackingError("角色跳格或转向轨迹无法唯一确认")
            route, direction = paths.pop()
            for cell in route:
                moved = advance(self.body, cell, frozenset(self.food))
                if moved is None:
                    raise TrackingError("观测到的角色位置与身体避碰预测不符")
                if cell in self.food:
                    self.eaten.add(cell)
                self.body, remaining = moved
                self.food = set(remaining)
            self.heading = direction
            if self.pending is None or direction == self.pending[1]:
                self.pending = None
        elif observed_motion and self.pending is None and self.heading != observed_motion:
            raise TrackingError("没有转向指令却观测到不同移动方向")
        self.point = point
        self.last_sample_at = now
        # 同格连续截图不重置跨格计时，以免低估移动周期。
        if target != old_cell:
            self.observed_at = now

    def near_center(self):
        if self.heading is None:
            return False
        axis = 0 if self.heading[0] else 1
        return abs(self.point[axis] - center(self.body[0])[axis]) <= 22


def normalized(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text)))


def parse_progress(text):
    match = re.fullmatch(r"今日可完成订单[:：](\d+)/(\d+)", normalized(text))
    if not match:
        return None
    done, total = map(int, match.groups())
    return (done, total) if 0 <= done <= total and total > 0 else None


def parse_quantity(text):
    match = re.fullmatch(r"数量[:：](\d+)件", normalized(text))
    return int(match[1]) if match and int(match[1]) > 0 else None


def production_quantity(result):
    if not result or not result.hit:
        return None
    # 按钮命中“生/产”即可；同一按钮区域内的数字独立用于实际产量记录。
    items = result.all_results or matches(result)
    numbers = [int(number) for item in items
               for number in re.findall(r"\d+", normalized(item.text))]
    return numbers[0] if len(numbers) == 1 and numbers[0] > 0 else None


def parse_mine_storage(text):
    match = re.fullmatch(r"能量石储存数量[:：](\d+)/(\d+)", normalized(text))
    if not match:
        return None
    count, capacity = map(int, match.groups())
    return (count, capacity) if capacity > 0 and 0 <= count <= capacity else None


def parse_tags(text):
    text = normalized(text).replace("素材标签：", "").replace("素材标签:", "")
    tags = []
    for choices in (TYPES, MATERIALS, COLORS):
        found = [tag for tag in choices if tag in text]
        if len(found) > 1:
            return None
        tags.append(found[0] if found else None)
        for tag in found:
            text = text.replace(tag, "")
    # 没给要求的类别可任选；陌生或损坏的标签不能当作“没给”。
    return tuple(tags) if not text and any(tags) else None


def parse_scheme(text):
    # 标签与标题的基线不同，OCR 文本框顺序可能把标题放在标签中间。
    return parse_tags(normalized(text).replace("方案使用素材:", "").replace("方案使用素材：", ""))


def parse_trend_label(text, prefix):
    text = normalized(text)
    match = re.fullmatch(re.escape(prefix) + r"[:：]?(.+)", text)
    label = match[1] if match else None
    return label if label in TYPES + MATERIALS + COLORS else None


def parse_uplift(text):
    found = re.findall(r"价格上涨(\d+)%", normalized(text))
    return int(found[0]) if len(found) == 1 else None


def calendar_label(now=None):
    now = datetime.now(BEIJING) if now is None else now
    day = (now.astimezone(BEIJING) - timedelta(hours=5)).day
    if day <= 2 or day >= 29:
        return "皮革"
    for last_day, label in (
        (11, "红色系"), (16, "蓝色系"), (19, "黄色系"),
        (25, "棕色系"), (28, "绿色系"),
    ):
        if day <= last_day:
            return label


@dataclass(frozen=True)
class Trend:
    today: str
    tomorrow: str
    uplift: int


def production_tags(mode, trend, now=None):
    if mode not in SALE_MODES:
        raise DreamError(f"未知售卖方案：{mode}")
    label = calendar_label(now) if mode == "stock" else trend.today
    if mode == "tomorrow" and trend.uplift != 50:
        label = trend.tomorrow
    if label not in TYPES + MATERIALS + COLORS:
        raise DreamError(f"未知潮流标签：{label}")
    return tuple(label if label in choices else None for choices in (TYPES, MATERIALS, COLORS))


def valid_box(box):
    if box is None:
        return None
    values = tuple(int(v) for v in box)
    return values if len(values) == 4 and values[2] > 0 and values[3] > 0 else None


def matches(result):
    if not result or not result.hit:
        return []
    return result.filtered_results or ([result.best_result] if result.best_result else [])


@dataclass(frozen=True)
class Card:
    stars: int
    box: tuple
    stars_confirmed: bool = False


def star_bands(image, roi, *, axis, minimum, maximum):
    """数橙色星星轮廓的投影段；文字、细边框和尺寸不符的区域不参与计数。"""
    x, y, w, h = roi
    height, width = image.shape[:2]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return []
    crop = image[y:y + h, x:x + w]
    blue, green, red = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
    orange = (red >= 235) & (green >= 115) & (green <= 210) & (blue <= 120)
    active = orange.sum(axis=axis) >= 3
    edges = np.flatnonzero(np.diff(np.r_[False, active, False].astype(np.int8)))
    bands = []
    for start, end in zip(edges[::2], edges[1::2]):
        # 抗锯齿可能使同一颗星的轮廓断开一行/列；相邻星之间至少有两行/列间隔。
        if bands and start - bands[-1][1] <= 1:
            bands[-1] = (bands[-1][0], int(end))
        else:
            bands.append((int(start), int(end)))
    return [(start, end) for start, end in bands if minimum <= end - start <= maximum]


def map_star_count(image, box):
    x, y, w, h = box
    # 一星模板也可能命中多星卡的最下方；以最后一颗星为锚点向上数，保留原命中框用于点击。
    last = y + h - 32
    top = max(0, last - 62)
    bands = star_bands(image, (x + 8, top, 24, last + 14 - top),
                       axis=1, minimum=16, maximum=26)
    centers = [(start + end - 1) / 2 + top for start, end in bands]
    if not 1 <= len(centers) <= 3 or abs(centers[-1] - last) > 4:
        return None
    if any(not 19 <= b - a <= 27 for a, b in zip(centers, centers[1:])):
        return None
    return len(centers)


def detail_star_count(image):
    # 720 短边横屏的“订单详情”右侧星星；区域来自已确认的订单详情截图。
    bands = star_bands(image, (960, 145, 160, 50), axis=0, minimum=25, maximum=42)
    centers = [(start + end - 1) / 2 for start, end in bands]
    if not 1 <= len(centers) <= 3 or not 15 <= centers[0] <= 35:
        return None
    if any(not 43 <= b - a <= 53 for a, b in zip(centers, centers[1:])):
        return None
    return len(centers)


def same_card(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    return intersection >= min(aw * ah, bw * bh) * 0.5


def deduplicate_cards(cards):
    kept = []
    # 优先已确认的星数及完整高星卡，避免其内部一颗星被低星素材重复命中。
    for card in sorted(cards, key=lambda c: (not c.stars_confirmed, -c.stars)):
        if not any(same_card(card.box, other.box) for other in kept):
            kept.append(card)
    return sorted(kept, key=lambda c: (c.stars, c.box[1], c.box[0]))


def check_roi(box, width, height):
    x, y, w, h = box
    left, top = max(0, x + w // 2), max(0, y + h // 4)
    right, bottom = min(width, x + w + w // 2), min(height, y + h + h // 2)
    return [left, top, right - left, bottom - top]


@dataclass
class Order:
    card: Card
    quantity: int
    tags: tuple
    scheduled: int = 0
    harvested: int = 0

    @property
    def production_target(self):
        return self.quantity


@dataclass
class SaleBatch:
    tags: tuple
    scheduled: int = 0
    harvested: int = 0

    @property
    def quantity(self):
        return self.scheduled


@dataclass
class ProductionBatch:
    order: Order | SaleBatch
    quantity: int


class DreamRunner:
    def __init__(self, context):
        self.context = context
        self.pending = {}  # 本次任务工位 -> 实际排产批次；不跨任务、账号共享。

    def image(self):
        if self.context.tasker.stopping:
            raise DreamError("任务已取消")
        controller = self.context.tasker.controller
        screenshot = controller.post_screencap().wait()
        if not screenshot.succeeded:
            raise DreamError("截图失败，不使用上一帧继续判断")
        image = controller.cached_image
        if image is None:
            raise DreamError("截图失败")
        return image

    def reco(self, node, image, roi=None):
        if roi is None:
            result = self.context.run_recognition(node, image)
        else:
            override = {node: {"recognition": {"param": {"roi": roi}}}}
            result = self.context.run_recognition(node, image, pipeline_override=override)
        if result is None:
            raise DreamError(f"识别未能启动：{node}")
        return result

    def hit(self, node, image, roi=None):
        result = self.reco(node, image, roi)
        return bool(result and result.hit)

    def text(self, node, image):
        result = self.reco(node, image)
        # 区域内多个文本框按行、列拼接，不能只读 best_result 的一个标签。
        items = matches(result)
        texts = sorted(
            ((valid_box(item.box), getattr(item, "text", "")) for item in items),
            key=lambda item: ((item[0][1] // 20, item[0][0]) if item[0] else (0, 0)),
        )
        return "".join(text for _, text in texts) or extract_ocr_text(result)

    def box(self, node, image, roi=None):
        result = self.reco(node, image, roi)
        return valid_box(result.box) if result and result.hit else None

    def wait(self, predicate, description, *, deadline=None, confirmations=2):
        deadline = time.monotonic() + 30 if deadline is None else deadline
        consecutive = 0
        while time.monotonic() < deadline:
            value = predicate(self.image())
            if value is not None and value is not False:
                consecutive += 1
                if consecutive == confirmations:
                    return value
            else:
                consecutive = 0
            # 仅限识别轮询的节流；实际结束由页面/工位状态决定。
            time.sleep(0.2)
        raise DreamError(f"等待超时：{description}")

    def click(self, box):
        if self.context.tasker.stopping:
            raise DreamError("任务已取消")
        box = valid_box(box)
        if not box:
            raise DreamError("点击目标无有效识别框")
        result = self.context.run_action_direct("Click", JClick(target=box))
        if not result or not result.success:
            raise DreamError("点击失败")

    def button(self, node, *, deadline=None):
        box = self.wait(lambda image: self.box(node, image), node, deadline=deadline)
        self.click(box)

    def space_ready(self, image):
        return (
            self.hit("导航在织梦空间页面", image)
            and not self.hit("织梦每日奖励文字", image)
            and not self.hit("织梦获得奖励文字", image)
        )

    def sign_in(self):
        self.wait(self.space_ready, "织梦空间页面")
        self.button("织梦今日签到图标")
        self.wait(lambda image: self.hit("织梦每日奖励文字", image), "每日奖励弹窗")
        self.click([550, 625, 180, 55])
        self.wait(lambda image: self.hit("织梦获得奖励文字", image), "奖励弹窗")
        self.click([550, 625, 180, 55])
        self.wait(self.space_ready, "奖励弹窗关闭并回到织梦空间")
        logger.info("织梦空间：今日签到完成")

    def management_ready(self, image, *, full=True):
        return (
            self.hit("织梦经营销售额", image)
            and (not full or (
                self.hit("织梦经营仓库", image)
                and not self.hit("织梦设计页面", image)
                and not self.hit("织梦加速页面", image)
            ))
        )

    def enter_management(self):
        def location(image):
            if self.management_ready(image, full=True):
                return "经营"
            return "空间" if self.space_ready(image) else None

        if self.wait(location, "织梦空间或织梦经营页面") == "空间":
            self.button("织梦经营入口")
            self.wait(self.management_ready, "织梦经营页面")

    def management_entry(self, node):
        self.wait(self.management_ready, "点击入口前确认织梦经营页面")
        if self.context.tasker.stopping:
            raise DreamError("任务已取消")
        result = self.context.run_action(node)
        if not result or not result.success:
            raise DreamError(f"固定位置入口点击失败：{node}")

    def tile_state(self, tile, image, expected=None, *, state_only=False):
        row, col = tile
        if row not in range(3) or col not in range(4):
            raise DreamError(f"无效工位：{tile}")
        x = 640 + col * 141
        choices = (expected,) if isinstance(expected, str) else expected
        if choices is None:
            choices = ("可收获", "空闲中", "生产中", "休息中")
        # 每个框直接对应排、列；不从文字位置反推工位，也不跨列识别。
        regions = []
        if "可收获" in choices:
            regions.append(("织梦工位状态", [x, 145 + row * 164, 137, 48], ("可收获",)))
        lower = tuple(text for text in ("空闲中", "生产中") if text in choices)
        if lower:
            regions.append(("织梦工位状态", [x, 232 + row * 164, 137, 68], lower))
        if "休息中" in choices:
            regions.append(("织梦休息工位", [x, 142 + row * 164, 137, 164], ("休息中",)))
        for node, roi, allowed in regions:
            for item in matches(self.reco(node, image, roi)):
                box = valid_box(item.box)
                text = normalized(item.text).rstrip("!！。.")
                if "生产中" in allowed and re.search("[生产]", text):
                    text = "生产中"
                # 整块 ROI 只可用于生产状态确认，不能拿来点击。
                if not box or (box == tuple(roi) and not (state_only and text == "生产中")):
                    continue
                if text not in allowed:
                    continue
                bx, by, bw, bh = box
                if not (roi[0] <= bx + bw / 2 < roi[0] + roi[2]
                        and roi[1] <= by + bh / 2 < roi[1] + roi[3]):
                    continue
                return text, box
        return None

    def tile_states(self, image, expected=None, *, tiles=None):
        states = {}
        tiles = ((row, col) for row in range(3) for col in range(4)) if tiles is None else sorted(tiles)
        for tile in tiles:
            state = self.tile_state(tile, image, expected, state_only=True)
            if state:
                states[tile] = state
        return states

    def wait_tile(self, tile, expected, *, deadline=None, confirmations=1, state_only=False):
        def predicate(image):
            if not self.management_ready(image, full=False):
                return None
            choices = (expected,) if isinstance(expected, str) else expected
            state = self.tile_state(tile, image, choices, state_only=state_only)
            return state[1] if state else None
        return self.wait(predicate, f"工位 {tile} {expected}", deadline=deadline,
                         confirmations=confirmations)

    def harvest(self, *, deadline=None, tiles=None):
        local_deadline = time.monotonic() + 30
        deadline = local_deadline if deadline is None else min(deadline, local_deadline)
        self.wait(lambda image: self.management_ready(image, full=False),
                  "收获前的织梦经营页面", deadline=deadline, confirmations=1)
        clicked = []
        states = self.tile_states(self.image(), "可收获", tiles=tiles)
        for ready in sorted(states):
            box = self.wait_tile(ready, "可收获", deadline=deadline, confirmations=1)
            self.click(box)
            clicked.append(ready)
        # 先依次收掉当前可收获的工位，再确认空闲/休息并记录实际产量。
        for ready in clicked:
            self.wait_tile(ready, ("空闲中", "休息中"), deadline=deadline)
            batch = self.pending.pop(ready, None)
            if batch is not None:
                order = batch.order
                order.harvested += batch.quantity
                label = "织梦售卖" if isinstance(order, SaleBatch) else "织梦订单"
                logger.info(f"{label}：已收获 {order.harvested}/{order.quantity} 件，标签 {order.tags}")
        return set(clicked)

    def progress(self, image):
        return parse_progress(self.text("织梦订单今日进度", image))

    def map_ready(self, image):
        if self.hit("织梦订单详情标题", image) or self.hit("织梦选择产品页面", image):
            return None
        return self.progress(image) if self.hit("织梦订单页面", image) else None

    def enter_orders(self):
        self.management_entry("织梦经营订单入口")
        self.wait(self.map_ready, "订单驿站地图及今日进度")

    def deliver_available(self):
        while True:
            done, total = self.wait(self.map_ready, "订单地图")
            if done == total:
                return done, total
            # 每交完一单都重新截图、查找；不保存原来“可交付”的列表。
            box = self.box("织梦可交付订单", self.image())
            if box is None:
                return done, total
            # 标签靠近相邻订单；从文字中心向左下偏移，落在本张订单卡片内。
            x, y, w, h = box
            self.click((x + w // 2 - 20, y + h // 2 + 20, 1, 1))
            self.wait(self.order_detail, "订单详情")
            self.button("织梦详情交付按钮")
            self.wait(lambda image: self.hit("织梦选择产品页面", image), "选择产品")
            self.click([319, 221, 17, 11])

            def selected(image):
                if not self.hit("织梦选择产品页面", image):
                    return False
                text = normalized(self.text("织梦产品已选数量", image))
                match = re.fullmatch(r"已选择[:：](\d+)/(\d+)", text)
                return bool(match and int(match[1]) == int(match[2]) and int(match[2]) > 0)

            self.wait(selected, "产品自动选满")
            self.button("织梦产品交付按钮")

            def delivered(image):
                current = self.map_ready(image)
                return current if current == (done + 1, total) else None

            self.wait(delivered, "交付后今日完成数增加一次")
            logger.info(f"织梦订单：今日已完成 {done + 1}/{total}")

    def require_order_images(self):
        # Agent 与 resource 在发布包中同级；开发树的资源在 assets 下。
        root = Path(__file__).resolve().parents[2]
        resource = root / "resource"
        if not resource.is_dir():
            resource = root / "assets/resource"
        missing = [name for name in ORDER_IMAGES if not (resource / "image" / name).is_file()]
        if missing:
            raise DreamError("订单素材未补齐，请放入 resource/image：" + "、".join(missing))

    def completed(self, card, image):
        height, width = image.shape[:2]
        left, top, w, h = check_roi(card.box, width, height)
        # 全屏识别所有对号，再按卡片右下方的位置关联，避免其他订单的对号误判本单。
        for item in matches(self.reco("织梦订单完成对号", image)):
            box = valid_box(item.box)
            if box:
                x, y, bw, bh = box
                if left <= x and top <= y and x + bw <= left + w and y + bh <= top + h:
                    return True
        return False

    def cards(self, image):
        found = []
        for stars, node in ((3, "织梦三星订单"), (2, "织梦二星订单"), (1, "织梦一星订单")):
            for item in matches(self.reco(node, image)):
                box = valid_box(item.box)
                if box:
                    count = map_star_count(image, box)
                    found.append(Card(count or stars, box, count is not None))
        return deduplicate_cards(found)

    def order_detail(self, image):
        if not self.hit("织梦订单详情标题", image):
            return None
        # 元组让星级未知时也结束原有页面等待，不增加数星重试或固定延迟。
        return (detail_star_count(image),)

    def read_orders(self, remaining):
        cards = self.cards(self.image())
        orders = []
        for card in cards:
            image = self.image()
            if self.completed(card, image):
                continue
            # 下单前用新截图重认卡片和对应右下角对号。
            current = next((c for c in self.cards(image) if same_card(c.box, card.box)), None)
            if current is None or self.completed(current, image):
                continue
            if self.box("织梦可交付订单", image) is not None:
                raise DreamError("读取订单时出现可交付订单，请重新执行以重新交付")
            self.click(current.box)
            detail_stars, = self.wait(self.order_detail, "读取订单详情")
            if detail_stars is not None:
                current = Card(detail_stars, current.box, True)
            else:
                logger.warning("织梦订单：详情页星级未确认，保留地图结果并继续读取数量及素材")

            def requirements(image):
                quantity = parse_quantity(self.text("织梦订单数量", image))
                tags = parse_tags(self.text("织梦订单素材标签", image))
                return (quantity, tags) if quantity and tags is not None else None

            quantity, tags = self.wait(requirements, "订单数量及素材标签")
            orders.append(Order(current, quantity, tags))
            star_text = f"{current.stars} 星" if current.stars_confirmed else "星级未确认"
            logger.info(f"织梦订单：{star_text}，位置 {current.box}，需求 {quantity} 件，标签 {tags}")
            self.click([250, 70, 100, 60])  # 已确认点击左侧地图空白关闭详情。
            self.wait(self.map_ready, "关闭详情回到订单地图")
        if len(orders) != remaining:
            raise DreamError(f"尚有 {remaining} 个订单，但仅识别 {len(orders)} 个未完成订单；请检查星级/对号素材")
        return orders

    def choose_category(self, image, node, choices, requested, *, deadline=None):
        deadline = time.monotonic() + PRODUCTION_CONFIRM_SECONDS if deadline is None else deadline
        target = requested
        next_click = 0.0
        swiped = False
        while time.monotonic() < deadline:
            if self.context.tasker.stopping:
                raise DreamError("任务已取消")
            if self.hit("织梦设计页面", image):
                options = {
                    normalized(item.text): valid_box(item.box)
                    for item in matches(self.reco(node, image))
                }
                if target is None:
                    target = next((text for text in choices if options.get(text)), None)
                # 其他类别尚未选择时可显示 XX，只确认当前类别；完整方案仍由 schedule 严格核验。
                selected = normalized(self.text("织梦方案使用素材", image))
                if target and [text for text in choices if text in selected] == [target]:
                    return target
                box = options.get(target)
                now = time.monotonic()
                if box and next_click <= now < deadline:
                    x, y, w, h = box
                    # OCR 框边缘可能超出按钮热区，固定点在本次识别框的中心。
                    self.click((x + w // 2, y + h // 2, 1, 1))
                    next_click = time.monotonic() + 0.5
                elif target == "特殊" and not box and not swiped and now < deadline:
                    action = self.context.run_action_direct(
                        "Swipe", JSwipe(begin=[1063, 342], end=[[945, 345]])
                    )
                    if not action or not action.success:
                        raise DreamError("查找特殊材质时左划失败")
                    swiped = True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.2, remaining))
            image = self.image()
        raise DreamError(f"等待超时：生产选项 {target or node} 已选中")

    def option_box(self, node, target, image):
        for item in matches(self.reco(node, image)):
            if normalized(item.text) == target:
                return valid_box(item.box)
        return None

    def submit_production(self, *, deadline):
        clicked = False
        quantity = None
        while time.monotonic() < deadline:
            image = self.image()
            if clicked and self.management_ready(image, full=True):
                return quantity
            result = self.reco("织梦生产五件按钮", image) if self.hit("织梦设计页面", image) else None
            box = valid_box(result.box) if result and result.hit else None
            amount = production_quantity(result) if box else None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if box and amount is not None:
                # 每点一次就重新识别页面和按钮，不能在返回工位页后沿用旧坐标。
                # 必须执行识别+动作节点，直接调用 Custom 会让 SDK 收到 reco_id=0。
                result = self.context.run_task(
                    "织梦生产五件按钮",
                    {"织梦生产五件按钮": {"action": {"param": {
                        "custom_action_param": {
                            "max_retries": 1,
                            "timeout": 0,
                            "after_click_delay": min(300, int(remaining * 1000)),
                        },
                    }}}},
                )
                if not result or not result.status.succeeded:
                    raise DreamError("点击生产按钮失败")
                clicked = True
                quantity = amount
            else:
                time.sleep(min(0.2, remaining))
        raise DreamError("等待超时：识别生产件数、点击生产并返回工位页")

    def schedule(self, tile, order, *, deadline=None):
        if tile in self.pending:
            raise DreamError(f"工位 {tile} 已有生产记录，禁止重复排产")
        local_deadline = time.monotonic() + PRODUCTION_CONFIRM_SECONDS
        deadline = local_deadline if deadline is None else min(deadline, local_deadline)
        self.click(self.wait_tile(tile, "空闲中", deadline=deadline))
        self.wait(lambda image: self.hit("织梦设计页面", image), "设计生产方案", deadline=deadline, confirmations=1)
        selected = []
        for node, choices, requested in zip(
            ("织梦版型选项", "织梦材质选项", "织梦颜色选项"),
            (TYPES, MATERIALS, COLORS), order.tags,
        ):
            selected.append(
                self.choose_category(
                    self.image(), node, choices, requested, deadline=deadline
                )
            )
        self.wait(
            lambda image: parse_scheme(self.text("织梦方案使用素材", image)) == tuple(selected),
            "生产方案三个已选标签", deadline=deadline, confirmations=1,
        )
        quantity = self.submit_production(deadline=deadline)
        # 返回工位页后只确认生产状态，不再重复点击生产按钮。
        self.wait_tile(tile, "生产中", deadline=deadline, confirmations=1, state_only=True)
        self.pending[tile] = ProductionBatch(order, quantity)
        order.scheduled += quantity
        label = "织梦售卖" if isinstance(order, SaleBatch) else "织梦订单"
        logger.info(f"{label}：已开始生产 {order.scheduled}/{order.quantity} 件，工位 {tile}")

    def accelerate(self, tile, deadline):
        # 调度已确认生产；点击前只需一次新框复核，避免再次卡在连续命中。
        click_deadline = min(deadline, time.monotonic() + PRODUCTION_CONFIRM_SECONDS)
        self.click(self.wait_tile(tile, "生产中", deadline=click_deadline, confirmations=1))
        self.wait(lambda image: self.hit("织梦加速页面", image), "生产加速页面", deadline=deadline)
        self.button("织梦立即完成按钮", deadline=deadline)
        # 扣能量石只点一次；动画未完成时继续认结果，不能再次扣除。
        self.wait_tile(tile, "可收获", deadline=deadline)
        return self.harvest(deadline=deadline, tiles=(tile,))

    def produce(self, orders, use_stone):
        states = self.wait(self.ready_tiles, "订单生产前的工位状态", confirmations=1)
        known_tiles = set(states)
        if self.pending or any(state[0] not in ("空闲中", "休息中") for state in states.values()):
            self.finish_tiles(tuple(known_tiles), use_stone, time.monotonic() + NATURAL_WAIT_SECONDS)
        while any(order.harvested < order.production_target for order in orders):
            states = self.wait(lambda image: self.ready_tiles(image, known_tiles), "本轮订单空闲工位", confirmations=1)
            known_tiles.update(states)
            if any(state[0] not in ("空闲中", "休息中") for state in states.values()):
                raise DreamError("本轮开始时仍有工位未空闲")
            tiles = sorted(tile for tile, state in states.items() if state[0] == "空闲中")
            if not tiles:
                raise DreamError("所有已解锁工位已休息，但订单需求尚未完成")
            deadline = time.monotonic() + NATURAL_WAIT_SECONDS
            batch_tiles = []
            for tile in tiles:
                order = next((order for order in orders if order.scheduled < order.production_target), None)
                if order is None:
                    break
                self.schedule(tile, order, deadline=deadline)
                batch_tiles.append(tile)
            if not batch_tiles:
                raise DreamError("订单生产记录与收获数量不一致，停止以避免重复生产")
            self.finish_tiles(tuple(batch_tiles), use_stone, deadline)
            if self.pending:
                raise DreamError("本轮订单未全部收获，停止以避免重复生产")

    def orders(self, use_stone):
        self.require_order_images()
        self.enter_management()
        states = self.wait(self.ready_tiles, "已有工位状态")
        self.finish_tiles(tuple(states), use_stone, time.monotonic() + NATURAL_WAIT_SECONDS)
        self.enter_orders()
        done, total = self.deliver_available()
        if done == total:
            return
        orders = self.read_orders(total - done)
        self.button("织梦订单返回按钮")
        self.wait(self.management_ready, "订单返回织梦经营")
        self.produce(orders, use_stone)
        self.enter_orders()
        done, total = self.deliver_available()
        if done != total:
            raise DreamError("记录的订单需求已生产并收获，但仍有订单不可交付；停止以避免重复生产")
        logger.info("织梦空间：今日订单完成")

    def shop_frame(self, image):
        # 弹窗关闭与列表就绪在同一张新截图内判断，避免背景文字提前放行。
        if (
            not self.hit("织梦产品售卖页面", image)
            and self.hit("织梦售卖橱窗页面", image)
            and self.hit("织梦售卖产品列表", image)
        ):
            return image
        return None

    def enter_shop(self):
        self.management_entry("织梦售卖橱窗入口")
        return self.wait(self.shop_frame, "售卖橱窗产品列表", confirmations=1)

    def trend(self, image):
        today = parse_trend_label(self.text("织梦今日潮流", image), "今日潮流")
        tomorrow = parse_trend_label(self.text("织梦明日预告", image), "明日预告")
        uplift = parse_uplift(self.text("织梦潮流涨价", image))
        return Trend(today, tomorrow, uplift) if today and tomorrow and uplift is not None else None

    def wait_trend(self):
        def predicate(image):
            return self.trend(image) if self.shop_frame(image) is not None else None
        return self.wait(predicate, "今日潮流、涨价比例和明日预告", confirmations=1)

    def ready_tiles(self, image, tiles=()):
        if not self.management_ready(image):
            return None
        states = self.tile_states(image)
        # 已记录的工位识别丢失不能被当成生产结束。
        return states if states and set(tiles).issubset(states) else None

    def finish_tiles(self, tiles, use_stone, deadline):
        known_tiles = set(tiles)
        if not known_tiles:
            return {}

        def observe(image, expected=None):
            if not self.management_ready(image, full=False):
                return None
            states = self.tile_states(image, expected, tiles=known_tiles)
            return states if known_tiles.issubset(states) else None

        states = self.wait(observe, "织梦经营工位状态", deadline=deadline, confirmations=1)
        remaining = {tile for tile, state in states.items()
                     if state[0] not in ("空闲中", "休息中") or tile in self.pending}
        if not remaining:
            return states
        while remaining:
            def ready(image):
                if not self.management_ready(image, full=False):
                    return None
                expected = ("可收获", "生产中") if use_stone else "可收获"
                return self.tile_states(image, expected, tiles=remaining) or None
            states = self.wait(ready, "本轮未收获工位", deadline=deadline, confirmations=1)
            ready_tiles = [tile for tile, state in states.items() if state[0] == "可收获"]
            if ready_tiles:
                remaining.difference_update(self.harvest(deadline=deadline, tiles=ready_tiles))
            elif use_stone:
                tile = min(states)
                remaining.difference_update(self.accelerate(tile, deadline))
        return self.wait(
            lambda image: observe(image, ("空闲中", "休息中")), "本轮收获后工位状态",
            deadline=deadline, confirmations=1,
        )

    def shop_produce(self, tags, rounds, use_stone):
        states = self.wait(self.ready_tiles, "已有工位状态", confirmations=1)
        self.finish_tiles(tuple(states), use_stone, time.monotonic() + NATURAL_WAIT_SECONDS)
        if self.pending:
            raise DreamError("开始橱窗生产前仍有未收获的记录工位")
        # 进入时已有生产不计入轮数；每轮所有可用工位各生产一批，件数以按钮为准。
        known_tiles = set(states)
        number = 0
        while rounds == 0 or number < rounds:
            states = self.wait(
                lambda image: self.ready_tiles(image, known_tiles), "本轮全部工位状态",
                confirmations=1,
            )
            known_tiles.update(states)
            if any(state[0] not in ("空闲中", "休息中") for state in states.values()):
                raise DreamError("本轮开始时仍有工位未空闲")
            tiles = tuple(sorted(tile for tile, state in states.items() if state[0] == "空闲中"))
            if not tiles:
                logger.info(f"织梦售卖：所有已解锁工位已休息，生产结束，实际完成 {number} 轮")
                return number
            batches = []
            deadline = None
            for tile in tiles:
                if deadline is None:
                    deadline = time.monotonic() + NATURAL_WAIT_SECONDS
                batch = SaleBatch(tags)
                self.schedule(tile, batch, deadline=deadline)
                batches.append(batch)
            self.finish_tiles(tiles, use_stone, deadline)
            if self.pending or any(batch.scheduled <= 0 or batch.harvested != batch.scheduled for batch in batches):
                raise DreamError("本轮产品未全部确认收获，停止售卖流程")
            number += 1
            limit = "不限" if rounds == 0 else str(rounds)
            logger.info(f"织梦售卖：已完成 {number}/{limit} 轮，共收获 {sum(batch.harvested for batch in batches)} 件")
        return number

    def sell_products(self):
        while True:
            image = self.wait(self.shop_frame, "售出后返回产品列表", confirmations=1)
            boxes = [valid_box(item.box) for item in matches(self.reco("织梦潮流产品", image))]
            boxes = [box for box in boxes if box]
            if not boxes:
                logger.info("织梦售卖：列表已无潮流UP产品，售卖结束")
                return
            x, y, w, h = min(boxes, key=lambda box: (box[1], box[0]))
            target = (x + w // 2 + 70, y + h // 2 + 70, 1, 1)
            height, width = image.shape[:2]
            if not (0 <= target[0] < width and 0 <= target[1] < height):
                raise DreamError("潮流产品偏移点击位置超出截图")
            self.click(target)

            def popup_button(image, node):
                return self.box(node, image) if self.hit("织梦产品售卖页面", image) else None
            self.click(self.wait(
                lambda image: popup_button(image, "织梦售卖MAX按钮"),
                "产品售卖MAX", confirmations=1,
            ))
            self.click(self.wait(
                lambda image: popup_button(image, "织梦确认售出按钮"),
                "确认售出按钮", confirmations=1,
            ))
            # 确认售出只点一次，随后重认列表；不比较库存与所选数量。

    def shop(self, mode, rounds, use_stone, *, after_orders=False):
        if mode not in SALE_MODES or type(rounds) is not int or rounds < 0:
            raise DreamError("售卖方案或工位循环次数无效，请填写非负整数")
        if after_orders:
            self.button("织梦订单返回按钮")
            self.wait(self.management_ready, "订单驿站返回后开始橱窗任务")
        else:
            self.enter_management()
        self.enter_shop()
        initial = self.wait_trend()
        tags = production_tags(mode, initial)
        limit = "不限" if rounds == 0 else str(rounds)
        logger.info(f"织梦售卖：方案 {SALE_MODES[mode]}，生产标签 {tags}，计划 {limit} 轮")
        self.button("织梦橱窗返回按钮")
        self.wait(self.management_ready, "橱窗返回织梦经营后生产")
        self.shop_produce(tags, rounds, use_stone)
        self.enter_shop()
        # 完成全部生产后才售卖；跨越五点时按重新读取的涨价比例判断。
        final = self.wait_trend()
        if mode == "current" or final.uplift == 50:
            self.sell_products()
        else:
            logger.info(f"织梦售卖：当前涨价 {final.uplift}%，本次产品继续囤积")

    def mine_state(self, image):
        if self.hit("织梦获得奖励文字", image) or not self.hit("织梦能量晶矿页面", image):
            return None
        return parse_mine_storage(self.text("织梦能量晶矿库存", image))

    def mine(self):
        self.enter_management()
        self.management_entry("织梦能量晶矿入口")
        count, capacity = self.wait(self.mine_state, "能量晶矿页面及储存数量")
        if count > 0:
            # 灰色领取按钮也有文字，必须先识别库存，不能仅凭按钮文字点击。
            self.button("织梦能量晶矿领取按钮")
            self.wait(lambda image: self.hit("织梦获得奖励文字", image), "领取能量石奖励弹窗")
            self.click([550, 625, 180, 55])
            self.wait(
                lambda image: self.mine_state(image) == (0, capacity),
                "奖励弹窗关闭且能量晶矿库存归零",
            )
            logger.info(f"织梦能量晶矿：已领取 {count} 颗能量石")
        else:
            logger.info("织梦能量晶矿：暂无能量石可领取")
        self.click([550, 625, 180, 55])
        self.wait(self.management_ready, "关闭能量晶矿回到织梦经营")

    def seven_ready(self, image):
        return self.hit("七日妙想页面", image)

    def enter_seven(self):
        deadline = time.monotonic() + 30

        def location(image):
            if self.seven_ready(image):
                return "七日"
            if self.space_ready(image):
                return "空间"
            if self.shop_frame(image) is not None:
                return "橱窗"
            if self.map_ready(image) is not None:
                return "订单"
            if self.management_ready(image):
                return "经营"
            return None

        # 经营可能结束在橱窗、订单地图或经营页；每次返回都确认实际落点。
        for _ in range(4):
            page = self.wait(location, "进入七日妙想前确认所在页面", deadline=deadline)
            if page == "七日":
                return
            if page == "空间":
                self.button("七日妙想入口", deadline=deadline)
                self.wait(self.seven_ready, "七日妙想页面", deadline=deadline)
                return
            node = {
                "橱窗": "织梦橱窗返回按钮",
                "订单": "织梦订单返回按钮",
                "经营": "织梦经营返回按钮",
            }[page]
            self.button(node, deadline=deadline)
            destination = self.space_ready if page == "经营" else self.management_ready
            self.wait(destination, "返回后确认页面", deadline=deadline)
        raise DreamError("未能从经营返回织梦空间")

    def seven_titles(self, image):
        titles = []
        for item in matches(self.reco("七日妙想任务标题", image)):
            box = valid_box(item.box)
            if not box:
                continue
            center = box[0] + box[2] / 2
            column = int((center - 55) // 300)
            # 最右侧露出的半张卡片不作为可点击任务。
            if 0 <= column < 4 and 55 + column * 300 <= box[0] and box[0] + box[2] <= 315 + column * 300:
                titles.append((column, normalized(getattr(item, "text", ""))))
        return sorted(titles)

    def return_from_seven(self):
        deadline = time.monotonic() + 30

        def location(image):
            if self.hit("织梦空间专属加载中", image):
                return None
            if self.space_ready(image):
                return "空间", None
            if self.seven_ready(image):
                box = self.box("七日分享返回按钮", image)
                return ("七日", box) if box else None
            return None

        page, box = self.wait(location, "七日妙想后续任务前确认页面", deadline=deadline)
        if page == "七日":
            self.click(box)
            self.wait(
                lambda image: not self.hit("织梦空间专属加载中", image) and self.space_ready(image),
                "退出七日妙想加载结束后回到织梦空间",
                deadline=deadline,
            )

    def seven_candidate(self, image, completed):
        if self.seven_card_complete(image, 0):
            return "全部完成", None, None
        titles = self.seven_titles(image)
        for column, text in titles:
            task = re.fullmatch(r"制作1件(连衣裙|上衣|下装)", text)
            kind = task[1] if task else {
                "分享1次定制成衣": "分享", "进行1次人台换衣": "换衣",
                "完成1次灵感拾趣": "拾趣", "进行1次灵感讨论": "讨论",
            }.get(text)
            if kind is None or kind in completed:
                continue
            if self.seven_card_complete(image, column):
                continue
            if kind in ("换衣", "拾趣"):
                # 没有前往完成按钮的任务从织梦空间入口进入。
                return kind, column, None
            box = self.box("七日妙想前往完成", image, [55 + column * 300, 631, 260, 67])
            if box is not None:
                return kind, column, box
        return None

    def seven_card_complete(self, image, column):
        return self.hit("七日妙想完成对号", image, [55 + 300 * column, 540, 280, 160])

    def seven_current(self, completed):
        previous = None

        def settled(image):
            nonlocal previous
            if not self.seven_ready(image):
                previous = None
                return None
            # 第一张完整卡片已完成，按业务规则结束全部任务；无需再读标题。
            if self.seven_card_complete(image, 0):
                return (("全部完成", None, None),)
            titles = self.seven_titles(image)
            if not titles or titles != previous:
                previous = titles or None
                return None
            # 包装 None，区分“没有要做的任务”和“页面尚未就绪”。
            return (self.seven_candidate(image, completed),)

        return self.wait(settled, "七日妙想当前卡片就绪")[0]

    def seven_open_task(self, category, column, box):
        deadline = time.monotonic() + 30
        if category == "讨论":
            x, y, w, h = box
            self.click((x + w // 2, y + h // 2, 1, 1))
        else:
            self.click(box)
        unchanged = 0
        while time.monotonic() < deadline:
            image = self.image()
            if self.hit("七日制衣新建", image):
                return
            if self.seven_ready(image):
                title = {"分享": "分享1次定制成衣", "讨论": "进行1次灵感讨论"}.get(
                    category, "制作1件" + category,
                )
                same_task = (column, title) in self.seven_titles(image)
                target = self.box("七日妙想前往完成", image, [55 + column * 300, 631, 260, 67]) if same_task else None
                if target:
                    unchanged += 1
                    if unchanged == 2:
                        # 仅前往完成允许重试，而且必须仍是同一张任务卡。
                        if category == "讨论":
                            x, y, w, h = target
                            self.click((x + w // 2, y + h // 2, 1, 1))
                        else:
                            self.click(target)
                        unchanged = 0
                else:
                    unchanged = 0
            else:
                unchanged = 0
            time.sleep(0.2)
        raise DreamError("前往完成后未进入制衣页面")

    def seven_action(self, node):
        if self.context.tasker.stopping:
            raise DreamError("任务已取消")
        result = self.context.run_action(node)
        if not result or not result.success:
            raise DreamError(f"七日制衣动作失败：{node}")

    def seven_button_center(self, node, *, deadline=None):
        x, y, w, h = self.wait(lambda image: self.box(node, image), node, deadline=deadline)
        self.click((x + w // 2, y + h // 2, 1, 1))

    def seven_back_until_ready(self):
        deadline = time.monotonic() + 30
        next_click = 0.0
        confirmations = 0
        while time.monotonic() < deadline:
            image = self.image()
            if self.seven_ready(image):
                confirmations += 1
                if confirmations == 2:
                    return
            else:
                confirmations = 0
                box = None if self.hit("织梦空间专属加载中", image) else self.box("七日分享返回按钮", image)
                if box and next_click <= time.monotonic() < deadline:
                    x, y, w, h = box
                    self.click((x + w // 2, y + h // 2, 1, 1))
                    next_click = time.monotonic() + 0.5
            time.sleep(0.2)
        raise DreamError("逐次返回后未回到七日妙想页面")

    def seven_discussion(self):
        self.seven_button_center("七日制衣新建")
        self.seven_choose_custom("讨论")
        self.seven_button_center("七日灵感讨论灯泡")
        self.seven_button_center("七日灵感讨论我再想想")
        self.seven_back_until_ready()

        def completed(image):
            if not self.seven_ready(image):
                return False
            return any(text == "进行1次灵感讨论" and self.seven_card_complete(image, column)
                       for column, text in self.seven_titles(image))

        self.wait(completed, "灵感讨论任务完成对号")

    def seven_claim_rewards(self):
        deadline = time.monotonic() + 30

        def state(image):
            if self.hit("织梦获得奖励文字", image):
                return "奖励"
            if not self.seven_ready(image):
                return None
            return "可领取" if self.hit("七日黄色礼盒可领取", image) else "无需领取"

        def dismiss():
            self.seven_action("七日奖励点击空白")
            self.wait(lambda image: not self.hit("织梦获得奖励文字", image) and self.seven_ready(image),
                      "关闭七日奖励后回到七日妙想", deadline=deadline)

        current = self.wait(state, "七日妙想黄色礼盒状态", deadline=deadline)
        while current == "奖励":
            dismiss()
            current = self.wait(state, "关闭已有奖励后检查黄色礼盒", deadline=deadline)
        if current == "无需领取":
            return
        # 只在金色盒身和红色领取标记同时命中后点一次，不依赖外围涟漪。
        self.seven_action("七日黄色礼盒点击")
        self.wait(lambda image: self.hit("织梦获得奖励文字", image), "七日妙想恭喜获得", deadline=deadline)
        dismiss()
        logger.info("七日妙想：已领取黄色礼盒奖励")

    def seven_clothes_list(self, image):
        if not self.hit("七日制衣切换部位", image):
            return None
        for item in matches(self.reco("七日制衣右栏文字", image)):
            box = valid_box(item.box)
            if box and getattr(item, "text", "").strip() and 196 <= box[1] + box[3] / 2 <= 235:
                return True
        return None

    def seven_material_ready(self, image):
        return self.hit("七日制衣面料选择", image) and self.hit("七日制衣生成成衣", image)

    def seven_works(self, image):
        if self.hit("七日分享成衣作品", image):
            kind = "成衣"
        elif self.hit("七日分享作品草稿", image):
            kind = "草稿"
        else:
            return None
        counts = []
        for item in matches(self.reco("七日分享作品数量", image)):
            count = re.fullmatch(r"(\d+)/(\d+)", normalized(item.text))
            if count and 0 <= int(count[1]) <= int(count[2]) and int(count[2]) > 0:
                counts.append(int(count[1]))
        # 元组保留 0，避免将空作品列表当成未就绪而反复等待。
        return (kind, counts[0]) if len(counts) == 1 else None

    def seven_share_page(self, image):
        return self.hit("七日分享按钮", image) and self.hit("七日分享保存本地", image)

    def seven_qq_ready(self, image):
        return self.hit("七日分享QQ新用户", image) and self.hit("七日分享QQ登录", image)

    def seven_share(self):
        self.button("七日制衣连衣裙")
        self.wait(self.seven_clothes_list, "类别衣服列表及首件名称")
        self.seven_action("七日制衣选择首件")
        kind, count = self.wait(self.seven_works, "首件衣服的成衣作品或作品草稿及数量")
        if count == 0:
            raise DreamError(f"首件连衣裙的{kind}作品为空，无法完成分享任务")

        if kind == "草稿":
            self.seven_action("七日分享选择首个草稿")
            x, y, w, h = self.wait(
                lambda image: self.box("七日制衣生成成衣", image), "选中草稿后的生成成衣按钮",
            )
            self.click((x + w // 2, y + h // 2, 1, 1))
            self.seven_rename(for_share=True)
        else:
            self.seven_action("七日分享选择首个作品")

            def selected(image):
                if not self.hit("七日分享成衣作品", image) or not self.hit("七日分享前往编辑", image):
                    return None
                return self.box("七日分享作品图标", image)

            self.click(self.wait(selected, "选中首个作品后的分享图标"))

        def result_share(image):
            if self.hit("织梦空间专属加载中", image) or not self.hit("七日制衣离开制衣", image):
                return None
            return self.box("七日分享成衣图标", image)

        self.click(self.wait(result_share, "专属加载结束后的成衣分享图标"))
        self.wait(self.seven_share_page, "分享及保存本地页面")
        self.button("七日分享按钮")
        self.button("七日分享QQ渠道")
        # QQ 页面识别成功才关闭应用；不依赖固定等待，不进入联系人或发送消息。
        self.wait(self.seven_qq_ready, "QQ新用户及登录页面")
        self.seven_action("七日分享关闭QQ")
        self.wait(self.seven_share_page, "关闭QQ后返回游戏分享页面")
        self.button("七日分享返回按钮")
        self.button("七日制衣离开制衣")
        self.wait(self.seven_ready, "分享后离开制衣回到七日妙想")

    def seven_choose_custom(self, category):
        deadline = time.monotonic() + 30
        destination = "七日灵感讨论灯泡" if category == "讨论" else "七日制衣" + category
        clicks = 0
        next_click = 0.0
        confirmations = 0
        while time.monotonic() < deadline:
            image = self.image()
            # 新建前的作品页也有类别，必须先实际点击定制对话，且作品页已离开。
            ready = (clicks > 0 and not self.hit("七日制衣作品页", image)
                     and self.hit(destination, image))
            if time.monotonic() >= deadline:
                break
            if ready:
                confirmations += 1
                if confirmations >= 2:
                    return
                time.sleep(0.1)
                continue  # 第一帧命中后也不再点击旧按钮，下一帧确认。
            confirmations = 0
            box = self.box("七日制衣我来定制", image)
            if box and next_click <= time.monotonic() < deadline:
                if category == "讨论":
                    x, y, w, h = box
                    self.click((x + w // 2, y + h // 2, 1, 1))
                else:
                    self.click(box)
                clicks += 1
                next_click = time.monotonic() + 0.5
            time.sleep(0.1)
        target = "灵感讨论灯泡" if category == "讨论" else category + "类别"
        raise DreamError(f"我来定制后未出现{target}，已点击{clicks}次")

    def seven_input_name(self, node="七日制衣输入临时"):
        if self.context.tasker.stopping:
            raise DreamError("任务已取消")
        # Custom 回调需要有效的识别结果；run_action 会传入 reco_id=0。
        # 此节点没有后继，只执行一次默认 DirectHit 识别和文本输入。
        result = self.context.run_task(node)
        if not result or not result.status.succeeded:
            raise DreamError(f"七日制衣动作失败：{node}")

    def seven_rename(self, *, for_share=False):
        name = "分享成衣" if for_share else "临时"
        input_node = "七日分享输入名称" if for_share else "七日制衣输入临时"
        name_node = "七日分享名称已输入" if for_share else "七日制衣名称已输入"

        def confirm(image):
            if not self.hit("七日制衣名称弹窗", image):
                return None
            return self.box("七日制衣名称确定", image)

        confirm_box = self.wait(confirm, "更改作品名称弹窗及确定按钮")
        self.button("七日制衣点击名称框")
        self.seven_input_name(input_node)
        self.wait(lambda image: self.hit(name_node, image), f"作品名称输入{name}")
        # 第一次点击原确定位置只收起输入框；名称落到游戏弹窗后才提交。
        self.click(confirm_box)
        self.wait(
            lambda image: self.hit("七日制衣名称弹窗", image)
            and self.hit(name_node, image),
            f"收起输入框后作品名称显示{name}",
        )
        self.button("七日制衣名称确定")

    def seven_make(self, category):
        self.button("七日制衣新建")
        self.seven_choose_custom(category)
        self.button("七日制衣" + category)
        self.wait(self.seven_clothes_list, "类别衣服列表及首件名称")
        self.seven_action("七日制衣选择首件")
        self.wait(self.seven_material_ready, "面料选择及生成成衣按钮")
        self.seven_action("七日制衣选择首块面料")
        self.button("七日制衣生成成衣")
        self.seven_rename()

        def finished(image):
            if self.hit("织梦空间专属加载中", image):
                return None
            return self.box("七日制衣离开制衣", image)

        # 加载页可以不出现；一个期限覆盖加载和生成结果页的等待。
        self.click(self.wait(finished, "生成成衣后离开制衣"))
        self.wait(self.seven_ready, "离开制衣后回到七日妙想")

    def seven_mannequin_list(self, image):
        if not self.hit("七日人台展示换新", image):
            return None
        for item in matches(self.reco("七日人台衣服列表文字", image)):
            box = valid_box(item.box)
            if box and getattr(item, "text", "").strip() and 312 <= box[1] + box[3] / 2 <= 345:
                return True
        return None

    def seven_change_mannequin(self):
        def back(image):
            return self.box("七日分享返回按钮", image) if self.seven_ready(image) else None

        def space_after_loading(image):
            return not self.hit("织梦空间专属加载中", image) and self.space_ready(image)

        self.click(self.wait(back, "退出七日妙想的返回箭头"))
        self.wait(space_after_loading, "退出七日妙想加载结束后回到织梦空间")
        self.button("七日人台换衣入口")
        self.wait(self.seven_mannequin_list, "展示换新及首件衣服名称")
        self.seven_action("七日人台选择首件")

        def save(image):
            if not self.hit("七日人台展示换新", image) or not self.hit("七日人台选中对号", image):
                return None
            return self.box("七日人台保存", image)

        self.click(self.wait(save, "首件衣服已选中及保存按钮"))
        self.wait(
            space_after_loading,
            "保存换衣加载结束后回到织梦空间",
        )
        self.button("七日妙想入口")
        self.wait(self.seven_ready, "人台换衣后回到七日妙想")

    def inspiration_head(self, image):
        points = []
        for item in matches(self.reco("灵感拾趣角色", image)):
            box = valid_box(item.box)
            if box:
                # 素材只取头和上身；固定外观的棋盘锚点在头部中心下方 29 像素。
                point = (box[0] + box[2] / 2, box[1] + box[3] / 2 + 29)
                if not any(abs(point[0] - p[0]) < 15 and abs(point[1] - p[1]) < 15 for p in points):
                    points.append(point)
        return points[0] if len(points) == 1 else None

    def inspiration_food(self, image):
        cells = set()
        for item in matches(self.reco("灵感拾趣小物", image)):
            box = valid_box(item.box)
            if box:
                cells.add(cell_at((box[0] + box[2] / 2, box[1] + box[3] / 2)))
        return cells

    def inspiration_result(self, image):
        return (self.hit("灵感拾趣结算得分", image)
                and self.hit("灵感拾趣结算数量", image)
                and self.hit("灵感拾趣结算分享", image))

    def inspiration_turn(self, direction):
        if self.context.tasker.stopping:
            raise DreamError("任务已取消")
        x, y = 1157, 597
        # 80ms 短拖动后松开，下一次转向必须先得到角色位移反馈。
        action = JSwipe(begin=[x, y, 1, 1],
                        end=[[x + 60 * direction[0], y + 60 * direction[1], 1, 1]],
                        duration=[80])
        result = self.context.run_action_direct("Swipe", action)
        if not result or not result.success:
            raise DreamError("灵感拾趣摇杆转向失败")

    def inspiration_play(self):
        deadline = time.monotonic() + 90
        tracker = None
        controlled = True
        lost_since = None
        next_food_scan = 0.0
        pending_since = None
        while time.monotonic() < deadline:
            image = self.image()
            sampled_at = time.monotonic()
            point = self.inspiration_head(image)
            if point is None:
                # 结算不依赖胜负标题；先确认真实结果，再退出这一局。
                if self.inspiration_result(image):
                    if tracker is None:
                        logger.warning("灵感拾趣：未捕捉到开局倒计时，本局未主动转向")
                    self.wait(self.inspiration_result, "灵感拾趣结算页面", deadline=deadline)
                    return
                lost_since = sampled_at if lost_since is None else lost_since
                if tracker and controlled and sampled_at - lost_since > 2:
                    logger.warning("灵感拾趣：角色持续未识别，停止转向并等待本局结算")
                    controlled = False
                time.sleep(0.05)
                continue
            lost_since = None
            if tracker is None:
                # 必须见到开局倒计时，不能在中途把未知身体当成只有一格。
                if self.hit("灵感拾趣倒计时", image) and self.hit("灵感拾趣游戏得分", image):
                    tracker = Tracker(point, sampled_at)
                time.sleep(0.05)
                continue
            if not controlled:
                time.sleep(0.1)
                continue
            try:
                tracker.observe(point, sampled_at)
                if sampled_at >= next_food_scan:
                    tracker.see_food(self.inspiration_food(image))
                    next_food_scan = sampled_at + 1
                    # 小物扫描较慢；扫描结束后重新截图定位，不用旧角色位置转向。
                    image = self.image()
                    sampled_at = time.monotonic()
                    point = self.inspiration_head(image)
                    if point is None:
                        continue
                    tracker.observe(point, sampled_at)
                if tracker.pending:
                    if pending_since is not None and sampled_at - pending_since > 2:
                        raise TrackingError("摇杆转向未得到角色移动确认")
                    time.sleep(0.05)
                    continue
                pending_since = None
                if tracker.heading is not None and tracker.near_center():
                    direction = choose_direction(tracker.body, tracker.heading, tracker.food)
                    if direction is None:
                        raise TrackingError("没有可预测的安全方向")
                    if direction != tracker.heading:
                        # 识别耗时已超过一格移动时间时，不执行过期的转向计划。
                        age = time.monotonic() - sampled_at
                        if tracker.period is None or age < tracker.period * 0.6:
                            self.inspiration_turn(direction)
                            tracker.request(direction)
                            pending_since = time.monotonic()
            except TrackingError as error:
                logger.warning(f"灵感拾趣：{error}，停止转向并等待本局结算")
                controlled = False
            time.sleep(0.05)
        raise DreamError("灵感拾趣未在限时内出现结算页面（不重复开局）")

    def seven_inspiration(self):
        def space_after_loading(image):
            return not self.hit("织梦空间专属加载中", image) and self.space_ready(image)

        self.click(self.wait(
            lambda image: self.box("七日分享返回按钮", image) if self.seven_ready(image) else None,
            "退出七日妙想的返回箭头",
        ))
        self.wait(space_after_loading, "退出七日妙想加载结束后回到织梦空间")
        self.button("灵感拾趣入口")
        self.inspiration_play()
        self.click(self.wait(
            lambda image: self.box("灵感拾趣返回按钮", image) if self.inspiration_result(image) else None,
            "灵感拾趣结算返回箭头",
        ))
        self.wait(space_after_loading, "灵感拾趣结算返回织梦空间")
        self.button("七日妙想入口")
        self.wait(self.seven_ready, "灵感拾趣后回到七日妙想")

    def seven(self):
        self.enter_seven()
        completed = set()  # 只记录本次已执行并返回的任务，不跨任务或账号保存。
        while True:
            candidate = self.seven_current(completed)
            if candidate is None:
                self.seven_claim_rewards()
                logger.info("七日妙想：当前画面没有待做的制作、分享、人台换衣、灵感拾趣或灵感讨论任务")
                return
            category, column, box = candidate
            if category == "全部完成":
                self.seven_claim_rewards()
                logger.info("七日妙想：第一张任务卡出现完成对号，全部任务已完成")
                return
            if category == "换衣":
                self.seven_change_mannequin()
            elif category == "拾趣":
                self.seven_inspiration()
            else:
                self.seven_open_task(category, column, box)
                if category == "分享":
                    self.seven_share()
                elif category == "讨论":
                    self.seven_discussion()
                else:
                    self.seven_make(category)
            completed.add(category)
            outcome = {"分享": "已分享1次定制成衣", "换衣": "已进行1次人台换衣",
                       "拾趣": "已完成1局灵感拾趣", "讨论": "已进行1次灵感讨论"}.get(category, f"已制作1件{category}")
            logger.info(f"七日妙想：{outcome}")

    def cleanup(self, name):
        CleanupRunner(self, name).run()


class CleanupRunner:
    """保持管理状态，按完整名称点击可清理作品；不识别选中状态、不滑动。"""

    def __init__(self, runner, name):
        if not isinstance(name, str) or not name.strip() or "\n" in name or "\r" in name:
            raise DreamError("清理成衣：请输入完整作品名称，多个名称用逗号分隔")
        names = [part.strip() for part in re.split(r"[,，]", name)]
        if not all(names):
            raise DreamError("清理成衣：逗号之间的作品名称不能为空")
        self.r = runner
        self.name = name.strip()
        self.names = tuple(dict.fromkeys(names))
        self.deadline = time.monotonic() + 600
        self.location = ""
        self.deleted = 0

    def phase_deadline(self):
        if self.r.context.tasker.stopping:
            raise DreamError("任务已取消")
        if time.monotonic() >= self.deadline:
            raise DreamError("清理成衣：遍历超过总时限")
        return min(self.deadline, time.monotonic() + 30)

    def button(self, node):
        self.r.seven_button_center(node, deadline=self.phase_deadline())

    def wait(self, predicate, description):
        return self.r.wait(predicate, description, deadline=self.phase_deadline())

    def items(self, image, node, roi=None):
        found = []
        for item in matches(self.r.reco(node, image, roi)):
            box = valid_box(item.box)
            text = getattr(item, "text", "").strip()
            if box and text:
                found.append((text, box))
        return sorted(found, key=lambda item: (item[1][1], item[1][0]))

    @staticmethod
    def signature(items):
        # 不比较背景粒子、选中后的服装预览或涟漪，只比较列表文字与位置。
        return tuple((text, round(box[0] / 8), round(box[1] / 8)) for text, box in items)

    def stable(self, reader, description):
        previous = None

        def settled(image):
            nonlocal previous
            items = reader(image)
            if items is None:
                previous = None
                return None
            signature = self.signature(items)
            if signature != previous:
                previous = signature
                return None
            return image, items

        return self.wait(settled, description)

    def center(self, box):
        self.phase_deadline()
        x, y, w, h = box
        self.r.click((x + w // 2, y + h // 2, 1, 1))

    def sidebar(self, image, *, categories):
        node = "清理成衣成衣标签" if categories else "七日制衣切换部位"
        if not self.r.hit(node, image):
            return None
        excluded = {"普通", "稀有", "非凡", "闪耀", "传说", "成衣", "草稿", "切换部位"}
        rows = []
        for text, box in self.items(image, "清理成衣侧栏文字"):
            if normalized(text) in excluded or box[1] < 85 or box[1] + box[3] > 620:
                continue
            # 不滚动时名称位于缩略图下方的固定行；图内稀有度即使误读也不参与选择。
            if not categories and not any(
                top <= box[1] and box[1] + box[3] <= top + 40
                for top in (195, 348, 501)
            ):
                continue
            row = next((row for row in rows if abs(row[0][1][1] - box[1]) <= 10), None)
            if row is None:
                rows.append([(text, box)])
            else:
                row.append((text, box))
        found = []
        if categories:
            for category in ("连衣裙", "外套", "上衣", "下装"):
                node = "清理成衣外套" if category == "外套" else "七日制衣" + category
                box = self.r.box(node, image)
                if box and box[1] >= 85 and box[1] + box[3] <= 620:
                    found.append((category, box))
        for row in rows:
            row.sort(key=lambda item: item[1][0])
            text = normalized("".join(item[0] for item in row))
            left = min(box[0] for _, box in row)
            top = min(box[1] for _, box in row)
            right = max(box[0] + box[2] for _, box in row)
            bottom = max(box[1] + box[3] for _, box in row)
            box = (left, top, right - left, bottom - top)
            if not any(same_card(box, old) for _, old in found) and len(text) > 1:
                found.append((text, box))
        return sorted(found, key=lambda item: item[1][1])

    def walk_sidebar(self, *, categories):
        reader = lambda image: self.sidebar(image, categories=categories)
        seen = set()
        while True:
            _, items = self.stable(reader, "清理成衣服装列表稳定")
            candidate = next(((text, box) for text, box in items if text not in seen), None)
            if candidate:
                text, box = candidate
                seen.add(text)
                yield text, box
                continue  # 处理后重新读取，不能按原来的第二行坐标继续点击。
            return  # 只遍历当前可见内容，不上划、下划或回到顶部。

    def count(self, image):
        if not self.r.hit("七日分享成衣作品", image):
            return None
        values = []
        for text, _ in self.items(image, "七日分享作品数量"):
            match = re.fullmatch(r"(\d+)/(\d+)", normalized(text))
            if match and 0 <= int(match[1]) <= int(match[2]) and int(match[2]) > 0:
                values.append(int(match[1]))
        return values[0] if len(values) == 1 else None

    def work_items(self, image):
        if self.count(image) is None or not self.r.hit("清理成衣取消管理", image):
            return None
        return [(text, box) for text, box in self.items(image, "清理成衣作品文字")
                if text != "审核中" and box[1] >= 270 and box[1] + box[3] <= 560]

    def review(self, image, name_box):
        x, y, w, h = name_box
        # 实机名称中心 (769,287)，审核字中心 (769.5,254.5)，相差约 33 像素。
        roi = (x + w // 2 - 32, y + h // 2 - 46, 64, 28)
        height, width = image.shape[:2]
        if roi[0] < 700 or roi[1] < 165 or roi[0] + roi[2] > min(width, 1095) or roi[1] + roi[3] > min(height, 560):
            return "状态不明"
        texts = self.items(image, "清理成衣作品文字", roi)
        if any(any(char in normalized(text) for char in "审核") for text, _ in texts):
            return "审核中"
        return "可清理"

    def skip(self, name, reason):
        logger.warning(f"清理成衣：{self.location}，跳过作品 {name!r}：{reason}")

    def manage(self):
        if not self.r.hit("清理成衣取消管理", self.r.image()):
            self.button("清理成衣管理")
            self.wait(lambda image: self.r.hit("清理成衣取消管理", image), "清理成衣管理模式")
        return True

    def select_batch(self, image, items):
        clicked = []
        for text, name_box in items:
            if text not in self.names:  # 任一完整名称字面相等，不能用包含或正则。
                continue
            if any(same_card(name_box, box) for _, box in clicked):
                continue
            image = self.r.image()
            x, y, w, h = name_box
            column = round((x + w / 2 - 770) / 125)
            if not 0 <= column < 3 or abs(x + w / 2 - (770 + 125 * column)) > 30:
                self.skip(text, "作品位置不明")
                continue
            fresh = [(name, box) for name, box in self.items(image, "清理成衣作品文字")
                     if same_card(box, name_box)]
            if len(fresh) != 1 or fresh[0][0] != text:
                self.skip(text, "点击前完整名称无法确认")
                continue
            status = self.review(image, fresh[0][1])
            if status != "可清理":
                self.skip(text, status)
                continue
            self.center(fresh[0][1])
            clicked.append((text, fresh[0][1]))
        return clicked

    def delete_batch(self, clicked, before_count):
        image = self.r.image()
        if (not clicked or not isinstance(before_count, int) or before_count <= 0
                or len(clicked) > before_count or self.count(image) != before_count
                or any(name not in self.names for name, _ in clicked)
                or not self.r.hit("清理成衣取消管理", image)):
            raise DreamError("清理成衣：点击记录、作品数量或管理页面不一致，未执行删除")
        if not self.r.hit("清理成衣删除可用", image):
            self.skip(self.name, "删除按钮不可用")
            return False
        self.button("清理成衣删除可用")
        self.wait(lambda image: self.r.hit("清理成衣删除弹窗", image), "清理成衣删除确认及无法恢复提示")
        self.button("清理成衣确认删除")
        expected = before_count - len(clicked)

        def changed(image):
            if self.r.hit("清理成衣删除弹窗", image):
                return False
            count = self.count(image)
            return count == expected if count is not None else (
                expected == 0 and (self.r.hit("七日制衣切换部位", image)
                                   or self.r.hit("清理成衣成衣标签", image))
            )

        self.wait(changed, "清理成衣删除后作品数量减少或自动返回")
        self.deleted += len(clicked)
        deleted_names = tuple(dict.fromkeys(name for name, _ in clicked))
        logger.info(f"清理成衣：{self.location}，已删除 {len(clicked)} 件成衣作品，名称 {deleted_names!r}")
        return True

    def garment(self):
        self.wait(lambda image: self.r.hit("七日分享成衣作品", image), "清理成衣作品列表")
        initial = self.count(self.r.image())
        if not initial:
            if initial is None:
                self.skip(self.name, "作品数量无法确认")
            return
        if not self.manage():
            return
        reader = self.work_items
        while True:
            image, items = self.stable(reader, "清理成衣作品名称就绪")
            before = self.count(image)
            if not before:
                if before is None:
                    self.skip(self.name, "作品数量无法确认")
                return
            clicked = self.select_batch(image, items)
            if clicked:
                if not self.delete_batch(clicked, before):
                    return
                if before == len(clicked):  # 已由数量变化/自动返回验证清空。
                    return
                # 删除导致重排，保持管理状态并重新读取当前可见作品。
                if not self.manage():
                    return
                continue
            if not items and before:
                self.skip(self.name, "作品名称无法识别")
            return

    def reopen_category(self, category):
        _, categories = self.stable(lambda image: self.sidebar(image, categories=True),
                                    "清理成衣重新识别服装类别")
        matches = [box for name, box in categories if name == category]
        if len(matches) != 1:
            raise DreamError(f"清理成衣：无法重新确认服装类别 {category!r}")
        self.center(matches[0])
        self.wait(lambda image: self.r.hit("七日制衣切换部位", image), "清理成衣重新打开原类别")

    def clean_category(self, category):
        reader = lambda image: self.sidebar(image, categories=False)
        _, initial = self.stable(reader, "清理成衣记录类别服装列表")
        # 删除后列表可能变化，保留初始名称和总数；点击位置每次重新识别。
        garments = tuple(dict.fromkeys(name for name, _ in initial))
        logger.info(f"清理成衣：{category}，已记录 {len(garments)} 件可见服装，按顺序检查 {garments!r}")
        for index, garment in enumerate(garments, 1):
            if self.r.hit("清理成衣成衣标签", self.r.image()):
                self.reopen_category(category)
            _, current = self.stable(reader, "清理成衣定位下一件服装")
            boxes = [box for name, box in current if name == garment]
            self.location = f"{category}/{garment}"
            if len(boxes) != 1:
                logger.warning(f"清理成衣：{self.location}，跳过服装：原名单中的名称无法确认，进度 {index}/{len(garments)}")
                continue
            self.center(boxes[0])
            self.garment()
            logger.info(f"清理成衣：{self.location}，已检查 {index}/{len(garments)} 件服装")

        # 最后一件删除后也可能已自动返回类别页，不再点击不存在的“切换部位”。
        if not self.r.hit("清理成衣成衣标签", self.r.image()):
            self.button("七日制衣切换部位")
            self.wait(lambda image: self.r.hit("清理成衣成衣标签", image), "清理成衣返回服装类别")

    def run(self):
        self.wait(self.r.space_ready, "清理成衣进入织梦空间")
        self.button("清理成衣至臻定制")
        self.button("清理成衣成衣标签")
        for category, box in self.walk_sidebar(categories=True):
            self.center(box)
            self.wait(lambda image: self.r.hit("七日制衣切换部位", image), "清理成衣服装类别列表")
            self.clean_category(category)
        self.button("七日分享返回按钮")
        self.wait(self.r.space_ready, "清理成衣返回织梦空间")
        logger.info(f"清理成衣：完成，共删除 {self.deleted} 件作品")


@AgentServer.custom_action("dreamweaver_input_text")
class DreamweaverInputTextAction(CustomAction):
    def run(self, context, argv):
        try:
            if context.tasker.stopping:
                return False
            params = json.loads(argv.custom_action_param)
            text = params["input_text"]
            if not isinstance(text, str) or not text:
                raise DreamError("作品名称不能为空")
            info = context.tasker.controller.info
            config = info.get("config", {})
            if isinstance(config, str):
                config = json.loads(config)
            mumu = config.get("extras", {}).get("mumu", {})
            if info.get("type") != "adb" or not mumu.get("enable"):
                result = context.run_action_direct("InputText", JInputText(input_text=text))
                return bool(result and result.success)

            # MaaTouch 在 Android 15 上无法调用剪贴板；仅此文本动作走 MuMu 官方命令。
            index = mumu.get("index")
            root = mumu.get("path")
            if type(index) is not int or index < 0 or not isinstance(root, str) or not root:
                raise DreamError("MuMu 配置缺少有效路径或实例编号")
            manager = next((path for path in (
                Path(root) / "nx_main" / "MuMuManager.exe",
                Path(root) / "MuMuManager.exe",
            ) if path.is_file()), None)
            if manager is None:
                raise DreamError("MuMu 配置路径下未找到 MuMuManager.exe")
            if context.tasker.stopping:
                return False
            result = subprocess.run(
                [str(manager), "control", "-v", str(index), "tool", "cmd",
                 "--cmd", "input_text", "--text", text],
                capture_output=True, encoding="utf-8-sig", errors="replace", timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode != 0:
                raise DreamError(f"MuMu 文本输入进程失败：{result.returncode}")
            response = json.loads(result.stdout)
            if not isinstance(response, dict) or response.get("errcode") != 0:
                raise DreamError(f"MuMu 文本输入失败：{response}")
            logger.debug(f"七日制衣：MuMu 实例 {index} 已接收文本输入，等待页面核验")
            return not context.tasker.stopping
        except Exception as error:
            logger.error(f"七日制衣文本输入失败：{error}")
            return False


@AgentServer.custom_action("dreamweaver_space_run")
class DreamweaverSpaceAction(CustomAction):
    def run(self, context, argv):
        def enabled(node):
            data = context.get_node_data(node)
            return bool(data and data.get("enabled"))

        runner = DreamRunner(context)
        try:
            if enabled("织梦选择七日妙想"):
                runner.seven()
                has_management = enabled("织梦选择经营") and any(
                    enabled(node) for node in ("织梦选择每日能量晶矿", "织梦选择订单驿站", "织梦选择售卖橱窗")
                )
                if enabled("织梦选择今日签到") or has_management or enabled("织梦选择清理成衣"):
                    runner.return_from_seven()
            if enabled("织梦选择清理成衣"):
                data = context.get_node_data("清理成衣设置") or {}
                runner.cleanup(data.get("attach", {}).get("name"))
            if enabled("织梦选择今日签到"):
                runner.sign_in()
            if enabled("织梦选择经营") and enabled("织梦选择每日能量晶矿"):
                runner.mine()
            if enabled("织梦选择经营") and enabled("织梦选择订单驿站"):
                runner.orders(enabled("织梦选择能量石加速"))
            if enabled("织梦选择经营") and enabled("织梦选择售卖橱窗"):
                params = json.loads(getattr(argv, "custom_action_param", "") or "{}")
                current = enabled("织梦售卖仅当期")
                tomorrow = enabled("织梦售卖攒明日")
                if current and tomorrow:
                    raise DreamError("售卖方案选择冲突")
                mode = "current" if current else ("tomorrow" if tomorrow else "stock")
                runner.shop(
                    mode, params.get("sale_rounds", 0),
                    enabled("织梦售卖选择能量石"),
                    after_orders=enabled("织梦选择订单驿站"),
                )
            return True
        except Exception as error:
            logger.error(f"织梦空间：{error}")
            return False
