"""Read whole stage labels from white glyphs, without filling in missing digits.

Templates are binary strokes extracted from game screenshots, not whole buttons.
Grouping happens before comparison with the requested stage, so 1-10 cannot match
1-1. Unknown or ambiguous glyphs invalidate the entire label.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "data/main_storyline_digits.json"


@dataclass
class Glyph:
    x: int
    y: int
    mask: np.ndarray

    @property
    def w(self):
        return self.mask.shape[1]

    @property
    def h(self):
        return self.mask.shape[0]


@dataclass
class StageCandidate:
    box: list[int]
    left: tuple[str | None, ...]
    right: tuple[str | None, ...]
    scores: tuple[tuple[tuple[str, float], ...], ...]
    rejection: str = ""

    @property
    def text(self):
        return "".join(c or "?" for c in self.left) + "-" + "".join(c or "?" for c in self.right)

    @property
    def complete(self):
        return not self.rejection and None not in self.left + self.right

    def check_target(self, target):
        a, b = target.split("-")
        if self.rejection:
            return self.rejection
        if len(self.left) != len(a) or len(self.right) != len(b):
            return f"数字数量不符：需要 {len(a)}+{len(b)} 位，实际 {len(self.left)}+{len(self.right)} 位"
        for i, (observed, expected) in enumerate(zip(self.left + self.right, a + b), 1):
            if observed is not None and observed != expected:
                return f"已确认的模板字符冲突：第 {i} 位是 {observed}，目标要求 {expected}"
        return ""

    def score_summary(self):
        return "；".join(
            f"第 {i} 位=" + "/".join(f"{char}:{score:.3f}" for char, score in ranked)
            for i, ranked in enumerate(self.scores, 1)
        )


def white_strokes(image):
    pixels = image.astype(np.int16)
    return (pixels.min(axis=2) >= 205) & (
        pixels.max(axis=2) - pixels.min(axis=2) <= 65
    )


def components(mask):
    """Eight-connected components using row runs rather than per-pixel objects."""
    parents, runs = [], []

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    previous = []
    for y, row in enumerate(mask):
        edges = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        current, cursor = [], 0
        for start, end in zip(starts, ends):
            start, end = int(start), int(end)
            i = len(parents)
            parents.append(i)
            runs.append((y, start, end))
            current.append((start, end, i))
            while cursor < len(previous) and previous[cursor][1] < start:
                cursor += 1
            j = cursor
            while j < len(previous) and previous[j][0] <= end:
                parents[root(previous[j][2])] = root(i)
                j += 1
        previous = current

    groups = {}
    for i, run in enumerate(runs):
        groups.setdefault(root(i), []).append(run)
    result = []
    for group in groups.values():
        x = min(run[1] for run in group)
        right = max(run[2] for run in group)
        y, bottom = group[0][0], group[-1][0] + 1
        if sum(end - start for _, start, end in group) < 3:
            continue
        glyph = np.zeros((bottom - y, right - x), dtype=bool)
        for py, start, end in group:
            glyph[py - y, start - x:end - x] = True
        result.append(Glyph(x, y, glyph))
    return result


def resize_mask(mask, height=32, width=24):
    ys = np.minimum(((np.arange(height) + 0.5) * mask.shape[0] / height).astype(int), mask.shape[0] - 1)
    xs = np.minimum(((np.arange(width) + 0.5) * mask.shape[1] / width).astype(int), mask.shape[1] - 1)
    return mask[np.ix_(ys, xs)]


@lru_cache(maxsize=1)
def load_templates():
    data = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return tuple(
        (sample["char"], np.array([[pixel == "1" for pixel in row] for row in sample["rows"]], dtype=bool))
        for sample in data["samples"]
    )


def classify_digit_detail(mask):
    h, w = mask.shape
    normalized = resize_mask(mask)
    scores = {}
    for char, template in load_templates():
        if char == "-":
            continue
        th, tw = template.shape
        # Retain aspect ratio evidence: stretching a narrow 1 must not produce 7.
        if abs(w / h - tw / th) > 0.13:
            continue
        other = resize_mask(template)
        score = np.count_nonzero(normalized & other) / np.count_nonzero(normalized | other)
        scores[char] = max(float(score), scores.get(char, 0))
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    if not ranked or ranked[0][1] < 0.72:
        return None, tuple(ranked[:2])
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.10:
        return None, tuple(ranked[:2])
    return ranked[0][0], tuple(ranked[:2])


def classify_digit(mask):
    return classify_digit_detail(mask)[0]


def read_stage_candidates(image):
    """Keep whole rows and uncertain slots for OCR verification and diagnostics."""
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        return []
    glyphs = components(white_strokes(image))
    # Keep unknown shapes in a numeral row too: omitting them could turn a
    # partially recognised two-digit suffix into a valid one-digit suffix.
    digits = [g for g in glyphs if 12 <= g.h <= 48]
    dashes = [g for g in glyphs if 1 <= g.h <= 7 and 4 <= g.w <= 24 and g.w >= 2 * g.h]
    candidates = []
    for dash in dashes:
        center_y = dash.y + dash.h / 2
        nearby = [g for g in digits if g.y + 0.35 * g.h <= center_y <= g.y + 0.8 * g.h]
        left = sorted((g for g in nearby if g.x + g.w <= dash.x), key=lambda g: g.x, reverse=True)
        right = sorted((g for g in nearby if g.x >= dash.x + dash.w), key=lambda g: g.x)
        if not left or not right:
            continue
        height = max(left[0].h, right[0].h)
        top_y = min(left[0].y, right[0].y)
        if not (0.22 * height <= dash.w <= 0.55 * height and dash.h <= 0.22 * height):
            continue
        if np.count_nonzero(dash.mask) / dash.mask.size < 0.75:
            continue
        dash_shape = resize_mask(dash.mask, 4, 16)
        if not any(
            np.count_nonzero(dash_shape & resize_mask(template, 4, 16))
            / np.count_nonzero(dash_shape | resize_mask(template, 4, 16)) >= 0.72
            for char, template in load_templates() if char == "-"
        ):
            continue

        def collect(candidates, boundary, direction):
            group = []
            for glyph in candidates:
                # Card borders cross the text line, but have a different top
                # and baseline; they are not characters in the label.
                if abs(glyph.y - top_y) > 0.2 * height:
                    continue
                gap = boundary - glyph.x - glyph.w if direction < 0 else glyph.x - boundary
                if gap > 0.6 * height:
                    break
                if abs(glyph.h - height) > 0.25 * height:
                    break
                group.append(glyph)
                boundary = glyph.x if direction < 0 else glyph.x + glyph.w
            return group

        left = list(reversed(collect(left, dash.x, -1)))
        right = collect(right, dash.x + dash.w, 1)
        if not left or not right:
            continue
        sequence = left + right
        details = [classify_digit_detail(g.mask) for g in sequence]
        chars = tuple(char for char, _ in details)
        first, last = sequence[0], sequence[-1]
        top = min(g.y for g in sequence)
        bottom = max(g.y + g.h for g in sequence)
        rejection = ""
        if len(left) > 2 or len(right) > 2:
            rejection = "编号任一侧超过两位数字"
        # A clipped label has no reliable outer boundary.
        if first.x == 0 or top == 0 or last.x + last.w >= image.shape[1] or bottom >= image.shape[0]:
            rejection = "编号位于截图边界，可能被截断"
        if chars[0] == "0" or chars[len(left)] == "0":
            rejection = "编号以零开头，不是有效关卡"
        candidates.append(StageCandidate(
            box=[first.x, top, last.x + last.w - first.x, bottom - top],
            left=chars[:len(left)],
            right=chars[len(left):],
            scores=tuple(scores for _, scores in details),
            rejection=rejection,
        ))
    return candidates


def read_stage_labels(image):
    """Return only fully confirmed labels; never search a target substring."""
    return [(candidate.text, candidate.box) for candidate in read_stage_candidates(image) if candidate.complete]
