import json
import time

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger


@AgentServer.custom_action("card_matching")
class CardMatchingAction(CustomAction):
    """Solve a concentration-style card board by remembering revealed images."""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}

        rows = self._as_int(params.get("rows"), 5)
        columns = self._as_int(params.get("columns"), 4)
        origin = params.get("origin", [72, 297])
        card_size = params.get("card_size", [123, 121])
        step = params.get("step", [152, 151])
        crop_margin = params.get("crop_margin", [12, 12, 12, 12])
        match_threshold = self._as_float(params.get("match_threshold"), 0.12)
        fallback_match_threshold = self._as_float(
            params.get("fallback_match_threshold"), 0.16
        )
        flip_change_threshold = self._as_float(
            params.get("flip_change_threshold"), 0.04
        )
        flip_retry_count = self._as_int(params.get("flip_retry_count"), 1)
        back_match_threshold = self._as_float(
            params.get("back_match_threshold"), 0.03
        )
        completion_retry_count = self._as_int(
            params.get("completion_retry_count"), 2
        )
        completion_check_delay = self._as_float(
            params.get("completion_check_delay"), 0.80
        )
        # The game ignores a second tap while the first card is still turning.
        # Keep these defaults above the observed input-lock/animation duration.
        flip_delay = self._as_float(params.get("flip_delay"), 0.60)
        pair_delay = self._as_float(params.get("pair_delay"), 0.65)
        mismatch_delay = self._as_float(params.get("mismatch_delay"), 1.20)
        max_turns = self._as_int(params.get("max_turns"), rows * columns * 3)

        if not self._valid_pair(origin) or not self._valid_pair(card_size) or not self._valid_pair(step):
            logger.error("card_matching: origin/card_size/step 必须是包含两个数字的数组")
            return False
        if not self._valid_margin(crop_margin):
            logger.error("card_matching: crop_margin 必须是 [left, top, right, bottom]")
            return False
        if rows <= 0 or columns <= 0 or rows * columns % 2:
            logger.error("card_matching: 牌数必须是正偶数")
            return False
        if (
            match_threshold <= 0
            or fallback_match_threshold < match_threshold
            or flip_change_threshold <= 0
            or flip_retry_count < 0
            or back_match_threshold <= 0
            or completion_retry_count < 0
            or completion_check_delay < 0
            or max_turns <= 0
        ):
            logger.error(
                "card_matching: 阈值必须大于 0，且 fallback_match_threshold "
                "不能小于 match_threshold"
            )
            return False

        cards = self._build_cards(rows, columns, origin, card_size, step)
        controller = context.tasker.controller
        image = self._screencap(controller)
        if not self._board_fits(image, cards):
            logger.error(
                f"card_matching: 棋盘超出截图范围，image={image.shape[:2]}, "
                f"last_card={cards[-1]}"
            )
            return False

        unresolved = set(range(len(cards)))
        known = {}
        back_signatures = [
            self._signature(image, card, crop_margin) for card in cards
        ]
        completion_retry = 0
        turns = 0

        logger.info(
            f"card_matching: 开始翻牌，rows={rows}, columns={columns}, "
            f"threshold={match_threshold:.4f}"
        )

        while True:
            if not unresolved:
                time.sleep(completion_check_delay)
                face_down = self._find_face_down_cards(
                    controller,
                    cards,
                    crop_margin,
                    back_signatures,
                    back_match_threshold,
                )
                if not face_down:
                    break
                if completion_retry >= completion_retry_count:
                    logger.error(
                        "card_matching: 完成后补扫次数已用尽，仍检测到未翻开的牌："
                        + ", ".join(self._label(index, columns) for index in face_down)
                    )
                    return False

                completion_retry += 1
                unresolved.update(face_down)
                known.clear()
                logger.warning(
                    f"card_matching: 完成后检测到 {len(face_down)} 张牌仍为背面，"
                    f"开始第 {completion_retry}/{completion_retry_count} 次补扫："
                    + ", ".join(self._label(index, columns) for index in face_down)
                )

            if turns >= max_turns:
                break

            remembered_pair = self._find_known_pair(known, unresolved, match_threshold)
            if remembered_pair is None and not unresolved.difference(known):
                remembered_pair = self._find_known_pair(
                    known, unresolved, fallback_match_threshold
                )
                if remembered_pair is not None:
                    logger.info(
                        "card_matching: 剩余牌均已记录，使用兜底阈值 "
                        f"{fallback_match_threshold:.4f} 完成最接近的配对"
                    )
            if remembered_pair is not None:
                first, second, distance = remembered_pair
                logger.info(
                    f"card_matching: 使用记忆配对 {self._label(first, columns)} + "
                    f"{self._label(second, columns)}, distance={distance:.4f}"
                )
                self._reveal(
                    controller,
                    cards[first],
                    crop_margin,
                    flip_delay,
                    flip_change_threshold,
                    flip_retry_count,
                )
                self._reveal(
                    controller,
                    cards[second],
                    crop_margin,
                    flip_delay,
                    flip_change_threshold,
                    flip_retry_count,
                )
                time.sleep(pair_delay)
                self._resolve(first, second, unresolved, known)
                turns += 1
                continue

            unseen = sorted(unresolved.difference(known))
            if not unseen:
                logger.error(
                    "card_matching: 所有剩余牌均已记录但未找到配对；"
                    "请适当调大 fallback_match_threshold"
                )
                return False

            first = unseen[0]
            first_signature = self._reveal(
                controller,
                cards[first],
                crop_margin,
                flip_delay,
                flip_change_threshold,
                flip_retry_count,
            )
            turns += 1
            match = self._best_match(first_signature, known, unresolved, match_threshold)
            if match is not None:
                second, distance = match
                logger.info(
                    f"card_matching: 新牌 {self._label(first, columns)} 命中记忆牌 "
                    f"{self._label(second, columns)}, distance={distance:.4f}"
                )
                self._reveal(
                    controller,
                    cards[second],
                    crop_margin,
                    flip_delay,
                    flip_change_threshold,
                    flip_retry_count,
                )
                time.sleep(pair_delay)
                self._resolve(first, second, unresolved, known)
                continue

            known[first] = first_signature
            unseen = sorted(unresolved.difference(known))
            if not unseen:
                logger.error(
                    "card_matching: 最后一张牌没有找到配对；请检查网格参数或调大 "
                    "match_threshold"
                )
                return False

            second = unseen[0]
            second_signature = self._reveal(
                controller,
                cards[second],
                crop_margin,
                flip_delay,
                flip_change_threshold,
                flip_retry_count,
            )
            second_match = self._best_match(
                second_signature, known, unresolved, match_threshold
            )

            if second_match is not None and second_match[0] == first:
                logger.info(
                    f"card_matching: 当轮配对 {self._label(first, columns)} + "
                    f"{self._label(second, columns)}, distance={second_match[1]:.4f}"
                )
                time.sleep(pair_delay)
                self._resolve(first, second, unresolved, known)
                continue

            known[second] = second_signature
            if second_match is None:
                logger.info(
                    f"card_matching: 记录未配对牌 {self._label(first, columns)}、"
                    f"{self._label(second, columns)}"
                )
            else:
                logger.info(
                    f"card_matching: {self._label(second, columns)} 与记忆牌 "
                    f"{self._label(second_match[0], columns)} 相同，等待本轮翻回后再配对"
                )
            time.sleep(mismatch_delay)

        if unresolved:
            logger.error(
                f"card_matching: 超过最大轮数 {max_turns}，仍有 {len(unresolved)} 张牌"
            )
            return False

        logger.info(f"card_matching: 完成全部 {len(cards) // 2} 组配对")
        return True

    def _reveal(
        self,
        controller,
        card,
        crop_margin,
        delay,
        change_threshold,
        retry_count,
    ):
        before = self._signature(self._screencap(controller), card, crop_margin)
        after = before

        for attempt in range(retry_count + 1):
            self._click_card(controller, card)
            time.sleep(delay)
            after = self._signature(self._screencap(controller), card, crop_margin)
            distance = self._image_distance(before, after)
            if distance >= change_threshold:
                if attempt > 0:
                    logger.info(
                        f"card_matching: 补点成功，card={card}, "
                        f"change={distance:.4f}"
                    )
                return after

            if attempt < retry_count:
                logger.warning(
                    f"card_matching: 点击后牌面未变化，准备补点，card={card}, "
                    f"attempt={attempt + 1}/{retry_count + 1}, change={distance:.4f}"
                )

        logger.warning(
            f"card_matching: 补点后牌面仍未变化，继续使用当前截图，card={card}"
        )
        return after

    def _signature(self, image, card, margin):
        x, y, w, h = card
        left, top, right, bottom = margin
        crop = image[y + top:y + h - bottom, x + left:x + w - right]
        if crop.size == 0:
            raise ValueError("card_matching: crop_margin 使牌面截图为空")
        return crop.copy()

    def _best_match(self, signature, known, unresolved, threshold):
        best_index = None
        best_distance = float("inf")
        for index, remembered in known.items():
            if index not in unresolved:
                continue
            distance = self._image_distance(signature, remembered)
            if distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is not None and best_distance <= threshold:
            return best_index, best_distance
        return None

    def _find_known_pair(self, known, unresolved, threshold):
        indices = sorted(index for index in known if index in unresolved)
        best = None
        for offset, first in enumerate(indices):
            for second in indices[offset + 1:]:
                distance = self._image_distance(known[first], known[second])
                if distance <= threshold and (best is None or distance < best[2]):
                    best = first, second, distance
        return best

    def _image_distance(self, first, second):
        if first.shape != second.shape or first.size == 0:
            return float("inf")
        difference = np.abs(first.astype(np.int16) - second.astype(np.int16))
        pixel_distance = float(np.mean(difference) / 255.0)
        histogram_distance = self._histogram_distance(first, second)
        # Mean pixel distance alone confuses pale cards with very different
        # drawings. Requiring their quantized color distributions to agree
        # keeps those cards separate while tolerating small render variations.
        return max(pixel_distance, histogram_distance)

    def _histogram_distance(self, first, second):
        bins_per_channel = 8

        def normalized_histogram(image):
            pixels = image.reshape(-1, image.shape[-1])[:, :3].astype(np.uint16)
            quantized = pixels * bins_per_channel // 256
            indices = (
                quantized[:, 0] * bins_per_channel * bins_per_channel
                + quantized[:, 1] * bins_per_channel
                + quantized[:, 2]
            )
            histogram = np.bincount(
                indices.astype(np.int32),
                minlength=bins_per_channel ** 3,
            ).astype(np.float32)
            return histogram / max(len(pixels), 1)

        first_histogram = normalized_histogram(first)
        second_histogram = normalized_histogram(second)
        return float(np.sum(np.abs(first_histogram - second_histogram)) / 2.0)

    def _find_face_down_cards(
        self,
        controller,
        cards,
        crop_margin,
        back_signatures,
        threshold,
    ):
        image = self._screencap(controller)
        face_down = []
        for index, card in enumerate(cards):
            current = self._signature(image, card, crop_margin)
            distance = self._image_distance(current, back_signatures[index])
            if distance <= threshold:
                face_down.append(index)
        return face_down

    def _build_cards(self, rows, columns, origin, card_size, step):
        cards = []
        for row in range(rows):
            for column in range(columns):
                cards.append(
                    [
                        int(origin[0] + column * step[0]),
                        int(origin[1] + row * step[1]),
                        int(card_size[0]),
                        int(card_size[1]),
                    ]
                )
        return cards

    def _resolve(self, first, second, unresolved, known):
        unresolved.discard(first)
        unresolved.discard(second)
        known.pop(first, None)
        known.pop(second, None)

    def _click_card(self, controller, card):
        x, y, w, h = card
        controller.post_click(x + w // 2, y + h // 2).wait()

    def _screencap(self, controller):
        controller.post_screencap().wait()
        return controller.cached_image.copy()

    def _board_fits(self, image, cards):
        if image is None or image.ndim < 2:
            return False
        height, width = image.shape[:2]
        return all(
            x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= width and y + h <= height
            for x, y, w, h in cards
        )

    def _label(self, index, columns):
        return f"R{index // columns + 1}C{index % columns + 1}"

    def _valid_pair(self, value):
        return isinstance(value, (list, tuple)) and len(value) == 2

    def _valid_margin(self, value):
        return (
            isinstance(value, (list, tuple))
            and len(value) == 4
            and all(self._as_int(item, -1) >= 0 for item in value)
        )

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
