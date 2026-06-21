import json
import re
import time

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JRecognitionType, JOCR, JTemplateMatch
from utils import logger


EXCLUDED_DIAMOND_NAMES = {
    "时间之诗",
    "记忆钥匙",
    "金币",
    "印象碎片",
    "体力",
    "心意币",
    "记忆之轨",
    "冰凝之心"
}


DEFAULT_SLOTS = [
    {
        "name_roi": [165, 526, 190, 58],
        "discount_roi": [19, 520, 62, 59],
        "price_roi": [246, 620, 106, 48],
        "currency_roi": [248, 633, 32, 27],
        "button_roi": [176, 620, 176, 48],
    },
    {
        "name_roi": [520, 526, 195, 58],
        "discount_roi": [372, 520, 62, 59],
        "price_roi": [600, 620, 106, 48],
        "currency_roi": [604, 633, 32, 27],
        "button_roi": [530, 620, 176, 48],
    },
    {
        "name_roi": [165, 728, 190, 58],
        "discount_roi": [20, 721, 61, 60],
        "price_roi": [246, 820, 106, 48],
        "currency_roi": [268, 833, 32, 27],
        "button_roi": [176, 820, 176, 48],
    },
    {
        "name_roi": [520, 728, 195, 58],
        "discount_roi": [372, 721, 62, 60],
        "price_roi": [601, 820, 106, 48],
        "currency_roi": [606, 833, 32, 27],
        "button_roi": [531, 820, 176, 48],
    },
    {
        "name_roi": [165, 930, 190, 58],
        "discount_roi": [20, 923, 61, 60],
        "price_roi": [246, 1022, 106, 48],
        "currency_roi": [268, 1035, 32, 27],
        "button_roi": [176, 1022, 176, 48],
    },
    {
        "name_roi": [520, 930, 195, 58],
        "discount_roi": [372, 923, 62, 60],
        "price_roi": [601, 1022, 106, 48],
        "currency_roi": [606, 1035, 32, 27],
        "button_roi": [531, 1022, 176, 48],
    },
]


@AgentServer.custom_action("memory_shop_check")
class MemoryShopCheckAction(CustomAction):
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}

        slots = params.get("slots", DEFAULT_SLOTS)
        refresh_roi = params.get("refresh_roi", [516, 1167, 190, 49])
        refresh_text_roi = params.get("refresh_text_roi", refresh_roi)
        refresh_exhausted_roi = params.get("refresh_exhausted_roi", [231, 270, 241, 104])
        confirm_roi = params.get("confirm_roi", [122, 708, 490, 72])
        confirm_text_roi = params.get("confirm_text_roi", [335, 711, 230, 68])
        reward_close_roi = params.get("reward_close_roi", [318, 1183, 69, 35])
        back_roi = params.get("back_roi", [0, 36, 100, 64])
        shop_page_roi = params.get("shop_page_roi", [0, 448, 720, 790])
        pause_next = params.get("pause_next", ["stop"])
        refresh_next = params.get("refresh_next", [argv.node_name])
        auto_buy_gold = self._as_bool(params.get("auto_buy_gold"), False)
        continue_refresh_when_not_free = self._as_bool(params.get("continue_refresh_when_not_free"), False)
        diamond_discount_threshold = self._as_int(params.get("diamond_discount_threshold"), 4)
        diamond_price_threshold = self._as_int(params.get("diamond_price_threshold"), 0)
        ocr_threshold = self._as_float(params.get("ocr_threshold"), 0.3)
        currency_threshold = self._as_float(params.get("currency_threshold"), 0.8)
        gold_template = params.get("gold_template", "回忆小铺-金币.png")
        diamond_template = params.get("diamond_template", "回忆小铺-钻石.png")
        post_buy_delay = self._as_float(params.get("post_buy_delay"), 1.0)
        post_refresh_delay = self._as_float(params.get("post_refresh_delay"), 2.0)
        purchase_recovery_attempts = self._as_int(params.get("purchase_recovery_attempts"), 8)
        purchase_recovery_delay = self._as_float(params.get("purchase_recovery_delay"), 0.8)
        purchase_close_clicks = self._as_int(params.get("purchase_close_clicks"), 3)
        excluded_names = set(params.get("excluded_names", EXCLUDED_DIAMOND_NAMES))

        if not self._valid_roi(refresh_roi):
            logger.error("memory_shop_check: invalid refresh_roi")
            return False
        if not self._valid_roi(refresh_text_roi):
            logger.error("memory_shop_check: invalid refresh_text_roi")
            return False
        for key, roi in (
            ("refresh_exhausted_roi", refresh_exhausted_roi),
            ("reward_close_roi", reward_close_roi),
            ("back_roi", back_roi),
            ("shop_page_roi", shop_page_roi),
        ):
            if not self._valid_roi(roi):
                logger.error(f"memory_shop_check: invalid {key}")
                return False
        if not isinstance(slots, list) or not slots:
            logger.error("memory_shop_check: invalid slots")
            return False

        controller = context.tasker.controller
        image = self._screencap(controller)
        products = []
        gold_products = []
        diamond_price_pause_products = []
        diamond_pause_products = []

        for index, slot in enumerate(slots):
            product = self._read_product(
                context,
                image,
                slot,
                index,
                ocr_threshold,
                currency_threshold,
                gold_template,
                diamond_template,
            )
            products.append(product)
            logger.info(
                "memory_shop_check: "
                f"slot={index + 1}, name={product['name']!r}, discount={product['discount']}, "
                f"currency={product['currency']}, price={product['price']}, "
                f"price_text={product['price_text']!r}"
            )

            if product["currency"] == "gold":
                gold_products.append(product)
                continue

            if product["currency"] == "diamond" and not self._is_excluded_name(product["name"], excluded_names):
                if (
                    diamond_price_threshold > 0
                    and product["price"] is not None
                    and product["price"] < diamond_price_threshold
                ):
                    diamond_price_pause_products.append(product)
                    continue

                if (
                    product["discount"] is not None
                    and product["discount"] <= diamond_discount_threshold
                ):
                    diamond_pause_products.append(product)

        if gold_products and not auto_buy_gold:
            logger.info("memory_shop_check: gold product found, pausing")
            context.override_next(argv.node_name, pause_next)
            return True

        if auto_buy_gold:
            for product in gold_products:
                self._click_roi(controller, product["button_roi"])
                time.sleep(post_buy_delay)
                self._confirm_if_needed(context, controller, confirm_text_roi, confirm_roi, ocr_threshold)
                time.sleep(post_buy_delay)
                if not self._recover_after_purchase(
                    context,
                    controller,
                    reward_close_roi,
                    shop_page_roi,
                    back_roi,
                    ocr_threshold,
                    purchase_recovery_attempts,
                    purchase_recovery_delay,
                    purchase_close_clicks,
                ):
                    logger.info("memory_shop_check: purchase recovery failed, pausing")
                    context.override_next(argv.node_name, pause_next)
                    return True

        if diamond_price_pause_products:
            logger.info(
                "memory_shop_check: low-price diamond product found, pausing: "
                + ", ".join(
                    f"{item['name']}({item['price']}钻)"
                    for item in diamond_price_pause_products
                )
            )
            context.override_next(argv.node_name, pause_next)
            return True

        if diamond_pause_products:
            logger.info(
                "memory_shop_check: valuable diamond product found, pausing: "
                + ", ".join(
                    f"{item['name']}({item['discount']}折)"
                    for item in diamond_pause_products
                )
            )
            context.override_next(argv.node_name, pause_next)
            return True

        image = self._screencap(controller)
        refresh_free = self._is_refresh_free(context, image, refresh_text_roi, ocr_threshold)
        if not refresh_free and not continue_refresh_when_not_free:
            logger.info("memory_shop_check: refresh is not free, pausing")
            context.override_next(argv.node_name, pause_next)
            return True

        logger.info(
            "memory_shop_check: no product matched pause rules, refreshing, "
            f"refresh_free={refresh_free}"
        )
        self._click_roi(controller, refresh_roi)
        time.sleep(post_refresh_delay)
        image = self._screencap(controller)
        if self._is_refresh_exhausted_popup(
            context,
            image,
            refresh_exhausted_roi,
            ocr_threshold,
        ):
            logger.info("memory_shop_check: refresh card exhausted popup found, pausing")
            context.override_next(argv.node_name, pause_next)
            return True

        context.override_next(argv.node_name, refresh_next)
        return True

    def _read_product(
        self,
        context,
        image,
        slot,
        index,
        ocr_threshold,
        currency_threshold,
        gold_template,
        diamond_template,
    ):
        name_roi = slot.get("name_roi")
        discount_roi = slot.get("discount_roi")
        currency_roi = slot.get("currency_roi")
        button_roi = slot.get("button_roi")
        price_roi = slot.get("price_roi") or self._right_half_roi(button_roi)
        for key, roi in (
            ("name_roi", name_roi),
            ("discount_roi", discount_roi),
            ("button_roi", button_roi),
            ("price_roi", price_roi),
        ):
            if not self._valid_roi(roi):
                raise ValueError(f"memory_shop_check: invalid {key} at slot {index + 1}")

        name = self._read_text(context, image, name_roi, ocr_threshold)
        discount_text = self._read_text(context, image, discount_roi, ocr_threshold)
        price_text = self._read_text(context, image, price_roi, ocr_threshold)
        return {
            "name": self._normalize_name(name),
            "discount": self._parse_discount(discount_text),
            "discount_text": discount_text,
            "price": self._parse_price(price_text),
            "price_text": price_text,
            "currency": self._detect_currency(
                context,
                image,
                button_roi,
                currency_threshold,
                gold_template,
                diamond_template,
            ),
            "button_roi": button_roi,
        }

    def _read_text(self, context, image, roi, threshold):
        result = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(roi=roi, threshold=threshold, order_by="Horizontal"),
            image,
        )
        texts = []
        seen = set()
        for item in self._iter_ocr_results(result):
            text = getattr(item, "text", "")
            text = str(text or "")
            if not text:
                continue

            box = getattr(item, "box", None)
            key = (text, tuple(box) if box else None)
            text_key = ("text", text)
            if key in seen or text_key in seen:
                continue

            seen.add(key)
            seen.add(text_key)
            texts.append(text)
        return "".join(texts)

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

    def _parse_discount(self, text):
        match = re.search(r"([1-9])\s*折", str(text or ""))
        if match:
            return int(match.group(1))

        digits = re.sub(r"\D", "", str(text or ""))
        if len(digits) == 1:
            return int(digits)
        return None

    def _parse_price(self, text):
        numbers = re.findall(r"\d+", str(text or ""))
        if not numbers:
            return None
        return int(numbers[-1])

    def _is_refresh_free(self, context, image, roi, threshold):
        text = self._read_text(context, image, roi, threshold)
        normalized = self._normalize_name(text)
        is_free = "免费" in normalized
        logger.info(
            "memory_shop_check: refresh button text="
            f"{text!r}, normalized={normalized!r}, is_free={is_free}"
        )
        return is_free

    def _is_refresh_exhausted_popup(
        self,
        context,
        image,
        text_roi,
        ocr_threshold,
    ):
        text = self._read_text(context, image, text_roi, ocr_threshold)
        normalized = self._normalize_name(text)
        hit = "刷新卡" in normalized and "获取途径" in normalized
        logger.info(
            "memory_shop_check: refresh exhausted popup text="
            f"{text!r}, normalized={normalized!r}, hit={hit}"
        )
        return hit

    def _detect_currency(self, context, image, roi, threshold, gold_template, diamond_template):
        gold_hit = self._template_hit(context, image, roi, gold_template, threshold)
        diamond_hit = self._template_hit(context, image, roi, diamond_template, threshold)
        logger.info(
            "memory_shop_check: currency template hit "
            f"gold={gold_hit}, diamond={diamond_hit}, roi={roi}"
        )
        if gold_hit and not diamond_hit:
            return "gold"
        if diamond_hit and not gold_hit:
            return "diamond"
        if gold_hit:
            return "gold"
        if diamond_hit:
            return "diamond"
        return "unknown"

    def _template_hit(self, context, image, roi, template, threshold):
        try:
            result = context.run_recognition_direct(
                JRecognitionType.TemplateMatch,
                JTemplateMatch(
                    template=[template],
                    roi=tuple(roi),
                    threshold=[threshold],
                ),
                image,
            )
            return bool(result and getattr(result, "hit", False))
        except Exception as e:
            logger.warning(
                "memory_shop_check: currency template match failed, "
                f"template={template!r}, roi={roi}, error={e}"
            )
            return False

    def _confirm_if_needed(self, context, controller, text_roi, confirm_roi, threshold):
        image = self._screencap(controller)
        text = self._read_text(context, image, text_roi, threshold)
        if re.search(r"购买|确定|确认", text) or self._has_orange_button(image, confirm_roi):
            logger.info(f"memory_shop_check: confirming purchase popup, text={text!r}")
            self._click_roi(controller, confirm_roi)
            return True
        logger.info(f"memory_shop_check: no purchase popup detected, text={text!r}")
        return False

    def _recover_after_purchase(
        self,
        context,
        controller,
        reward_close_roi,
        shop_page_roi,
        back_roi,
        threshold,
        attempts,
        delay,
        close_clicks,
    ):
        for click_index in range(1, close_clicks + 1):
            logger.info(
                "memory_shop_check: clicking blank area after purchase, "
                f"click={click_index}/{close_clicks}"
            )
            self._click_roi(controller, reward_close_roi)
            time.sleep(delay)

        for attempt in range(1, attempts + 1):
            image = self._screencap(controller)
            if self._is_memory_shop_page(context, image, shop_page_roi, threshold):
                logger.info(f"memory_shop_check: returned to memory shop, attempt={attempt}/{attempts}")
                return True

            logger.info(
                "memory_shop_check: not on memory shop after purchase, clicking back, "
                f"attempt={attempt}/{attempts}"
            )
            self._click_roi(controller, back_roi)
            time.sleep(delay)
            self._click_roi(controller, reward_close_roi)
            time.sleep(delay)

        image = self._screencap(controller)
        return self._is_memory_shop_page(context, image, shop_page_roi, threshold)

    def _is_memory_shop_page(self, context, image, roi, threshold):
        text = self._read_text(context, image, roi, threshold)
        normalized = self._normalize_name(text)
        hit = "回忆小铺" in normalized and (
            "换一批" in normalized or "刷新卡" in normalized or "剩余" in normalized
        )
        logger.info(
            "memory_shop_check: memory shop page text="
            f"{text!r}, normalized={normalized!r}, hit={hit}"
        )
        return hit

    def _has_orange_button(self, image, roi):
        x, y, w, h = roi
        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            return False

        crop = crop.astype(np.int16)
        b = crop[:, :, 0]
        g = crop[:, :, 1]
        r = crop[:, :, 2]
        orange = np.count_nonzero((r > 190) & (g > 130) & (g < 210) & (b < 140))
        ratio = orange / (w * h)
        logger.info(f"memory_shop_check: confirm button orange_ratio={ratio:.3f}")
        return ratio > 0.12

    def _screencap(self, controller):
        controller.post_screencap().wait()
        return controller.cached_image

    def _click_roi(self, controller, roi):
        x, y, w, h = roi
        controller.post_click(x + w // 2, y + h // 2).wait()

    def _right_half_roi(self, roi):
        if not self._valid_roi(roi):
            return roi

        x, y, w, h = roi
        left = x + max(w // 2 - 18, 0)
        return [left, y, x + w - left, h]

    def _normalize_name(self, text):
        return re.sub(r"[\s:：,，.。·\-]+", "", str(text or ""))

    def _is_excluded_name(self, name, excluded_names):
        normalized = self._normalize_name(name)
        for excluded in excluded_names:
            excluded = self._normalize_name(excluded)
            if excluded and (normalized == excluded or excluded in normalized):
                return True
        return False

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
