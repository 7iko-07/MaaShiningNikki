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
        # Independent UI options use separate nodes so their overrides do not
        # replace one another's custom_action_param object.
        shop = params.get("shop", "回忆小铺")
        for key, node_name in (
            ("auto_buy_gold", "回忆小铺自动购买金币商品"),
            ("diamond_discount_threshold", "回忆小铺钻石暂停折扣"),
            ("diamond_price_threshold", "回忆小铺钻石暂停价格"),
            ("continue_refresh_when_not_free", "回忆小铺非免费继续刷新"),
        ):
            config = context.get_node_data(node_name.replace("回忆小铺", shop, 1)) or {}
            attached = config.get("attach", {})
            if "value" in attached:
                params[key] = attached["value"]

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
        finish_next = params.get("finish_next", ["stop"])
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
        if not self._wait_shop_ready(context, controller):
            logger.error(f"memory_shop_check: {shop} page is not ready")
            return False
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
            logger.debug(
                "memory_shop_check: "
                f"shop={shop}, slot={index + 1}, name={product['name']!r}, discount={product['discount']}, "
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
            return self._route(context, argv.node_name, pause_next)

        if auto_buy_gold:
            for product in gold_products:
                if context.tasker.stopping or not self._click_roi(controller, product["button_roi"]):
                    return False
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
                    return False

        if diamond_price_pause_products:
            logger.info(
                "memory_shop_check: low-price diamond product found, pausing: "
                + ", ".join(
                    f"{item['name']}({item['price']}钻)"
                    for item in diamond_price_pause_products
                )
            )
            return self._route(context, argv.node_name, pause_next)

        if diamond_pause_products:
            logger.info(
                "memory_shop_check: valuable diamond product found, pausing: "
                + ", ".join(
                    f"{item['name']}({item['discount']}折)"
                    for item in diamond_pause_products
                )
            )
            return self._route(context, argv.node_name, pause_next)

        image = self._screencap(controller)
        refresh_free = self._is_refresh_free(context, image, refresh_text_roi, ocr_threshold)
        if not refresh_free and not continue_refresh_when_not_free:
            logger.info(f"memory_shop_check: {shop} refresh is not free, finishing shop")
            return self._route(context, argv.node_name, finish_next)

        logger.info(
            "memory_shop_check: no product matched pause rules, refreshing, "
            f"refresh_free={refresh_free}"
        )
        if context.tasker.stopping or not self._click_roi(controller, refresh_roi):
            return False
        time.sleep(post_refresh_delay)
        image = self._screencap(controller)
        if self._is_refresh_exhausted_popup(
            context,
            image,
            refresh_exhausted_roi,
            ocr_threshold,
        ):
            logger.info(f"memory_shop_check: {shop} refresh cards exhausted, closing popup")
            if not self._click_roi(controller, back_roi):
                return False
            if not self._wait_shop_ready(context, controller):
                logger.error("memory_shop_check: refresh popup did not close")
                return False
            return self._route(context, argv.node_name, finish_next)

        return self._route(context, argv.node_name, refresh_next)

    def _route(self, context, node_name, next_nodes):
        success = context.override_next(node_name, next_nodes)
        if not success:
            logger.error(f"memory_shop_check: cannot set {node_name}.next={next_nodes}")
        return bool(success)

    def _wait_shop_ready(self, context, controller, timeout=10.0):
        """Require two fresh shop frames without a known modal; never click here."""
        deadline = time.monotonic() + timeout
        clear_frames = 0
        while time.monotonic() < deadline and not context.tasker.stopping:
            image = self._screencap(controller)
            ready = self._is_memory_shop_page(context, image, [0, 448, 720, 790], 0.3)
            if time.monotonic() >= deadline or context.tasker.stopping:
                return False
            clear_frames = clear_frames + 1 if ready else 0
            if clear_frames >= 2:
                return True
            time.sleep(min(0.5, max(0, deadline - time.monotonic())))
        return False

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
        logger.debug(
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
        logger.debug(
            "memory_shop_check: refresh exhausted popup text="
            f"{text!r}, normalized={normalized!r}, hit={hit}"
        )
        return hit

    def _detect_currency(self, context, image, roi, threshold, gold_template, diamond_template):
        gold_hit = self._template_hit(context, image, roi, gold_template, threshold)
        diamond_hit = self._template_hit(context, image, roi, diamond_template, threshold)
        logger.debug(
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
        logger.debug(f"memory_shop_check: no purchase popup detected, text={text!r}")
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
        deadline = time.monotonic() + max(10.0, (2 * attempts + close_clicks) * delay)
        for click_index in range(1, close_clicks + 1):
            logger.debug(
                "memory_shop_check: clicking blank area after purchase, "
                f"click={click_index}/{close_clicks}"
            )
            if (
                context.tasker.stopping or time.monotonic() >= deadline
                or not self._click_roi(controller, reward_close_roi)
            ):
                return False
            time.sleep(delay)

        for attempt in range(1, attempts + 1):
            if context.tasker.stopping or time.monotonic() >= deadline:
                return False
            image = self._screencap(controller)
            ready = self._is_memory_shop_page(context, image, shop_page_roi, threshold)
            if context.tasker.stopping or time.monotonic() >= deadline:
                return False
            if ready:
                logger.debug(f"memory_shop_check: returned to memory shop, attempt={attempt}/{attempts}")
                return True

            logger.debug(
                "memory_shop_check: not on memory shop after purchase, clicking back, "
                f"attempt={attempt}/{attempts}"
            )
            if context.tasker.stopping or not self._click_roi(controller, back_roi):
                return False
            time.sleep(delay)
            if (
                context.tasker.stopping or time.monotonic() >= deadline
                or not self._click_roi(controller, reward_close_roi)
            ):
                return False
            time.sleep(delay)

        if context.tasker.stopping or time.monotonic() >= deadline:
            return False
        image = self._screencap(controller)
        ready = self._is_memory_shop_page(context, image, shop_page_roi, threshold)
        return ready and not context.tasker.stopping and time.monotonic() < deadline

    def _is_memory_shop_page(self, context, image, roi, threshold):
        if image is None or self._has_blocking_popup(context, image, threshold):
            return False
        text = self._read_text(context, image, roi, threshold)
        normalized = self._normalize_name(text)
        hit = "回忆小铺" in normalized and (
            "换一批" in normalized or "刷新卡" in normalized or "剩余" in normalized
        )
        logger.debug(
            "memory_shop_check: memory shop page text="
            f"{text!r}, normalized={normalized!r}, hit={hit}"
        )
        return hit

    def _has_blocking_popup(self, context, image, threshold):
        if self._is_refresh_exhausted_popup(context, image, [231, 270, 241, 104], threshold):
            return True
        confirm_text = self._read_text(context, image, [335, 711, 230, 68], threshold)
        if re.search(r"购买|确定|确认", confirm_text) or self._has_orange_button(image, [122, 708, 490, 72]):
            return True
        reward_text = self._read_text(context, image, [100, 300, 520, 220], threshold)
        return bool(re.search(r"恭喜获得|获得奖励", reward_text))

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
        logger.debug(f"memory_shop_check: confirm button orange_ratio={ratio:.3f}")
        return ratio > 0.12

    def _screencap(self, controller):
        if not controller.post_screencap().wait().succeeded:
            raise RuntimeError("memory_shop_check: screenshot failed")
        return controller.cached_image

    def _click_roi(self, controller, roi):
        x, y, w, h = roi
        return bool(controller.post_click(x + w // 2, y + h // 2).wait().succeeded)

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


@AgentServer.custom_action("memory_shop_select")
class MemoryShopSelectAction(MemoryShopCheckAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        params = json.loads(argv.custom_action_param)
        config = context.get_node_data(argv.node_name) or {}
        # Selection and progress live in this task's explicit pipeline edges.
        # Every invocation writes its next edge, including skipped shops.
        if not self._as_bool(config.get("attach", {}).get("selected"), False):
            return self._route(context, argv.node_name, params["finish_next"])

        shop = params["shop"]
        controller = context.tasker.controller
        image = self._screencap(controller)
        if self._has_blocking_popup(context, image, 0.3):
            if not self._recover_after_purchase(
                context, controller, [318, 1183, 69, 35], [0, 448, 720, 790],
                [0, 36, 100, 64], 0.3, 8, 0.8, 3,
            ):
                logger.error(f"memory_shop_select: cannot close popup before {shop}")
                return False
        if not self._wait_shop_ready(context, controller):
            logger.error(f"memory_shop_select: page is not ready before {shop}")
            return False

        deadline = time.monotonic() + 3.0
        clean_missing = False
        while time.monotonic() < deadline and not context.tasker.stopping:
            image = self._screencap(controller)
            ready = self._is_memory_shop_page(context, image, [0, 448, 720, 790], 0.3)
            clean_missing = False
            if ready:
                result = context.run_recognition(params["tab_node"], image)
                if result is None:
                    logger.error(f"memory_shop_select: {shop} label recognition failed")
                    return False
                if time.monotonic() >= deadline or context.tasker.stopping:
                    return False
                if result.hit:
                    box = result.box
                    if hasattr(box, "x"):
                        box = [box.x, box.y, box.w, box.h]
                    if not isinstance(box, (list, tuple)) or len(box) != 4 or box[2] <= 0 or box[3] <= 0:
                        logger.error(f"memory_shop_select: invalid {shop} label box")
                        return False
                    action = context.run_action(params["tab_node"], box=box)
                    if not action or not action.success:
                        return False
                    if not self._wait_shop_ready(context, controller):
                        logger.error(f"memory_shop_select: {shop} page did not become ready")
                        return False
                    logger.info(f"memory_shop_select: 已切换到{shop}")
                    return self._route(context, argv.node_name, [params["check_node"]])
                clean_missing = True
            time.sleep(min(0.5, max(0, deadline - time.monotonic())))

        if shop == "限时回忆小铺" and clean_missing and not context.tasker.stopping:
            logger.info("memory_shop_select: 未找到限时回忆小铺，跳过")
            return self._route(context, argv.node_name, params["finish_next"])
        logger.error(f"memory_shop_select: {shop} label not found or page blocked")
        return False
