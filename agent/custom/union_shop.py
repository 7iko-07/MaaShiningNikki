"""联盟商店按勾选顺序之外的固定顺序兑换，并核验实际扣款。"""
import re
import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.pipeline import JClick, JOCR, JTemplateMatch
from utils import logger, extract_ocr_text


@AgentServer.custom_action("union_shop_buy")
class UnionShopBuy(CustomAction):
    MINUS = [308, 656, 38, 38]
    PLUS = [435, 656, 38, 38]
    QUANTITY = [355, 656, 70, 38]
    STOCK = [162, 660, 127, 32]
    PRICE = [285, 719, 87, 48]
    TITLE = [275, 454, 220, 57]
    BLANK = [351, 221, 17, 12]

    def _image(self, context):
        controller = context.tasker.controller
        controller.post_screencap().wait()
        return controller.cached_image

    def _ocr(self, context, image, roi, expected=None):
        return context.run_recognition_direct(
            "OCR", JOCR(roi=roi, expected=expected or []), image
        )

    def _text(self, context, image, roi):
        return extract_ocr_text(self._ocr(context, image, roi))

    @staticmethod
    def _number(text):
        text = re.sub(r"[\s,，]", "", text)
        return int(text) if re.fullmatch(r"\d+", text) else None

    def _balance(self, context, image):
        result = context.run_recognition_direct(
            "TemplateMatch",
            JTemplateMatch(roi=[550, 0, 130, 38], template=["联盟币.png"], threshold=[0.8]),
            image,
        )
        if not result or not result.hit:
            raise ValueError("无法定位右上角联盟币图标")
        x, y, w, h = result.best_result.box
        left = x + w
        value = self._number(self._text(context, image, [left, 0, 678 - left, 38]))
        if value is None:
            raise ValueError("无法识别联盟币余额")
        return value

    def _click(self, context, roi):
        result = context.run_action_direct("Click", JClick(target=roi))
        if not result or not result.success:
            raise ValueError("点击失败")
        time.sleep(0.6)

    def _report(self, context, message, warning=False):
        (logger.warning if warning else logger.info)(message)

    def _close(self, context):
        self._click(context, self.BLANK)
        self._click(context, self.BLANK)
        image = self._image(context)
        if "立即购买" in self._text(context, image, [393, 716, 160, 59]):
            raise ValueError("购买弹窗未关闭")

    def _buy(self, context, name, requested):
        image = self._image(context)
        result = self._ocr(context, image, [164, 550, 556, 682], [name])
        if not result or not result.hit:
            raise ValueError(f"未找到{name}商品")
        box = list(result.best_result.box)
        # 商品卡片中剩余次数位于名称下方；售罄时不尝试打开弹窗。
        stock_text = self._text(context, image, [box[0], box[1] + box[3], min(190, 720 - box[0]), 48])
        match = re.search(r"(\d+)\s*/\s*(\d+)", stock_text)
        if match and int(match[1]) == 0:
            return 0, "本周已售罄"
        self._click(context, box)
        image = self._image(context)
        if name not in self._text(context, image, self.TITLE):
            raise ValueError(f"未确认{name}购买弹窗")
        quantity = self._number(self._text(context, image, self.QUANTITY))
        if quantity is None or not 1 <= quantity <= 3:
            raise ValueError("无法确认初始购买数量")
        for _ in range(quantity - 1):
            self._click(context, self.MINUS)
        image = self._image(context)
        if self._number(self._text(context, image, self.QUANTITY)) != 1:
            raise ValueError("购买数量未归一")
        price = self._number(self._text(context, image, self.PRICE))
        stock_text = self._text(context, image, self.STOCK)
        stock = re.search(r"(\d+)\s*/\s*(\d+)", stock_text)
        if price is None or price <= 0 or not stock:
            raise ValueError("无法确认单价或本周剩余数量")
        balance = self._balance(context, image)
        remaining = int(stock[1])
        count = min(requested, remaining, balance // price)
        reasons = []
        if remaining < requested:
            reasons.append(f"库存不足（剩余{remaining}张）")
        if balance // price < min(requested, remaining):
            reasons.append(f"联盟币不足（余额{balance}，单价{price}）")
        reason = "；".join(reasons)
        if count == 0:
            self._close(context)
            return 0, reason or "无可购买数量"
        for _ in range(count - 1):
            self._click(context, self.PLUS)
        image = self._image(context)
        if (self._number(self._text(context, image, self.QUANTITY)) != count
                or self._number(self._text(context, image, self.PRICE)) != price * count
                or self._balance(context, image) != balance
                or name not in self._text(context, image, self.TITLE)
                or "立即购买" not in self._text(context, image, [393, 716, 160, 59])):
            raise ValueError("购买前数量、总价、余额或商品核验失败")
        self._click(context, [400, 723, 130, 42])
        time.sleep(1)
        self._close(context)
        after = self._balance(context, self._image(context))
        if balance - after != price * count:
            raise ValueError("购买后扣款核验失败，实际购买结果待检查；不会重试购买")
        self._report(context, f"{name}实际购买{count}张，花费{price * count}联盟币，剩余{after}")
        return count, reason

    def run(self, context, argv):
        results = []
        partial = False
        for name in ("幻之券", "绮之券"):
            config = context.get_node_data(f"联盟商店选择{name}")
            if not config or not config.get("enabled", False):
                continue
            requested = config.get("repeat", 3)
            try:
                if not isinstance(requested, int) or not 1 <= requested <= 3:
                    raise ValueError("购买数量必须为1～3")
                count, reason = self._buy(context, name, requested)
            except Exception as exc:
                message = f"联盟商店中止：{name}，{exc}"
                if results:
                    message += "；此前结果：" + "；".join(results)
                self._report(context, message, True)
                return False
            item = f"{name}{count}/{requested}张" + (f"：{reason}" if reason else "")
            results.append(item)
            partial |= count < requested
            if reason:
                self._report(context, item, True)
        self._report(context, "联盟商店" + ("部分完成：" if partial else "完成：")
                     + ("；".join(results) or "未勾选商品，跳过"), partial)
        return True
