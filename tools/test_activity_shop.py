"""原生调度验证活动商店购买顺序、弹窗等待及失败出口，不连接游戏。"""

import copy
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from maa.library import Library

Library.version()

from maa.controller import CustomController
from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.define import LoggingLevelEnum
from maa.resource import Resource
from maa.tasker import Tasker


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = json.loads((ROOT / "assets/resource/pipeline/activity_shop.json").read_text(encoding="utf-8"))
spec = importlib.util.spec_from_file_location("activity_shop_click_region", ROOT / "agent/custom/click_region.py")
click_region = importlib.util.module_from_spec(spec)
with patch.object(AgentServer, "register_custom_action", return_value=True):
    spec.loader.exec_module(click_region)


class ShopScreen(CustomController):
    def __init__(self):
        super().__init__()
        self.stage = "shop"
        self.clicks = []
        self.cycles = 0
        self.fail_stage = None
        self.missing = None
        self.pending = 0
        self.outside_roi = False
        self.cancel_at = None
        self.max_box = (498, 657, 39, 35)
        self.insufficient_after = None
        self.toast_frames = 0

    def connect(self): return True
    def request_uuid(self): return "activity-shop-offline"
    def get_features(self): return 0

    def screencap(self):
        if self.cancel_at == self.stage:
            self.tasker.post_stop()
        if self.stage == "insufficient":
            self.toast_frames += 1
        if self.pending:
            self.pending -= 1
            if self.stage == "closing" and not self.pending:
                self.stage = "shop" if self.cycles < 2 else "done"
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        if self.stage in ("shop", "closing"):
            # 保留真实 ColorMatch；右侧商品不能成为点击目标。
            x = 530 if self.outside_roi else 180
            image[650:695, x:x + 165] = [128, 173, 229]
        return image

    def visible(self, kind):
        if kind == self.missing:
            return False
        if kind == "不足":
            return self.stage == "insufficient" and self.toast_frames == 1
        if kind == "MAX":
            # 关闭奖励动画结束后，MAX 才重新出现。
            return (self.stage in ("dialog", "maxed")
                    or self.stage == "after_reward" and not self.pending)
        if kind == "立即购买":
            return self.stage in ("dialog", "maxed", "reward", "after_reward", "closing")
        if kind == "恭喜获得":
            # 关闭奖励动画期间，购买文字也可见，不能再次点空白。
            return (self.stage == "reward" and not self.pending
                    or self.stage == "after_reward" and self.pending > 0)
        return False

    def click(self, x, y):
        self.clicks.append((self.stage, x, y))
        if self.stage == self.fail_stage:
            return False
        if self.stage == "shop":
            self.stage = "dialog"
        elif self.stage == "dialog":
            self.stage = "maxed"
        elif self.stage == "maxed":
            if self.insufficient_after is not None and self.cycles >= self.insufficient_after:
                self.stage = "insufficient"
            else:
                self.stage, self.pending = "reward", 3
        elif self.stage == "reward":
            self.stage, self.pending = "after_reward", 3
        elif self.stage == "after_reward":
            self.cycles += 1
            self.stage, self.pending = "closing", 3
        else:
            return False
        return True

    def unused(self, *args): return False
    start_app = stop_app = swipe = touch_down = touch_move = touch_up = unused
    click_key = input_text = key_down = key_up = unused


class ShopText(CustomRecognition):
    def __init__(self, screen):
        super().__init__()
        self.screen = screen

    def analyze(self, context, argv):
        kind = json.loads(argv.custom_recognition_param)
        if not self.screen.visible(kind):
            return None
        return {"MAX": self.screen.max_box, "立即购买": (403, 731, 106, 27),
                "恭喜获得": (240, 420, 240, 45), "不足": (290, 630, 140, 24)}[kind]


class ActivityShopTests(unittest.TestCase):
    def setUp(self):
        Tasker.set_stdout_level(LoggingLevelEnum.Off)
        self.resource = Resource()
        self.screen = ShopScreen()
        self.assertTrue(self.screen.post_connection().wait().succeeded)
        self.tasker = Tasker()
        self.assertTrue(self.tasker.bind(self.resource, self.screen))
        self.screen.tasker = self.tasker
        self.text = ShopText(self.screen)
        self.assertTrue(self.resource.register_custom_recognition("shop_text", self.text))
        self.center = click_region.ClickCenterAction()
        self.assertTrue(self.resource.register_custom_action("click_center", self.center))
        self.nodes = copy.deepcopy(PIPELINE)

        def replace_ocr(value):
            if isinstance(value, dict):
                if value.get("type") == "OCR":
                    kind = value["param"]["expected"][0]
                    value.update(type="Custom", param={"custom_recognition": "shop_text",
                                                       "custom_recognition_param": kind})
                else:
                    for child in value.values():
                        replace_ocr(child)
            elif isinstance(value, list):
                for child in value:
                    replace_ocr(child)

        replace_ocr(self.nodes)
        for node in self.nodes.values():
            node.update(timeout=200, rate_limit=5, pre_delay=0, post_delay=0)
        self.assertTrue(self.resource.override_pipeline(self.nodes))

    def run_shop(self):
        return self.tasker.post_task("活动商店任务链入口").wait().succeeded

    def test_two_complete_cycles_then_missing_button_is_failure(self):
        self.assertFalse(self.run_shop())
        self.assertEqual(self.screen.cycles, 2)
        self.assertEqual([stage for stage, _, _ in self.screen.clicks],
                         ["shop", "dialog", "maxed", "reward", "after_reward"] * 2)
        for stage, x, y in self.screen.clicks:
            if stage == "shop":
                self.assertTrue(180 <= x < 345 and 650 <= y < 695)
            elif stage == "dialog":
                self.assertEqual((x, y), (517, 674))
            elif stage in ("reward", "after_reward"):
                self.assertTrue(342 <= x < 367 and 296 <= y < 315)

    def test_left_shifted_max_uses_text_center_and_closes_with_blank(self):
        self.screen.max_box = (436, 661, 40, 23)
        self.assertFalse(self.run_shop())
        self.assertEqual(self.screen.cycles, 2)
        for stage, x, y in self.screen.clicks:
            if stage == "dialog":
                self.assertEqual((x, y), (456, 672))
            elif stage == "after_reward":
                self.assertTrue(342 <= x < 367 and 296 <= y < 315)

    def test_orange_outside_roi_never_clicks(self):
        self.screen.outside_roi = True
        self.assertFalse(self.run_shop())
        self.assertEqual(self.screen.clicks, [])

    def test_single_frame_insufficient_ends_successfully_without_further_clicks(self):
        self.screen.insufficient_after = 0
        self.assertTrue(self.run_shop())
        self.assertEqual(self.screen.toast_frames, 1)
        self.assertEqual(self.screen.cycles, 0)
        self.assertEqual([stage for stage, _, _ in self.screen.clicks], ["shop", "dialog", "maxed"])

    def test_reward_cycle_can_continue_then_end_on_insufficient(self):
        self.screen.insufficient_after = 1
        self.assertTrue(self.run_shop())
        self.assertEqual(self.screen.cycles, 1)
        self.assertEqual(self.screen.toast_frames, 1)
        self.assertEqual([stage for stage, _, _ in self.screen.clicks],
                         ["shop", "dialog", "maxed", "reward", "after_reward", "shop", "dialog", "maxed"])

    def test_missing_popup_cannot_advance_or_repeat_purchase(self):
        for missing, expected in [("MAX", ["shop"]),
                                  ("立即购买", ["shop", "dialog"]),
                                  ("恭喜获得", ["shop", "dialog", "maxed"])]:
            with self.subTest(missing=missing):
                self.screen.stage, self.screen.clicks = "shop", []
                self.screen.missing = missing
                self.assertFalse(self.run_shop())
                self.assertEqual([s for s, _, _ in self.screen.clicks], expected)

    def test_failed_purchase_click_does_not_retry(self):
        self.screen.fail_stage = "maxed"
        self.assertFalse(self.run_shop())
        self.assertEqual([s for s, _, _ in self.screen.clicks], ["shop", "dialog", "maxed"])

    def test_close_waits_for_max_even_when_purchase_text_is_visible(self):
        visible = self.screen.visible
        with patch.object(self.screen, "visible", side_effect=lambda kind: (
                False if kind == "MAX" and self.screen.stage == "after_reward" else visible(kind))):
            self.assertFalse(self.run_shop())
        self.assertEqual([s for s, _, _ in self.screen.clicks],
                         ["shop", "dialog", "maxed", "reward"])

    def test_manual_stop_while_waiting_does_not_click(self):
        self.screen.cancel_at = "shop"
        self.screen.outside_roi = True
        self.assertFalse(self.run_shop())
        self.assertEqual(self.screen.clicks, [])


if __name__ == "__main__":
    unittest.main()
