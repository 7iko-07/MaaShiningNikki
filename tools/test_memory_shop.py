"""Offline shop selection, independent settings, and modal guard regressions."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.define import Rect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location("memory_shop_tested", ROOT / "agent/custom/memory_shop.py")
shop = importlib.util.module_from_spec(spec)
with patch.object(AgentServer, "register_custom_action", return_value=True):
    spec.loader.exec_module(shop)

PIPELINE = json.loads((ROOT / "assets/resource/pipeline/memory_shop.json").read_text(encoding="utf-8"))


def argument(node):
    return SimpleNamespace(
        node_name=node,
        custom_action_param=json.dumps(PIPELINE[node]["action"]["param"]["custom_action_param"]),
    )


def ocr_result(*items):
    return SimpleNamespace(
        hit=True,
        filtered_results=[SimpleNamespace(text=text, box=box) for text, box in items],
    )


class ShopTests(unittest.TestCase):
    def context(self):
        context = Mock()
        context.tasker.stopping = False
        context.override_next.return_value = True
        return context

    def selector(self, name="限时回忆小铺", selected=True):
        action = shop.MemoryShopSelectAction()
        context = self.context()
        context.get_node_data.return_value = {"attach": {"selected": selected}}
        context.run_recognition.return_value = SimpleNamespace(hit=True, box=Rect(200, 460, 100, 40))
        context.run_action.return_value = SimpleNamespace(success=True)
        action._screencap = Mock(return_value=None)
        action._has_blocking_popup = Mock(return_value=False)
        action._wait_shop_ready = Mock(return_value=(None, ocr_result()))
        action._is_memory_shop_page = Mock(return_value=True)
        action._recover_after_purchase = Mock(return_value=True)
        return action, context, argument("商城选择" + name)

    def clock(self):
        now = [0.0]
        return now, patch.object(shop.time, "monotonic", side_effect=lambda: now[0]), patch.object(
            shop.time, "sleep", side_effect=lambda delay: now.__setitem__(0, now[0] + delay)
        )

    def test_both_shops_use_recognized_boxes_and_reentry_starts_again(self):
        for _ in range(2):
            for name in ("限时回忆小铺", "回忆小铺"):
                action, context, argv = self.selector(name)
                self.assertTrue(action.run(context, argv))
                context.run_action.assert_called_once_with("商城点击" + name, box=[200, 460, 100, 40])
                context.override_next.assert_called_once_with(argv.node_name, ["商城检查" + name + "商品"])
                action._wait_shop_ready.assert_called_once()

    def test_unselected_shops_do_not_capture_or_click(self):
        for name, following in (("限时回忆小铺", ["商城选择回忆小铺"]), ("回忆小铺", ["stop"])):
            action, context, argv = self.selector(name, selected=False)
            self.assertTrue(action.run(context, argv))
            action._screencap.assert_not_called()
            context.run_action.assert_not_called()
            context.override_next.assert_called_once_with(argv.node_name, following)

    def test_missing_limited_shop_skips_but_missing_regular_fails(self):
        for name in ("限时回忆小铺", "回忆小铺"):
            action, context, argv = self.selector(name)
            context.run_recognition.return_value = SimpleNamespace(hit=False)
            now, monotonic, sleep = self.clock()
            with monotonic, sleep:
                self.assertEqual(action.run(context, argv), name == "限时回忆小铺")
            self.assertEqual(now[0], 3)
            context.run_action.assert_not_called()
            if name == "回忆小铺":
                context.override_next.assert_not_called()

    def test_modal_must_recover_before_any_tab_click(self):
        action, context, argv = self.selector()
        action._has_blocking_popup.return_value = True
        action._recover_after_purchase.side_effect = lambda *args: context.run_action.assert_not_called() or True
        self.assertTrue(action.run(context, argv))
        action._recover_after_purchase.assert_called_once()

    def test_failed_recovery_and_readiness_never_click_or_skip(self):
        for recovery_failure in (True, False):
            action, context, argv = self.selector()
            action._has_blocking_popup.return_value = recovery_failure
            action._recover_after_purchase.return_value = False
            action._wait_shop_ready.return_value = False
            self.assertFalse(action.run(context, argv))
            context.run_action.assert_not_called()
            context.override_next.assert_not_called()

    def test_persistent_modal_cannot_be_treated_as_missing_limited_shop(self):
        action, context, argv = self.selector()
        action._wait_shop_ready.return_value = None
        _, monotonic, sleep = self.clock()
        with monotonic, sleep:
            self.assertFalse(action.run(context, argv))
        context.run_recognition.assert_not_called()
        context.run_action.assert_not_called()
        context.override_next.assert_not_called()

    def test_bad_recognition_invalid_box_and_failed_click_stop(self):
        for result, success in (
            (None, True),
            (SimpleNamespace(hit=True, box=[0, 0, 0, 20]), True),
            (SimpleNamespace(hit=True, box=[0, 0, 20, 20]), False),
        ):
            action, context, argv = self.selector()
            context.run_recognition.return_value = result
            context.run_action.return_value = SimpleNamespace(success=success)
            self.assertFalse(action.run(context, argv))
            context.override_next.assert_not_called()

    def test_slow_recognition_and_cancellation_cannot_click(self):
        for cancel in (True, False):
            action, context, argv = self.selector()
            now, monotonic, sleep = self.clock()

            def recognize(*args):
                context.tasker.stopping = cancel
                now[0] += 4 if not cancel else 0
                return SimpleNamespace(hit=True, box=[10, 10, 20, 20])

            context.run_recognition.side_effect = recognize
            with monotonic, sleep:
                self.assertFalse(action.run(context, argv))
            context.run_action.assert_not_called()

    def checker(self, name="限时回忆小铺", currency="diamond", discount=9, price=100):
        action = shop.MemoryShopCheckAction()
        context = self.context()
        context.get_node_data.return_value = {}
        action._wait_shop_ready = Mock(return_value=(None, ocr_result()))
        action._screencap = Mock(return_value=None)
        action._read_product = Mock(return_value={
            "currency": currency, "name": "测试服装", "discount": discount, "price": price,
            "price_text": str(price), "button_roi": [176, 620, 176, 48],
        })
        action._is_refresh_free = Mock(return_value=True)
        action._is_refresh_exhausted_popup = Mock(return_value=False)
        action._click_roi = Mock(return_value=True)
        action._confirm_if_needed = Mock(return_value=True)
        action._recover_after_purchase = Mock(return_value=True)
        return action, context, argument("商城检查" + name + "商品")

    @patch.object(shop.time, "sleep")
    def test_gold_options_are_independent(self, _sleep):
        configs = {
            "回忆小铺自动购买金币商品": {"attach": {"value": True}},
            "限时回忆小铺自动购买金币商品": {"attach": {"value": False}},
        }
        for name in ("限时回忆小铺", "回忆小铺"):
            action, context, argv = self.checker(name, currency="gold")
            context.get_node_data.side_effect = configs.get
            self.assertTrue(action.run(context, argv))
            if name.startswith("限时"):
                context.override_next.assert_called_once_with(argv.node_name, ["stop"])
                action._click_roi.assert_not_called()
            else:
                self.assertEqual(action._click_roi.call_count, 7)
                context.override_next.assert_called_once_with(argv.node_name, [argv.node_name])

    @patch.object(shop.time, "sleep")
    def test_diamond_discount_and_price_options_are_independent(self, _sleep):
        for suffix, value, product in (
            ("钻石暂停折扣", 8, {"discount": 7}),
            ("钻石暂停价格", 101, {"price": 100}),
        ):
            for name in ("限时回忆小铺", "回忆小铺"):
                action, context, argv = self.checker(name, **product)
                context.get_node_data.side_effect = lambda node: {"attach": {"value": value}} if node == "限时回忆小铺" + suffix else {}
                self.assertTrue(action.run(context, argv))
                expected = ["stop"] if name.startswith("限时") else [argv.node_name]
                context.override_next.assert_called_once_with(argv.node_name, expected)

    @patch.object(shop.time, "sleep")
    def test_nonfree_options_are_independent_and_finish_only_current_shop(self, _sleep):
        for name in ("限时回忆小铺", "回忆小铺"):
            action, context, argv = self.checker(name)
            context.get_node_data.side_effect = lambda node: {"attach": {"value": True}} if node == "回忆小铺非免费继续刷新" else {}
            action._is_refresh_free.return_value = False
            self.assertTrue(action.run(context, argv))
            expected = ["商城选择回忆小铺"] if name.startswith("限时") else [argv.node_name]
            context.override_next.assert_called_once_with(argv.node_name, expected)
            self.assertEqual(action._click_roi.call_count, int(name == "回忆小铺"))

    @patch.object(shop.time, "sleep")
    def test_exhausted_refresh_cards_close_before_advancing(self, _sleep):
        for recovered in (True, False):
            action, context, argv = self.checker()
            action._is_refresh_exhausted_popup.return_value = True
            action._wait_shop_ready.side_effect = [(None, ocr_result()), recovered]
            self.assertEqual(action.run(context, argv), recovered)
            self.assertEqual(action._click_roi.call_args.args[1], [0, 36, 100, 64])
            if recovered:
                context.override_next.assert_called_once_with(argv.node_name, ["商城选择回忆小铺"])
            else:
                context.override_next.assert_not_called()

    @patch.object(shop.time, "sleep")
    def test_purchase_recovery_and_route_failures_propagate(self, _sleep):
        action, context, argv = self.checker(currency="gold")
        context.get_node_data.return_value = {"attach": {"value": True}}
        action._recover_after_purchase.return_value = False
        self.assertFalse(action.run(context, argv))
        context.override_next.assert_not_called()
        context.override_next.return_value = False
        self.assertFalse(action._route(context, argv.node_name, ["stop"]))

    def test_background_shop_text_does_not_override_known_modals(self):
        action = shop.MemoryShopCheckAction()
        context = self.context()
        context.run_recognition_direct.return_value = ocr_result(
            ("回忆小铺", [201, 462, 98, 28]), ("换一批", [600, 1175, 78, 29]),
        )
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        for modal in ("刷新卡获取途径", "确认购买", "恭喜获得", ""):
            def read(context, image, roi, threshold):
                if roi == [231, 270, 241, 104]:
                    return modal if modal.startswith("刷新卡") else ""
                if roi == [335, 711, 230, 68]:
                    return modal if modal.startswith("确认") else ""
                if roi == [100, 300, 520, 220]:
                    return modal if modal.startswith("恭喜") else ""
                return "回忆小铺换一批剩余"

            action._read_text = read
            self.assertEqual(action._is_memory_shop_page(context, image, [0, 448, 720, 790], 0.3), not modal)

    def test_readiness_reuses_clear_frame_and_has_deadline(self):
        image, result = object(), ocr_result()
        for frames, expected, elapsed in (([None, result], (image, result), 0.5), ([None] * 30, None, 10)):
            action = shop.MemoryShopCheckAction()
            context = self.context()
            action._screencap = Mock(return_value=image)
            action._read_shop_page = Mock(side_effect=frames)
            now, monotonic, sleep = self.clock()
            with monotonic, sleep:
                self.assertEqual(action._wait_shop_ready(context, context.tasker.controller), expected)
            self.assertEqual(now[0], elapsed)

    def test_bulk_results_assign_repeated_names_and_discounts_to_their_slots(self):
        action = shop.MemoryShopCheckAction()
        items = []
        for index, slot in enumerate(shop.DEFAULT_SLOTS):
            items.extend((("记忆钥匙·稀有", slot["name_roi"]), (f"{index + 2}折", slot["discount_roi"])))
        result = ocr_result(*items)
        for index, slot in enumerate(shop.DEFAULT_SLOTS):
            self.assertEqual(action._text_in_roi(result, slot["name_roi"]), "记忆钥匙·稀有")
            self.assertEqual(action._parse_discount(action._text_in_roi(result, slot["discount_roi"])), index + 2)

    def test_price_uses_local_current_price_only_when_needed(self):
        slot = shop.DEFAULT_SLOTS[0]
        for currency, name, threshold, expected in (
            ("diamond", "记忆钥匙·稀有", 100, None),
            ("diamond", "测试服装", 0, None),
            ("diamond", "测试服装", 100, 96),
            ("gold", "测试服装", 0, 96),
        ):
            action = shop.MemoryShopCheckAction()
            action._detect_currency = Mock(return_value=currency)
            action._read_text = Mock(return_value="96")
            result = ocr_result(
                (name, slot["name_roi"]), ("8折", slot["discount_roi"]),
                ("12096", [211, 629, 100, 25]),
            )
            product = action._read_product(
                None, None, slot, 0, 0.3, 0.8, "gold.png", "diamond.png",
                result, shop.EXCLUDED_DIAMOND_NAMES, threshold,
            )
            self.assertEqual(product["price"], expected)
            if expected is None:
                action._read_text.assert_not_called()
            else:
                action._read_text.assert_called_once_with(None, None, slot["price_roi"], 0.3)

    @patch.object(shop.time, "sleep")
    def test_excluded_batch_reuses_page_text_and_only_waits_after_refresh(self, sleep):
        action, context, argv = self.checker()
        del action._read_product, action._is_refresh_free
        items = [("免费", [542, 1176, 51, 27]), ("换一批", [601, 1175, 78, 29])]
        names = ("记忆钥匙·稀有", "记忆之轨", "记忆钥匙·普通", "时间之诗", "体力", "印象碎片·非凡")
        for name, slot in zip(names, shop.DEFAULT_SLOTS):
            items.extend(((name, slot["name_roi"]), ("2折", slot["discount_roi"])))
        action._wait_shop_ready.return_value = (None, ocr_result(*items))
        action._detect_currency = Mock(return_value="diamond")
        action._read_text = Mock()
        self.assertTrue(action.run(context, argv))
        action._read_text.assert_not_called()
        self.assertEqual(action._screencap.call_count, 1)  # Only the post-refresh popup check.
        sleep.assert_called_once_with(2.0)
        context.override_next.assert_called_once_with(argv.node_name, [argv.node_name])

    def test_recovery_slow_recognition_cannot_click_after_deadline(self):
        action = shop.MemoryShopCheckAction()
        context = self.context()
        action._screencap = Mock(return_value=None)
        action._click_roi = Mock(return_value=True)
        now, monotonic, sleep = self.clock()

        def recognize(*args):
            now[0] += 30
            return False

        action._is_memory_shop_page = recognize
        with monotonic, sleep:
            self.assertFalse(action._recover_after_purchase(
                context, context.tasker.controller, [0, 0, 1, 1], [0, 0, 1, 1],
                [0, 0, 1, 1], 0.3, 8, 0.8, 0,
            ))
        action._click_roi.assert_not_called()

    def test_capture_and_click_job_failures_are_checked(self):
        action = shop.MemoryShopCheckAction()
        controller = Mock()
        controller.post_screencap.return_value.wait.return_value.succeeded = False
        controller.post_click.return_value.wait.return_value.succeeded = False
        with self.assertRaisesRegex(RuntimeError, "screenshot failed"):
            action._screencap(controller)
        self.assertFalse(action._click_roi(controller, [10, 20, 40, 60]))


if __name__ == "__main__":
    unittest.main()
