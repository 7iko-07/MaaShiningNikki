"""Offline purchase tests: no device clicks or currency spent."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from maa.agent.agent_server import AgentServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location("union_shop_tested", ROOT / "agent/custom/union_shop.py")
module = importlib.util.module_from_spec(spec)
with patch.object(AgentServer, "register_custom_action", return_value=True):
    spec.loader.exec_module(module)


class ShopTests(unittest.TestCase):
    def test_report_logs_once_without_running_notification_task(self):
        action = module.UnionShopBuy()
        context = Mock()
        for warning in (False, True):
            with self.subTest(warning=warning), patch.object(module, "logger") as logger:
                action._report(context, "购买结果", warning)
                expected = logger.warning if warning else logger.info
                expected.assert_called_once_with("购买结果")
                self.assertEqual(len(logger.mock_calls), 1)
        context.run_task.assert_not_called()

    def simulated(self, balance=7869, stock=3, bad_total=False, bad_debit=False):
        action = module.UnionShopBuy()
        state = {"quantity": 1, "balance": balance, "purchases": 0}
        action._image = Mock(return_value=None)
        action._ocr = Mock(return_value=SimpleNamespace(hit=True, best_result=SimpleNamespace(box=[280, 550, 100, 32])))
        action._balance = lambda *args: state["balance"]
        action._close = Mock()
        action._report = Mock()

        def text(context, image, roi):
            if roi == action.TITLE:
                return "幻之券"
            if roi == action.QUANTITY:
                return str(state["quantity"])
            if roi == action.PRICE:
                return str(120 * state["quantity"] + (1 if bad_total and state["quantity"] > 1 else 0))
            if roi == [393, 716, 160, 59]:
                return "立即购买"
            return f"本周剩余{stock}/3"

        def click(context, roi):
            if roi == action.PLUS:
                state["quantity"] += 1
            elif roi == action.MINUS:
                state["quantity"] -= 1
            elif roi == [400, 723, 130, 42]:
                state["purchases"] += 1
                if not bad_debit:
                    state["balance"] -= 120 * state["quantity"]
        action._text = text
        action._click = click
        return action, state

    @patch.object(module.time, "sleep")
    def test_full_partial_zero_and_stock(self, _sleep):
        for balance, stock, expected in [(7869, 3, 3), (200, 3, 1), (80, 3, 0), (7869, 1, 1), (7869, 0, 0)]:
            with self.subTest(balance=balance, stock=stock):
                action, state = self.simulated(balance, stock)
                count, reason = action._buy(None, "幻之券", 3)
                self.assertEqual(count, expected)
                self.assertEqual(state["purchases"], int(expected > 0))
                self.assertEqual(state["balance"], balance - 120 * expected)
                if expected < 3:
                    self.assertTrue(reason)

    @patch.object(module.time, "sleep")
    def test_bad_total_never_purchases(self, _sleep):
        action, state = self.simulated(bad_total=True)
        with self.assertRaisesRegex(ValueError, "购买前"):
            action._buy(None, "幻之券", 3)
        self.assertEqual(state["purchases"], 0)

    @patch.object(module.time, "sleep")
    def test_unconfirmed_purchase_never_retries(self, _sleep):
        action, state = self.simulated(bad_debit=True)
        with self.assertRaisesRegex(ValueError, "扣款核验失败"):
            action._buy(None, "幻之券", 3)
        self.assertEqual(state["purchases"], 1)

    def test_insufficient_balance_continues_and_summarizes(self):
        action = module.UnionShopBuy()
        context = Mock()
        context.get_node_data.return_value = {"enabled": True, "repeat": 3}
        action._buy = Mock(side_effect=[(0, "联盟币不足"), (1, "联盟币不足")])
        action._report = Mock()
        self.assertTrue(action.run(context, None))
        self.assertEqual([c.args[1] for c in action._buy.call_args_list], ["幻之券", "绮之券"])
        self.assertIn("部分完成", action._report.call_args.args[1])

    def test_unselected_skipped(self):
        action = module.UnionShopBuy()
        context = Mock()
        context.get_node_data.return_value = {"enabled": False}
        action._buy = Mock()
        action._report = Mock()
        self.assertTrue(action.run(context, None))
        action._buy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
