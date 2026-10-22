"""兑换码批次继续/停止和 UI 消息协议的离线检查。"""

import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch, call
from maa.agent.agent_server import AgentServer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location("redeem_codes_under_test", ROOT / "agent/custom/redeem_codes.py")
redeem = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = redeem
with patch.object(AgentServer, "register_custom_action", return_value=True):
    spec.loader.exec_module(redeem)


class RedeemCodesTests(unittest.TestCase):
    def execute(self, codes, outcomes, stopping=False):
        context = SimpleNamespace(tasker=SimpleNamespace(stopping=stopping))
        args = SimpleNamespace(custom_action_param=json.dumps({"codes": codes, "run_id": "batch-1"}))
        stream = io.StringIO()
        runner = Mock()
        action = runner.redeem_one
        action.side_effect = outcomes
        with redirect_stdout(stream), patch.object(redeem, "RedeemRunner", return_value=runner):
            result = redeem.RedeemCodesAction().run(context, args)
        messages = [json.loads(line.removeprefix("MFA_REDEEM:")) for line in stream.getvalue().splitlines()]
        self.assertTrue(all(message["run_id"] == "batch-1" for message in messages))
        self.assertEqual(messages[-1]["kind"], "end")
        return result, messages, action

    def test_missing_ui_codes_reports_error_without_game_action(self):
        result, messages, action = self.execute(None, [])
        self.assertFalse(result)
        self.assertTrue(any(m["kind"] == "error" for m in messages))
        action.assert_not_called()

    def test_failed_code_continues_without_reporting_success(self):
        result, messages, action = self.execute(["A", "B"], [
            redeem.RedemptionResult(False, "已兑换"),
            redeem.RedemptionResult(True, "兑换成功"),
        ])
        self.assertFalse(result)
        self.assertEqual(action.call_count, 2)
        self.assertEqual([m["success"] for m in messages if m["kind"] == "result"], [False, True])

    def test_page_failure_stops_remaining_codes(self):
        result, messages, action = self.execute(["A", "B", "C"], [
            redeem.RedemptionResult(False, "已兑换"), redeem.RedemptionError("页面不可用")])
        self.assertFalse(result)
        self.assertEqual(action.call_count, 2)
        self.assertEqual([m["code"] for m in messages if m["kind"] == "running"], ["A", "B"])

    def test_single_code_and_deduplication(self):
        result, messages, action = self.execute(["中文兑换码", "中文兑换码"], [redeem.RedemptionResult(True, "兑换成功")])
        self.assertTrue(result)
        self.assertEqual(action.call_count, 1)
        self.assertEqual(messages[0]["codes"], ["中文兑换码"])

    def test_empty_explicit_list_never_operates_game(self):
        result, _, action = self.execute([], [])
        self.assertTrue(result)
        action.assert_not_called()

    def test_cancel_does_not_attempt_a_code(self):
        result, _, action = self.execute(["A"], [], stopping=True)
        self.assertFalse(result)
        action.assert_not_called()

    def test_exception_reports_error_and_ends_batch(self):
        result, messages, _ = self.execute(["A"], RuntimeError("页面无法继续"))
        self.assertFalse(result)
        self.assertEqual(messages[-2]["kind"], "error")

    def test_success_is_reported_even_if_closing_reward_fails(self):
        runner = Mock()
        runner.redeem_one.return_value = redeem.RedemptionResult(True, "兑换成功")
        runner.close_reward.side_effect = redeem.RedemptionError("弹窗未关闭")
        stream = io.StringIO()
        with redirect_stdout(stream), patch.object(redeem, "RedeemRunner", return_value=runner):
            result = redeem.RedeemCodesAction().run(
                SimpleNamespace(tasker=SimpleNamespace(stopping=False)),
                SimpleNamespace(custom_action_param=json.dumps({"codes": ["A", "B"]})))
        messages = [json.loads(line.removeprefix("MFA_REDEEM:")) for line in stream.getvalue().splitlines()]
        self.assertFalse(result)
        runner.redeem_one.assert_called_once_with("A")
        self.assertTrue(next(m for m in messages if m["kind"] == "result")["success"])
        self.assertEqual(messages[-2]["kind"], "error")


class RedeemFlowTests(unittest.TestCase):
    def setUp(self):
        self.context = Mock()
        self.context.tasker.stopping = False
        self.runner = redeem.RedeemRunner(self.context)
        self.now = 0
        for patcher in (
            patch.object(redeem.time, "monotonic", side_effect=lambda: self.now),
            patch.object(redeem.time, "sleep", side_effect=self.advance),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def advance(self, seconds):
        self.now += seconds

    def test_failure_toast_is_reported_immediately(self):
        self.runner.image = Mock(return_value="frame")
        self.runner.hit = Mock(return_value=False)
        for text in ("兑换码已失效!", "您已使用过此兑换码!", "兑换码不存在!"):
            self.runner.reco = Mock(return_value=SimpleNamespace(hit=True, best_result=SimpleNamespace(text=text)))
            result = self.runner.wait_result()
            self.assertFalse(result.success)
            self.assertEqual(result.message, text)
            self.assertEqual(self.now, 0)

    def test_success_does_not_wait_for_deadline_or_read_failure(self):
        self.runner.image = Mock(return_value="frame")
        self.runner.hit = Mock(return_value=True)
        self.runner.reco = Mock()
        self.assertTrue(self.runner.wait_result().success)
        self.runner.reco.assert_not_called()
        self.assertEqual(self.now, 0)

    def test_no_result_uses_one_three_second_deadline(self):
        self.runner.image = Mock(return_value="frame")
        self.runner.hit = Mock(return_value=False)
        self.runner.reco = Mock(return_value=SimpleNamespace(hit=False))
        result = self.runner.wait_result()
        self.assertFalse(result.success)
        self.assertEqual(result.message, "未识别到兑换结果")
        self.assertAlmostEqual(self.now, 3)
        self.assertGreater(self.runner.image.call_count, 1)

    def test_cancellation_and_failed_capture_never_use_cached_frame(self):
        self.context.tasker.stopping = True
        with self.assertRaises(redeem.RedemptionError):
            self.runner.image()
        self.context.tasker.controller.post_screencap.assert_not_called()
        self.context.tasker.stopping = False
        self.context.tasker.controller.post_screencap.return_value.wait.return_value.succeeded = False
        with self.assertRaises(redeem.RedemptionError):
            self.runner.image()

    def test_missing_button_times_out_without_clicking(self):
        self.runner.image = Mock(return_value="frame")
        self.context.run_recognition.return_value = SimpleNamespace(hit=False)
        with self.assertRaises(redeem.RedemptionError):
            self.runner.button("兑换码领取")
        self.context.run_action_direct.assert_not_called()
        self.assertAlmostEqual(self.now, 10)

    def test_clear_old_code_closes_input_before_opening_for_new_code(self):
        events = Mock()
        self.runner.wait = Mock(return_value=True)
        self.runner.button = events.button
        self.runner.task = events.task
        self.runner.needs_clear = True
        self.runner.clear_previous()
        self.assertEqual(events.mock_calls, [
            call.button("兑换码输入区域"), call.task("兑换码全选旧码"), call.button("兑换码输入确定")])
        self.assertIs(self.runner.wait.call_args.args[0].__func__, redeem.RedeemRunner.empty_dialog)
        self.assertFalse(self.runner.needs_clear)

    def test_submission_has_separate_dismiss_and_claim_steps(self):
        events = Mock()
        self.runner.wait = events.wait
        events.wait.return_value = (300, 620, 120, 45)
        self.runner.button = events.button
        self.runner.task = events.task
        self.runner.click = events.click
        self.runner.wait_result = events.result
        events.result.return_value = redeem.RedemptionResult(False, "兑换码不存在!")
        self.runner.redeem_one("A+B")
        calls = events.mock_calls
        input_call = next(c for c in calls if c[0] == "task")
        self.assertEqual(input_call.args[0], "兑换码输入文本")
        self.assertEqual(input_call.args[1]["兑换码输入文本"]["action"]["param"]["custom_action_param"], {"input_text": "A+B"})
        dismiss = calls.index(call.click((300, 620, 120, 45)))
        self.assertEqual(calls[dismiss + 1][0], "wait")
        self.assertEqual(calls[dismiss + 2], call.button("兑换码领取"))
        self.assertEqual(calls[dismiss + 3], call.result())
        self.assertTrue(self.runner.needs_clear)

    def test_failed_clear_stops_before_inputting_next_code(self):
        self.runner.needs_clear = True
        self.runner.clear_previous = Mock(side_effect=redeem.RedemptionError("旧码未清空"))
        self.runner.task = Mock()
        with self.assertRaises(redeem.RedemptionError):
            self.runner.redeem_one("B")
        self.runner.task.assert_not_called()

    def test_closing_returns_through_settings_card_and_home(self):
        events = Mock()
        self.runner.wait = events.wait
        self.runner.click = events.click
        self.runner.close()
        self.assertEqual([c[0] for c in events.mock_calls], ["wait", "click", "wait", "click", "wait", "click", "wait"])
        self.assertEqual(events.click.call_args_list, [call([343, 85, 37, 20])] * 3)

    def test_custom_input_is_a_task_and_failure_propagates(self):
        self.context.run_task.return_value = SimpleNamespace(status=SimpleNamespace(succeeded=False))
        with self.assertRaises(redeem.RedemptionError):
            self.runner.task("兑换码输入文本", {"input": "example"})
        self.context.run_task.assert_called_once_with("兑换码输入文本", pipeline_override={"input": "example"})
        self.context.run_action.assert_not_called()


if __name__ == "__main__":
    unittest.main()
