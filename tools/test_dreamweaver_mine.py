"""每日能量晶矿的领取、弹窗关闭与多选衔接测试。"""

import itertools
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools.test_dreamweaver_space import ROOT, dream
from tools.validate_schema import load_jsonc


class MineTests(unittest.TestCase):
    def runner(self, count=6):
        context = Mock()
        context.tasker.stopping = False
        runner = dream.DreamRunner(context)
        runner.enter_management = Mock()
        runner.management_entry = Mock()
        runner.button = Mock()
        runner.click = Mock()
        runner.wait = Mock(side_effect=[(count, 6)] + ([True, True] if count else []) + [True])
        return runner

    def test_storage_field_accepts_zero_and_rejects_unknown_values(self):
        for text, expected in (
            ("能量石储存数量：6/6", (6, 6)),
            ("能量石储存数量:０ / ６", (0, 6)),
            ("能量石储存数量：2/9", (2, 9)),
        ):
            self.assertEqual(dream.parse_mine_storage(text), expected)
        for text in ("6/6", "库存：6/6", "能量石储存数量：7/6", "能量石储存数量：0/0"):
            self.assertIsNone(dream.parse_mine_storage(text))

    def test_reward_background_is_not_mine_ready(self):
        runner = self.runner()
        runner.hit = Mock(return_value=True)
        runner.text = Mock()
        self.assertIsNone(runner.mine_state(object()))
        runner.text.assert_not_called()

    def test_mine_page_requires_both_title_and_valid_storage(self):
        runner = self.runner()
        frame = object()
        runner.hit = Mock(side_effect=[False, True])
        runner.text = Mock(return_value="能量石储存数量：0/6")
        self.assertEqual(runner.mine_state(frame), (0, 6))
        runner.hit = Mock(side_effect=[False, False])
        self.assertIsNone(runner.mine_state(frame))

    def test_claim_once_and_close_reward_before_closing_mine(self):
        runner = self.runner()
        trace = Mock()
        for method in ("management_entry", "button", "click", "wait"):
            trace.attach_mock(getattr(runner, method), method)
        runner.mine()
        self.assertEqual([call[0] for call in trace.method_calls],
                         ["management_entry", "wait", "button", "wait", "click", "wait", "click", "wait"])
        runner.management_entry.assert_called_once_with("织梦能量晶矿入口")
        self.assertEqual([call.args[0] for call in runner.button.call_args_list],
                         ["织梦能量晶矿领取按钮"])
        self.assertEqual([call.args[0] for call in runner.click.call_args_list],
                         [[550, 625, 180, 55], [550, 625, 180, 55]])
        runner.mine_state = Mock(side_effect=[None, (1, 6), (0, 6)])
        zero_predicate = runner.wait.call_args_list[2].args[0]
        self.assertFalse(zero_predicate(object()))
        self.assertFalse(zero_predicate(object()))
        self.assertTrue(zero_predicate(object()))

    def test_empty_inventory_skips_grey_claim_button(self):
        runner = self.runner(0)
        runner.mine()
        runner.management_entry.assert_called_once_with("织梦能量晶矿入口")
        runner.button.assert_not_called()
        runner.click.assert_called_once_with([550, 625, 180, 55])

    def test_unknown_inventory_stops_without_claim_or_blank_click(self):
        runner = self.runner()
        runner.wait.side_effect = dream.DreamError("库存未识别")
        with self.assertRaises(dream.DreamError):
            runner.mine()
        runner.management_entry.assert_called_once_with("织梦能量晶矿入口")
        runner.button.assert_not_called()
        runner.click.assert_not_called()

    def test_reward_timeout_does_not_repeat_claim(self):
        runner = self.runner()
        runner.wait.side_effect = [(6, 6), dream.DreamError("奖励弹窗未出现")]
        with self.assertRaises(dream.DreamError):
            runner.mine()
        self.assertEqual(sum(call.args[0] == "织梦能量晶矿领取按钮"
                             for call in runner.button.call_args_list), 1)
        runner.click.assert_not_called()

    def test_zero_confirmation_timeout_does_not_close_unrecognized_page(self):
        runner = self.runner()
        runner.wait.side_effect = [(6, 6), True, dream.DreamError("库存未归零")]
        with self.assertRaises(dream.DreamError):
            runner.mine()
        runner.click.assert_called_once_with([550, 625, 180, 55])

    def test_mine_enters_management_before_opening_mine(self):
        runner = self.runner(0)
        trace = Mock()
        trace.attach_mock(runner.enter_management, "enter_management")
        trace.attach_mock(runner.management_entry, "management_entry")
        runner.mine()
        self.assertEqual([call[0] for call in trace.method_calls],
                         ["enter_management", "management_entry"])

    def test_all_eight_checkbox_combinations_run_selected_tasks_in_order(self):
        names = ("织梦选择每日能量晶矿", "织梦选择订单驿站", "织梦选择售卖橱窗")
        methods = ("mine", "orders", "shop")
        for selected in itertools.product((False, True), repeat=3):
            with self.subTest(selected=selected):
                flags = {name for name, yes in zip(names, selected) if yes} | {"织梦选择经营"}
                context = Mock()
                context.get_node_data.side_effect = lambda node: {"enabled": node in flags}
                with patch.object(dream, "DreamRunner") as factory:
                    self.assertTrue(dream.DreamweaverSpaceAction().run(context, SimpleNamespace()))
                    runner = factory.return_value
                    self.assertEqual([call[0] for call in runner.method_calls],
                                     [method for method, yes in zip(methods, selected) if yes])
                    if selected[0]:
                        runner.mine.assert_called_once_with()
                    if selected[2]:
                        self.assertEqual(runner.shop.call_args.kwargs["after_orders"], selected[1])

    def test_mine_requires_parent_management_selection(self):
        context = Mock()
        context.get_node_data.side_effect = lambda node: {"enabled": node == "织梦选择每日能量晶矿"}
        with patch.object(dream, "DreamRunner") as factory:
            self.assertTrue(dream.DreamweaverSpaceAction().run(context, SimpleNamespace()))
            factory.return_value.mine.assert_not_called()

    def test_interface_keeps_three_choices_as_checkbox_and_valid_override(self):
        interface = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
        pipeline = load_jsonc(ROOT / "assets/resource/pipeline/dreamweaver_space.json")
        option = interface["option"]["织梦经营子任务"]
        self.assertEqual(option["type"], "checkbox")
        self.assertEqual([case["name"] for case in option["cases"]],
                         ["每日能量晶矿", "订单驿站", "售卖橱窗"])
        self.assertEqual(option["cases"][0]["pipeline_override"],
                         {"织梦选择每日能量晶矿": {"enabled": True}})
        self.assertFalse(pipeline["织梦选择每日能量晶矿"]["enabled"])


class ManagementEntryTests(unittest.TestCase):
    def runner(self):
        context = Mock()
        context.tasker.stopping = False
        return dream.DreamRunner(context)

    def test_management_ready_uses_unobscured_anchors(self):
        runner = self.runner()
        frame = object()
        anchors = {"织梦经营销售额", "织梦经营仓库"}
        runner.hit = Mock(side_effect=lambda node, image: node in anchors)
        self.assertTrue(runner.management_ready(frame))
        self.assertFalse(any("入口" in call.args[0] for call in runner.hit.call_args_list))
        anchors.remove("织梦经营仓库")
        self.assertFalse(runner.management_ready(frame))
        anchors.add("织梦经营仓库")
        anchors.add("织梦设计页面")
        self.assertFalse(runner.management_ready(frame))

    def test_entry_waits_for_management_before_action(self):
        runner = self.runner()
        trace = Mock()
        runner.wait = Mock(return_value=True)
        trace.attach_mock(runner.wait, "wait")
        trace.attach_mock(runner.context.run_action, "action")
        runner.management_entry("织梦经营订单入口")
        self.assertEqual([call[0] for call in trace.method_calls], ["wait", "action"])
        runner.context.run_action.assert_called_once_with("织梦经营订单入口")

    def test_unrecognized_page_and_cancellation_prevent_click(self):
        runner = self.runner()
        runner.wait = Mock(side_effect=dream.DreamError("页面未就绪"))
        with self.assertRaises(dream.DreamError):
            runner.management_entry("织梦能量晶矿入口")
        runner.context.run_action.assert_not_called()
        runner.wait.side_effect = None
        runner.context.tasker.stopping = True
        with self.assertRaises(dream.DreamError):
            runner.management_entry("织梦能量晶矿入口")
        runner.context.run_action.assert_not_called()

    def test_failed_entry_action_is_not_success(self):
        for result in (None, SimpleNamespace(success=False)):
            runner = self.runner()
            runner.wait = Mock()
            runner.context.run_action.return_value = result
            with self.assertRaises(dream.DreamError):
                runner.management_entry("织梦售卖橱窗入口")

    def test_next_subtask_reuses_management_page(self):
        for inside in (False, True):
            runner = self.runner()
            runner.management_ready = Mock(return_value=inside)
            runner.space_ready = Mock(return_value=True)
            runner.wait = Mock(side_effect=lambda predicate, label: predicate(object()))
            runner.button = Mock()
            runner.enter_management()
            if inside:
                runner.button.assert_not_called()
            else:
                runner.button.assert_called_once_with("织梦经营入口")
                self.assertEqual(runner.wait.call_count, 2)

    def test_fixed_entries_keep_destination_waits(self):
        for method, node, predicate in (
            ("enter_orders", "织梦经营订单入口", "map_ready"),
            ("enter_shop", "织梦售卖橱窗入口", "shop_frame"),
        ):
            runner = self.runner()
            runner.management_entry = Mock()
            runner.wait = Mock()
            getattr(runner, method)()
            runner.management_entry.assert_called_once_with(node)
            self.assertEqual(runner.wait.call_args.args[0], getattr(runner, predicate))

    def test_pipeline_entries_only_define_fixed_clicks(self):
        pipeline = load_jsonc(ROOT / "assets/resource/pipeline/dreamweaver_space.json")
        for node, point in (
            ("织梦能量晶矿入口", [540, 405, 1, 1]),
            ("织梦经营订单入口", [113, 393, 1, 1]),
            ("织梦售卖橱窗入口", [375, 221, 1, 1]),
        ):
            self.assertEqual(pipeline[node], {"action": {"type": "Click", "param": {"target": point}}})


if __name__ == "__main__":
    unittest.main()
