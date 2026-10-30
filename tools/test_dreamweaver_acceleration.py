"""加速漏点、弹窗消失、动画误报和有限等待的离线回归。"""

import importlib.util
import json
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools.test_dreamweaver_space import ROOT, dream, load_jsonc


class AccelerationTests(unittest.TestCase):
    @contextmanager
    def clock(self):
        now = [10.0]
        with (patch.object(dream.time, "monotonic", side_effect=lambda: now[0]),
              patch.object(dream.time, "sleep", side_effect=lambda dt: now.__setitem__(0, now[0] + dt))):
            yield now

    def runner(self):
        context = Mock()
        context.tasker.stopping = False
        context.run_task.return_value = SimpleNamespace(status=SimpleNamespace(succeeded=True))
        context.get_node_data.return_value = load_jsonc(
            ROOT / "assets/resource/pipeline/dreamweaver_space.json"
        )["织梦立即完成按钮"]
        runner = dream.DreamRunner(context)
        runner.image = Mock(return_value=object())
        runner.hit = Mock(return_value=True)
        runner.box = Mock(return_value=(569, 493, 140, 41))
        runner.text = Mock(return_value="消耗1颗（拥有3颗）")
        runner.management_ready = Mock(return_value=True)
        runner.tile_state = Mock(return_value=None)
        return runner

    def test_real_change_action_uses_new_ocr_centers_after_a_missed_click(self):
        spec = importlib.util.spec_from_file_location(
            "acceleration_click_under_test", Path(dream.__file__).with_name("click_and_wait_change.py")
        )
        module = importlib.util.module_from_spec(spec)
        with patch.object(dream.AgentServer, "register_custom_action", return_value=True):
            spec.loader.exec_module(module)
        action = module.ClickAndWaitChangeAction()
        action._screencap = Mock(return_value=object())
        # 第一次只是动画变化，仍留在弹窗，必须重新确认后才重试。
        action._image_changed = Mock(return_value=(True, 0.5))
        runner = self.runner()
        controller = runner.context.tasker.controller
        runner.hit.side_effect = lambda *args: controller.post_click.call_count < 2
        runner.tile_state.return_value = ("可收获", (800, 150, 90, 30))
        node = load_jsonc(ROOT / "assets/resource/pipeline/dreamweaver_space.json")["织梦立即完成按钮"]
        self.assertEqual(node["action"]["param"]["custom_action"], "click_and_wait_change")
        self.assertFalse(node.get("next"))
        self.assertFalse(node.get("on_error"))
        boxes = iter(((569, 493, 140, 41), (579, 503, 140, 41)))

        def run_task(entry, override):
            self.assertEqual(entry, "织梦立即完成按钮")
            params = override[entry]["action"]["param"]["custom_action_param"]
            self.assertEqual(params["compare_roi"], [450, 100, 380, 470])
            self.assertEqual(params["interval"], 100)
            self.assertEqual(params["max_retries"], 1)
            self.assertTrue(params["fail_on_timeout"])
            box = next(boxes)
            argv = SimpleNamespace(custom_action_param=json.dumps(params),
                                   box=SimpleNamespace(**dict(zip(("x", "y", "w", "h"), box))))
            return SimpleNamespace(status=SimpleNamespace(succeeded=action.run(runner.context, argv)))

        runner.context.run_task.side_effect = run_task
        with self.clock():
            runner.complete_acceleration((0, 1), 40)
        self.assertEqual([c.args for c in controller.post_click.call_args_list], [(639, 513), (649, 523)])
        self.assertGreaterEqual(runner.box.call_count, 2)
        runner.context.run_action_direct.assert_not_called()
        runner.tile_state.assert_called_once_with((0, 1), runner.image.return_value, "可收获")

    def test_three_unsuccessful_attempts_never_become_completion(self):
        for changed in (True, False):
            with self.subTest(changed=changed), self.clock() as now:
                runner = self.runner()
                order = dream.Order(dream.Card(1, (100, 100, 70, 50)), 5, (None, "皮革", None))
                runner.pending[(0, 0)] = dream.ProductionBatch(order, 5)
                runner.context.run_task.return_value.status.succeeded = changed
                with self.assertRaisesRegex(dream.DreamError, "已尝试 3/3 次"):
                    runner.complete_acceleration((0, 0), 40)
                self.assertAlmostEqual(now[0], 40)
                self.assertEqual(runner.context.run_task.call_count, 3)
                self.assertFalse(runner._stone_exhausted)
                self.assertIn((0, 0), runner.pending)
                self.assertEqual(order.harvested, 0)

    def test_one_button_observation_can_click_without_waiting_for_a_second_hit(self):
        runner = self.runner()
        runner.hit.side_effect = [True, False]
        runner.box.side_effect = [(569, 493, 140, 41), None]
        runner.tile_state.return_value = ("可收获", (650, 150, 90, 30))
        with self.clock():
            runner.complete_acceleration((0, 0), 40)
        runner.context.run_task.assert_called_once()
        runner.box.assert_called_once()

    def test_missing_button_or_unknown_page_never_clicks_and_times_out(self):
        for popup in (True, False):
            with self.subTest(popup=popup), self.clock() as now:
                runner = self.runner()
                runner.hit.return_value = popup
                runner.box.return_value = None
                runner.management_ready.return_value = False
                with self.assertRaisesRegex(dream.DreamError, "已尝试 0/3 次"):
                    runner.complete_acceleration((0, 0), 40)
                runner.context.run_task.assert_not_called()
                self.assertAlmostEqual(now[0], 40)

    def test_closed_popup_never_repeats_click_even_when_tile_is_not_ready(self):
        runner = self.runner()
        runner.hit.side_effect = lambda *args: not runner.context.run_task.called
        with self.clock(), self.assertRaises(dream.DreamError):
            runner.complete_acceleration((0, 0), 40)
        runner.context.run_task.assert_called_once()

    def test_failed_visual_check_can_still_finish_when_actual_tile_is_ready(self):
        runner = self.runner()
        runner.context.run_task.return_value.status.succeeded = False
        runner.hit.side_effect = lambda *args: not runner.context.run_task.called
        runner.tile_state.return_value = ("可收获", (650, 150, 90, 30))
        with self.clock():
            runner.complete_acceleration((0, 0), 40)
        runner.context.run_task.assert_called_once()

    def test_cancellation_or_elapsed_deadline_prevents_click(self):
        runner = self.runner()
        runner.context.tasker.stopping = True
        with self.clock(), self.assertRaisesRegex(dream.DreamError, "任务已取消"):
            runner.complete_acceleration((0, 0), 40)
        runner.context.run_task.assert_not_called()
        with self.clock(), self.assertRaises(dream.DreamError):
            runner.complete_acceleration((0, 0), 10)
        runner.context.run_task.assert_not_called()

    def test_task_start_failure_does_not_retry_or_mark_stones_exhausted(self):
        runner = self.runner()
        runner.context.run_task.return_value = None
        with self.clock(), self.assertRaisesRegex(dream.DreamError, "未能启动"):
            runner.complete_acceleration((0, 0), 40)
        runner.context.run_task.assert_called_once()
        self.assertFalse(runner._stone_exhausted)

    def test_acceleration_respects_earlier_production_deadline(self):
        runner = self.runner()
        runner.click = Mock()
        runner.wait_tile = Mock(return_value=(650, 250, 90, 30))
        runner.complete_acceleration = Mock()
        runner.harvest = Mock()
        with self.clock():
            runner.accelerate((0, 0), 11)
        runner.complete_acceleration.assert_called_once_with((0, 0), 11)
        runner.harvest.assert_called_once_with(deadline=11, tiles=((0, 0),))

    def test_stone_parser_reads_owned_count_instead_of_cost(self):
        for text, expected in (("消耗1颗（拥有0颗）", 0), ("消耗1颗(拥有12颗)", 12),
                               ("消耗１颗（拥有 ０ 颗）", 0), ("拥有：3颗", 3)):
            with self.subTest(text=text):
                self.assertEqual(dream.parse_acceleration_stones(text), expected)
        for text in ("", "消耗1颗", "拥有O颗", "拥有-1颗", "拥有0颗拥有1颗", "拥有0/5颗"):
            with self.subTest(text=text):
                self.assertIsNone(dream.parse_acceleration_stones(text))

    def test_zero_stones_close_popup_without_spending_or_counting_harvest(self):
        runner = self.runner()
        runner.text.return_value = "消耗1颗（拥有0颗）"
        runner.context.run_action.return_value = SimpleNamespace(success=True)
        order = dream.Order(dream.Card(1, (100, 100, 70, 50)), 5, (None, "皮革", None))
        runner.pending[(0, 0)] = dream.ProductionBatch(order, 5)
        with self.clock(), patch.object(dream.logger, "info") as log:
            self.assertFalse(runner.complete_acceleration((0, 0), 40))
        self.assertTrue(runner._stone_exhausted)
        runner.context.run_action.assert_called_once_with("点击esc")
        runner.context.run_task.assert_not_called()
        runner.box.assert_not_called()
        self.assertIn((0, 0), runner.pending)
        self.assertEqual(order.harvested, 0)
        log.assert_called_once_with("能量石用完了，转为自然等待")

    def test_zero_stones_do_not_claim_fallback_until_popup_closes(self):
        for failure in ("action", "still_open", "cancel"):
            with self.subTest(failure=failure), self.clock():
                runner = self.runner()
                runner.text.return_value = "消耗1颗（拥有0颗）"
                runner.context.run_action.return_value = SimpleNamespace(success=failure != "action")
                runner.management_ready.return_value = failure != "still_open"
                runner.context.tasker.stopping = failure == "cancel"
                with self.assertRaises(dream.DreamError):
                    runner.complete_acceleration((0, 0), 40)
                self.assertFalse(runner._stone_exhausted)
                runner.context.run_task.assert_not_called()
                if failure == "cancel":
                    runner.context.run_action.assert_not_called()

    def test_unreadable_stones_never_click_or_mark_them_exhausted(self):
        runner = self.runner()
        runner.text.return_value = "消耗1颗（拥有O颗）"
        with self.clock(), self.assertRaisesRegex(dream.DreamError, "已尝试 0/3 次"):
            runner.complete_acceleration((0, 0), 40)
        runner.context.run_task.assert_not_called()
        runner.context.run_action.assert_not_called()
        self.assertFalse(runner._stone_exhausted)

    def test_transient_unreadable_count_waits_then_clicks_after_positive_count(self):
        runner = self.runner()
        runner.text.side_effect = ["", "消耗1颗（拥有1颗）"]
        runner.hit.side_effect = lambda *args: not runner.context.run_task.called
        runner.tile_state.return_value = ("可收获", (650, 150, 90, 30))
        with self.clock():
            self.assertTrue(runner.complete_acceleration((0, 0), 40))
        runner.context.run_task.assert_called_once()
        runner.context.run_action.assert_not_called()
        self.assertFalse(runner._stone_exhausted)

    def test_stones_are_rechecked_before_repeating_a_click(self):
        runner = self.runner()
        runner.text.side_effect = ["消耗1颗（拥有1颗）", "消耗1颗（拥有0颗）"]
        runner.context.run_task.return_value.status.succeeded = False
        runner.context.run_action.return_value = SimpleNamespace(success=True)
        with self.clock():
            self.assertFalse(runner.complete_acceleration((0, 0), 40))
        runner.context.run_task.assert_called_once()
        runner.context.run_action.assert_called_once_with("点击esc")
        self.assertTrue(runner._stone_exhausted)

    def test_depleted_popup_returns_to_natural_wait_with_original_deadline(self):
        runner = self.runner()
        runner.text.return_value = "消耗1颗（拥有0颗）"
        runner.context.run_action.return_value = SimpleNamespace(success=True)
        tile, box = (0, 0), (650, 250, 90, 30)
        runner.wait_tile = Mock(return_value=box)
        runner.click = Mock()
        states = iter(({tile: ("生产中", box)}, {tile: ("生产中", box)},
                       {tile: ("可收获", box)}, {tile: ("空闲中", box)}))
        runner.tile_states = Mock(side_effect=lambda *args, **kwargs: next(states))
        runner.harvest = Mock(return_value={tile})
        runner.wait = Mock(side_effect=lambda predicate, label, **kwargs: predicate(object()))
        with self.clock():
            runner.finish_tiles((tile,), True, 610)
        runner.harvest.assert_called_once_with(deadline=610, tiles=[tile])
        self.assertEqual(runner.tile_states.call_args_list[2].args[1], "可收获")
        runner.context.run_task.assert_not_called()
        self.assertTrue(runner._stone_exhausted)


if __name__ == "__main__":
    unittest.main()
