"""橱窗方案、循环终止与售卖动作的离线行为测试。"""

import json
import unittest
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools.test_dreamweaver_space import ROOT, dream
from tools.validate_schema import load_jsonc


class ShopTests(unittest.TestCase):
    def runner(self):
        context = Mock()
        context.tasker.stopping = False
        return dream.DreamRunner(context)

    def result(self, items):
        return SimpleNamespace(hit=bool(items), filtered_results=items, best_result=None)

    def test_calendar_ranges_include_sale_day(self):
        expected = (
            [(1, 2, "皮革"), (3, 11, "红色系"), (12, 16, "蓝色系"),
             (17, 19, "黄色系"), (20, 25, "棕色系"), (26, 28, "绿色系"),
             (29, 31, "皮革")]
        )
        for first, last, label in expected:
            for day in range(first, last + 1):
                with self.subTest(day=day):
                    now = datetime(2026, 10, day, 5, tzinfo=dream.BEIJING)
                    self.assertEqual(dream.calendar_label(now), label)

    def test_five_am_boundary_and_short_month(self):
        for now, label in (
            (datetime(2026, 10, 12, 4, 59, tzinfo=dream.BEIJING), "红色系"),
            (datetime(2026, 10, 12, 5, tzinfo=dream.BEIJING), "蓝色系"),
            (datetime(2026, 3, 1, 4, 59, tzinfo=dream.BEIJING), "绿色系"),
            (datetime(2026, 3, 1, 5, tzinfo=dream.BEIJING), "皮革"),
            (datetime(2026, 10, 11, 21, tzinfo=timezone.utc), "蓝色系"),
        ):
            self.assertEqual(dream.calendar_label(now), label)

    def test_trend_parsing_does_not_guess_unknown_tags(self):
        self.assertEqual(dream.parse_trend_label("今日潮流：黄色系", "今日潮流"), "黄色系")
        self.assertEqual(dream.parse_trend_label("明日预告：皮革", "明日预告"), "皮革")
        self.assertEqual(dream.parse_trend_label("明日预告红色系", "明日预告"), "红色系")
        for text in ("黄色系", "今日潮流：紫色系", "今日潮流：红色系蓝色系"):
            self.assertIsNone(dream.parse_trend_label(text, "今日潮流"))
        self.assertEqual(dream.parse_uplift("中型潮流，价格上涨20%~"), 20)
        self.assertEqual(dream.parse_uplift("价格上涨５０％"), 50)
        self.assertIsNone(dream.parse_uplift("价格上涨50%价格上涨20%"))
        self.assertIsNone(dream.parse_uplift("50"))

    def test_each_strategy_selects_only_required_category(self):
        now = datetime(2026, 10, 3, 5, tzinfo=dream.BEIJING)
        trend = dream.Trend("黄色系", "皮革", 20)
        self.assertEqual(dream.production_tags("stock", trend, now), (None, None, "红色系"))
        self.assertEqual(dream.production_tags("current", trend), (None, None, "黄色系"))
        self.assertEqual(dream.production_tags("tomorrow", trend), (None, "皮革", None))
        self.assertEqual(
            dream.production_tags("tomorrow", dream.Trend("黄色系", "皮革", 50)),
            (None, None, "黄色系"),
        )
        with self.assertRaises(dream.DreamError):
            dream.production_tags("unknown", trend)

    def simulate_rounds(self, rounds, rest_after, initial_busy=False, sizes=(5,)):
        runner = self.runner()
        states = {tile: ("空闲中", (1, 1, 10, 10)) for tile in rest_after}
        states[(2, 2)] = ("休息中", (1, 1, 10, 10))
        if initial_busy:
            states[next(iter(rest_after))] = ("生产中", (1, 1, 10, 10))
        counts = {tile: 0 for tile in rest_after}
        runner.image = Mock(return_value=object())
        runner.management_ready = Mock(return_value=True)
        runner.tile_states = Mock(side_effect=lambda image, expected=None, **kwargs: dict(states))

        def wait(predicate, description, **kwargs):
            value = predicate(object())
            if value is None or value is False:
                raise dream.DreamError(description)
            return value

        def finish(tiles, use_stone, deadline):
            for tile in tiles:
                batch = runner.pending.pop(tile, None)
                if batch:
                    batch.order.harvested = batch.quantity
                    counts[tile] += 1
                state = "休息中" if tile == (2, 2) or counts[tile] >= rest_after[tile] else "空闲中"
                states[tile] = (state, (1, 1, 10, 10))

        def schedule(tile, batch, **kwargs):
            self.assertEqual(states[tile][0], "空闲中")
            states[tile] = ("生产中", (1, 1, 10, 10))
            quantity = sizes[min(counts[tile], len(sizes) - 1)]
            batch.scheduled = quantity
            runner.pending[tile] = dream.ProductionBatch(batch, quantity)

        runner.wait = Mock(side_effect=wait)
        runner.harvest = Mock()
        runner.finish_tiles = Mock(side_effect=finish)
        runner.schedule = Mock(side_effect=schedule)
        completed = runner.shop_produce((None, None, "黄色系"), rounds, False)
        return runner, counts, completed

    def test_zero_continues_remaining_slots_until_all_rest(self):
        runner, counts, completed = self.simulate_rounds(0, {(0, 0): 1, (0, 1): 3})
        self.assertEqual(counts, {(0, 0): 1, (0, 1): 3})
        self.assertEqual(completed, 3)
        self.assertEqual(runner.schedule.call_count, 4)
        self.assertEqual(runner.pending, {})

    def test_positive_rounds_stop_at_configured_limit(self):
        runner, counts, completed = self.simulate_rounds(2, {(0, 0): 8, (2, 3): 8})
        self.assertEqual(completed, 2)
        self.assertEqual(list(counts.values()), [2, 2])
        self.assertEqual(runner.schedule.call_count, 4)

    def test_reduced_shop_batches_count_actual_yield_and_complete_rounds(self):
        runner, counts, completed = self.simulate_rounds(2, {(0, 0): 8}, sizes=(4, 3))
        self.assertEqual(completed, 2)
        batches = [call.args[1] for call in runner.schedule.call_args_list]
        self.assertEqual([batch.harvested for batch in batches], [4, 3])
        self.assertEqual(counts, {(0, 0): 2})

    def test_positive_rounds_end_early_when_all_rest(self):
        _, counts, completed = self.simulate_rounds(9, {(0, 0): 1, (0, 1): 2})
        self.assertEqual(completed, 2)
        self.assertEqual(list(counts.values()), [1, 2])

    def test_initial_busy_batch_is_not_counted(self):
        runner, counts, completed = self.simulate_rounds(1, {(0, 0): 8}, initial_busy=True)
        self.assertEqual(completed, 1)
        self.assertEqual(counts[(0, 0)], 1)
        self.assertEqual(runner.finish_tiles.call_count, 2)
        self.assertEqual(runner.schedule.call_count, 1)

    def test_already_all_rest_finishes_without_scheduling(self):
        runner, _, completed = self.simulate_rounds(0, {})
        self.assertEqual(completed, 0)
        runner.schedule.assert_not_called()

    def test_recorded_slot_disappearing_is_not_completion(self):
        runner = self.runner()
        runner.management_ready = Mock(return_value=True)
        runner.tile_states = Mock(return_value={(0, 0): ("休息中", (1, 1, 10, 10))})
        self.assertIsNone(runner.ready_tiles(object(), {(0, 0), (0, 1)}))

    def test_rest_recognition_covers_lower_rows(self):
        runner = self.runner()
        rest = SimpleNamespace(text="休息中", box=(1080, 530, 80, 25))
        def reco(node, image, roi):
            x, y, w, h = roi
            return self.result([rest] if node == "织梦休息工位"
                               and x <= rest.box[0] < x + w and y <= rest.box[1] < y + h else [])
        runner.reco = Mock(side_effect=reco)
        self.assertEqual(runner.tile_states(object())[(2, 3)][0], "休息中")

    def test_harvest_can_end_in_rest_state(self):
        runner = self.runner()
        batch = dream.SaleBatch((None, None, "黄色系"), scheduled=5)
        runner.pending[(0, 0)] = dream.ProductionBatch(batch, 5)
        runner.wait = Mock(return_value=True)
        runner.image = Mock(return_value=object())
        runner.tile_states = Mock(side_effect=[{(0, 0): ("可收获", (1, 1, 10, 10))}, {}])
        runner.wait_tile = Mock(return_value=(1, 1, 10, 10))
        runner.click = Mock()
        runner.harvest()
        self.assertEqual(batch.harvested, 5)
        self.assertEqual(runner.wait_tile.call_args.args, ((0, 0), ("空闲中", "休息中")))

    def test_natural_round_deadline_is_not_extended(self):
        runner = self.runner()
        runner.harvest = Mock()
        runner.wait = Mock(side_effect=[{(0, 0): ("生产中", (1, 1, 10, 10))}, dream.DreamError("超时")])
        with self.assertRaises(dream.DreamError):
            runner.finish_tiles(((0, 0),), False, 610)
        self.assertTrue(all(call.kwargs["deadline"] == 610 for call in runner.wait.call_args_list))

    def test_stone_round_waits_for_harvest_before_finishing(self):
        runner = self.runner()
        runner.harvest = Mock()
        runner.accelerate = Mock(return_value={(0, 0)})
        runner.wait = Mock(side_effect=[
            {(0, 0): ("生产中", (1, 1, 10, 10))},
            {(0, 0): ("生产中", (1, 1, 10, 10))},
            {(0, 0): ("休息中", (1, 1, 10, 10))},
        ])
        runner.finish_tiles(((0, 0),), True, 610)
        runner.accelerate.assert_called_once_with((0, 0), 610)

    def prepare_shop(self, uplift):
        runner = self.runner()
        for name in ("enter_management", "enter_shop", "button", "wait", "shop_produce", "sell_products"):
            setattr(runner, name, Mock())
        runner.wait_trend = Mock(side_effect=[dream.Trend("黄色系", "红色系", 20),
                                            dream.Trend("黄色系", "红色系", uplift)])
        return runner

    def test_production_leaves_shop_and_finishes_before_sale(self):
        runner = self.prepare_shop(50)
        trace = Mock()
        for name in ("enter_shop", "button", "shop_produce", "sell_products"):
            trace.attach_mock(getattr(runner, name), name)
        runner.shop("current", 2, True)
        self.assertEqual([call[0] for call in trace.method_calls],
                         ["enter_shop", "button", "shop_produce", "enter_shop", "sell_products"])
        runner.button.assert_called_once_with("织梦橱窗返回按钮")
        runner.shop_produce.assert_called_once_with((None, None, "黄色系"), 2, True)

    def test_only_current_mode_sells_below_fifty_percent(self):
        for mode in dream.SALE_MODES:
            runner = self.prepare_shop(20)
            runner.shop(mode, 1, False)
            self.assertEqual(runner.sell_products.call_count, int(mode == "current"))

    def test_fresh_percentage_after_production_controls_sale(self):
        runner = self.prepare_shop(50)
        runner.shop("tomorrow", 1, False)
        runner.shop_produce.assert_called_once_with((None, None, "红色系"), 1, False)
        runner.sell_products.assert_called_once()

    def test_production_failure_never_enters_sale(self):
        runner = self.prepare_shop(50)
        runner.shop_produce.side_effect = dream.DreamError("未收获")
        with self.assertRaises(dream.DreamError):
            runner.shop("current", 1, False)
        runner.sell_products.assert_not_called()
        runner.enter_shop.assert_called_once()

    def test_invalid_configuration_does_not_navigate_or_click(self):
        for mode, rounds in (("unknown", 1), ("stock", -1), ("stock", True), ("stock", "2")):
            runner = self.prepare_shop(50)
            with self.assertRaises(dream.DreamError):
                runner.shop(mode, rounds, False)
            runner.enter_management.assert_not_called()

    def test_popup_background_is_not_ready_shop(self):
        runner = self.runner()
        runner.hit = Mock(return_value=True)
        self.assertIsNone(runner.shop_frame(object()))
        runner.hit.assert_called_once_with("织梦产品售卖页面", unittest.mock.ANY)

    def test_single_confirmation_does_not_sleep_after_ready_frame(self):
        runner = self.runner()
        frame = object()
        runner.image = Mock(return_value=frame)
        with patch.object(dream.time, "sleep") as sleep:
            self.assertIs(runner.wait(lambda image: image, "列表", confirmations=1), frame)
        sleep.assert_not_called()

    def test_each_sale_rechecks_trend_up_and_uses_approved_offset(self):
        runner = self.runner()
        frame = SimpleNamespace(shape=(720, 1280, 3))
        max_box, confirm_box = (716, 393, 72, 39), (561, 492, 152, 46)
        runner.wait = Mock(side_effect=[frame, max_box, confirm_box,
                                      frame, max_box, confirm_box, frame])
        items = [SimpleNamespace(box=(828, 103, 102, 35)),
                 SimpleNamespace(box=(827, 283, 105, 39))]
        runner.reco = Mock(side_effect=[self.result(items), self.result(items[:1]), self.result([])])
        runner.click = Mock()
        runner.sell_products()
        self.assertEqual([call.args[0] for call in runner.click.call_args_list],
                         [(949, 190, 1, 1), max_box, confirm_box,
                          (949, 190, 1, 1), max_box, confirm_box])
        self.assertEqual(runner.reco.call_count, 3)
        self.assertTrue(all(call.kwargs["confirmations"] == 1 for call in runner.wait.call_args_list))

    def test_sale_return_timeout_does_not_repeat_confirm_click(self):
        runner = self.runner()
        frame = SimpleNamespace(shape=(720, 1280, 3))
        max_box, confirm_box = (716, 393, 72, 39), (561, 492, 152, 46)
        runner.wait = Mock(side_effect=[frame, max_box, confirm_box, dream.DreamError("未返回列表")])
        runner.reco = Mock(return_value=self.result([SimpleNamespace(box=(828, 103, 102, 35))]))
        runner.click = Mock()
        with self.assertRaises(dream.DreamError):
            runner.sell_products()
        self.assertEqual(sum(call.args[0] == confirm_box for call in runner.click.call_args_list), 1)

    def test_selected_tasks_run_in_order_and_stone_options_are_independent(self):
        context = Mock()
        flags = {"织梦选择今日签到", "织梦选择经营", "织梦选择订单驿站",
                 "织梦选择售卖橱窗", "织梦选择能量石加速", "织梦售卖攒明日"}
        context.get_node_data.side_effect = lambda node: {"enabled": node in flags}
        argv = SimpleNamespace(custom_action_param=json.dumps({"sale_rounds": 3}))
        with patch.object(dream, "DreamRunner") as factory:
            self.assertTrue(dream.DreamweaverSpaceAction().run(context, argv))
            runner = factory.return_value
            self.assertEqual([call[0] for call in runner.method_calls], ["sign_in", "orders", "shop"])
            runner.orders.assert_called_once_with(True)
            runner.shop.assert_called_once_with("tomorrow", 3, False, after_orders=True)

    def test_options_reference_real_nodes_and_keep_business_default_zero(self):
        interface = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
        pipeline = load_jsonc(ROOT / "assets/resource/pipeline/dreamweaver_space.json")
        options = interface["option"]
        for name in ("织梦售卖方案", "织梦售卖工位循环次数", "织梦售卖生产完成方式"):
            option = options[name]
            self.assertNotIn("default_case", option)
            overrides = [option] if option["type"] == "input" else option["cases"]
            for item in overrides:
                self.assertTrue(set(item["pipeline_override"]).issubset(pipeline))
        self.assertEqual(options["织梦售卖工位循环次数"]["inputs"][0]["default"], "0")
        self.assertEqual(pipeline["织梦空间执行子任务"]["action"]["param"]["custom_action_param"]["sale_rounds"], 0)


if __name__ == "__main__":
    unittest.main()
