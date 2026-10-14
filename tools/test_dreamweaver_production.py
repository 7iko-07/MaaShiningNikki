"""整轮生产/收获，以及整块 ROI 误识别的离线回归。"""

import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

from tools.test_dreamweaver_space import ROOT, dream, load_jsonc


class ProductionTests(unittest.TestCase):
    def runner(self):
        context = Mock()
        context.tasker.stopping = False
        runner = dream.DreamRunner(context)
        runner.management_ready = Mock(return_value=True)
        runner.image = Mock(return_value=object())
        return runner

    def order(self, quantity=5):
        return dream.Order(dream.Card(1, (100, 100, 70, 50)), quantity, ("上装", None, None))

    def result(self, text, box):
        items = [SimpleNamespace(text=text, box=box)]
        return SimpleNamespace(hit=True, box=box, filtered_results=items, all_results=items)

    def test_submit_retries_with_fresh_box_even_after_pixel_change(self):
        # 执行真正的 click_and_wait_change，验证中心点击和画面变化后的页面复核。
        spec = importlib.util.spec_from_file_location(
            "production_click_under_test", Path(dream.__file__).with_name("click_and_wait_change.py")
        )
        module = importlib.util.module_from_spec(spec)
        with patch.object(dream.AgentServer, "register_custom_action", return_value=True):
            spec.loader.exec_module(module)
        action = module.ClickAndWaitChangeAction()
        action._screencap = Mock(return_value=object())
        action._image_changed = Mock(return_value=(True, 0.5))
        runner = self.runner()
        runner.management_ready = Mock(side_effect=[False, True])
        runner.hit = Mock(return_value=True)
        runner.reco = Mock(side_effect=[self.result("生4件", (351, 621, 110, 38)),
                                        self.result("产4件", (353, 619, 110, 38))])
        now = [10.0]
        node = load_jsonc(ROOT / "assets/resource/pipeline/dreamweaver_space.json")["织梦生产五件按钮"]
        self.assertEqual(node["recognition"]["type"], "OCR")
        self.assertEqual(node["action"]["param"]["custom_action"], "click_and_wait_change")
        self.assertFalse(node.get("next"))
        self.assertFalse(node.get("on_error"))
        def run_task(entry, override):
            self.assertEqual(entry, "织梦生产五件按钮")
            # 模拟这次节点 OCR 框；原生回调另由 check_dreamweaver_production_native 验证。
            box = (351, 621, 110, 38) if runner.reco.call_count == 1 else (353, 619, 110, 38)
            argv = SimpleNamespace(
                custom_action_param=json.dumps(override[entry]["action"]["param"]["custom_action_param"]),
                box=SimpleNamespace(**dict(zip(("x", "y", "w", "h"), box))),
            )
            return SimpleNamespace(status=SimpleNamespace(succeeded=action.run(runner.context, argv)))
        runner.context.run_task.side_effect = run_task
        with (patch.object(dream.time, "monotonic", side_effect=lambda: now[0]),
              patch.object(dream.time, "sleep", side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds))):
            self.assertEqual(runner.submit_production(deadline=40), 4)
        clicks = runner.context.tasker.controller.post_click.call_args_list
        self.assertEqual([call.args for call in clicks], [(406, 640), (408, 638)])
        self.assertEqual(runner.context.run_task.call_count, 2)
        runner.context.run_action_direct.assert_not_called()

    def test_submit_stops_clicking_when_button_or_design_page_is_missing(self):
        runner = self.runner()
        runner.management_ready = Mock(side_effect=[False, False, True])
        runner.hit = Mock(side_effect=[True, False, True])
        runner.reco = Mock(side_effect=[self.result("生产4件", (351, 621, 110, 38)), None])
        with patch.object(dream.time, "sleep"):
            runner.submit_production(deadline=dream.time.monotonic() + 30)
        runner.context.run_task.assert_called_once()
        self.assertEqual(runner.reco.call_count, 2)

    def test_submit_unchanged_page_retries_until_original_deadline(self):
        runner = self.runner()
        runner.management_ready = Mock(return_value=False)
        runner.hit = Mock(return_value=True)
        runner.reco = Mock(return_value=self.result("生产4件", (351, 621, 110, 38)))
        now = [10.0]
        def run_task(*args, **kwargs):
            now[0] += 1
            return SimpleNamespace(status=SimpleNamespace(succeeded=True))
        runner.context.run_task.side_effect = run_task
        with patch.object(dream.time, "monotonic", side_effect=lambda: now[0]):
            with self.assertRaisesRegex(dream.DreamError, "等待超时"):
                runner.submit_production(deadline=13)
        self.assertEqual(runner.context.run_task.call_count, 3)
        self.assertEqual(runner.pending, {})

    def test_submit_action_failure_stops_immediately(self):
        runner = self.runner()
        runner.hit = Mock(return_value=True)
        runner.reco = Mock(return_value=self.result("生产4件", (351, 621, 110, 38)))
        runner.context.run_task.return_value = SimpleNamespace(status=SimpleNamespace(succeeded=False))
        with self.assertRaisesRegex(dream.DreamError, "点击生产按钮失败"):
            runner.submit_production(deadline=dream.time.monotonic() + 30)
        runner.context.run_task.assert_called_once()

    def test_production_letters_only_confirm_the_target_lower_band(self):
        runner = self.runner()
        for text in ("生", "产", "生产", "生中"):
            with self.subTest(text=text):
                runner.reco = Mock(return_value=self.result(text, (809, 244, 30, 18)))
                self.assertEqual(runner.tile_state((0, 1), object(), "生产中")[0], "生产中")
                runner.reco.assert_called_once_with("织梦工位状态", ANY, [781, 232, 137, 68])
        runner.reco = Mock(return_value=self.result("中", (809, 244, 30, 18)))
        self.assertIsNone(runner.tile_state((0, 1), object(), "生产中"))
        runner.reco = Mock(return_value=self.result("生", (781, 232, 137, 68)))
        self.assertIsNone(runner.tile_state((0, 1), object(), "生产中"))
        self.assertEqual(runner.tile_state((0, 1), object(), "生产中", state_only=True)[0], "生产中")

    def test_reduced_quantity_is_independent_of_complete_button_label(self):
        for text, expected in (("生产5件", 5), ("生4件", 4), ("产3件", 3), ("生１件", 1),
                               ("生产", None), ("生产0件", None), ("生产4件价格10", None)):
            with self.subTest(text=text):
                self.assertEqual(dream.production_quantity(self.result(text, (351, 621, 110, 38))), expected)

    def test_missing_quantity_never_clicks_or_defaults_to_five(self):
        runner = self.runner()
        runner.hit = Mock(return_value=True)
        runner.reco = Mock(return_value=self.result("生", (351, 621, 30, 38)))
        now = [10.0]
        with (patch.object(dream.time, "monotonic", side_effect=lambda: now[0]),
              patch.object(dream.time, "sleep", side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds))):
            with self.assertRaisesRegex(dream.DreamError, "识别生产件数"):
                runner.submit_production(deadline=11)
        runner.context.run_task.assert_not_called()

    def test_reduced_batches_meet_actual_order_without_rounding_to_five(self):
        runner = self.runner()
        order = self.order(6)
        tiles = ((0, 0), (2, 3))
        idle = {tile: ("空闲中", (1, 1, 10, 10)) for tile in tiles}
        runner.wait = Mock(return_value=idle)
        amounts = iter((4, 3))
        def schedule(tile, actual_order, **kwargs):
            quantity = next(amounts)
            actual_order.scheduled += quantity
            runner.pending[tile] = dream.ProductionBatch(actual_order, quantity)
        def finish(*args):
            for batch in runner.pending.values():
                batch.order.harvested += batch.quantity
            runner.pending.clear()
        runner.schedule = Mock(side_effect=schedule)
        runner.finish_tiles = Mock(side_effect=finish)
        runner.produce([order], False)
        self.assertEqual(order.production_target, 6)
        self.assertEqual((order.scheduled, order.harvested), (7, 7))
        self.assertEqual(runner.schedule.call_count, 2)

    def test_polling_drops_harvested_slots_and_never_scans_other_rows(self):
        runner = self.runner()
        first, second = (0, 0), (2, 3)
        order = self.order(7)
        runner.pending = {first: dream.ProductionBatch(order, 4), second: dream.ProductionBatch(order, 3)}
        box = (1, 1, 10, 10)
        snapshots = iter((
            {first: ("生产中", box), second: ("生产中", box)},
            {first: ("可收获", box)}, {first: ("可收获", box)},
            {second: ("可收获", box)}, {second: ("可收获", box)},
            {first: ("休息中", box), second: ("空闲中", box)},
        ))
        scans = []
        def scan(image, expected=None, *, tiles=None):
            scans.append((expected, set(tiles)))
            return next(snapshots)
        runner.tile_states = Mock(side_effect=scan)
        runner.wait_tile = Mock(return_value=box)
        runner.click = Mock()
        runner.finish_tiles((first, second), False, dream.time.monotonic() + 30)
        self.assertEqual(scans, [(None, {first, second}), ("可收获", {first, second}),
                                 ("可收获", {first}), ("可收获", {second}),
                                 ("可收获", {second}), (("空闲中", "休息中"), {first, second})])
        self.assertEqual(order.harvested, 7)
        self.assertEqual(runner.pending, {})

    def test_management_poll_uses_one_marker_and_transition_checks_full_page(self):
        runner = dream.DreamRunner(Mock())
        runner.hit = Mock(return_value=True)
        self.assertTrue(runner.management_ready(object(), full=False))
        runner.hit.assert_called_once_with("织梦经营销售额", ANY)
        runner.hit = Mock(side_effect=[True, True, False, False])
        self.assertTrue(runner.management_ready(object(), full=True))
        self.assertEqual(runner.hit.call_count, 4)

    def test_failure_log_full_roi_harvest_is_ignored_and_idle_is_kept(self):
        runner = self.runner()
        runner.reco = Mock(side_effect=[
            self.result("可收获！", (781, 145, 137, 48)),
            self.result("空闲中", (805, 262, 66, 28)),
        ])
        self.assertEqual(runner.tile_state((0, 1), object()), ("空闲中", (805, 262, 66, 28)))

    def test_harvest_phase_reads_only_upper_band_and_rejects_whole_roi(self):
        runner = self.runner()
        runner.reco = Mock(return_value=self.result("可收获！", (781, 145, 137, 48)))
        self.assertIsNone(runner.tile_state((0, 1), object(), "可收获"))
        runner.reco.assert_called_once_with("织梦工位状态", ANY, [781, 145, 137, 48])
        runner.context.run_action_direct.assert_not_called()

    def test_production_phase_accepts_punctuation_but_only_real_lower_frame(self):
        runner = self.runner()
        runner.reco = Mock(return_value=self.result("生产中。", (809, 241, 55, 22)))
        self.assertEqual(runner.tile_state((0, 1), object(), "生产中")[0], "生产中")
        self.assertEqual(runner.reco.call_args.args[2], [781, 232, 137, 68])
        runner.reco.return_value = self.result("生产中", (781, 232, 137, 68))
        self.assertIsNone(runner.tile_state((0, 1), object(), "生产中"))

    def test_harvest_clicks_all_ready_tiles_before_checking_idle(self):
        runner = self.runner()
        first, second = (0, 0), (2, 3)
        boxes = {first: (660, 150, 90, 30), second: (1080, 480, 90, 30)}
        order = self.order(10)
        order.scheduled = 10
        runner.pending = {tile: dream.ProductionBatch(order, 5) for tile in (first, second)}
        runner.wait = Mock(return_value=True)
        runner.tile_states = Mock(side_effect=[
            {tile: ("可收获", box) for tile, box in boxes.items()},
            {second: ("可收获", boxes[second])}, {},
        ])
        events = []
        def wait_tile(tile, expected, **kwargs):
            if expected != "可收获":
                events.append(("idle", tile))
            return boxes[tile]
        runner.wait_tile = Mock(side_effect=wait_tile)
        runner.click = Mock(side_effect=lambda box: events.append(("click", box)))
        runner.harvest()
        self.assertEqual(events, [("click", boxes[first]), ("click", boxes[second]),
                                  ("idle", first), ("idle", second)])
        self.assertEqual(order.harvested, 10)
        self.assertEqual(runner.pending, {})
        self.assertTrue(all(call.args[1] == "可收获" for call in runner.tile_states.call_args_list))

    def test_failed_production_confirmation_never_repeats_start_or_counts_success(self):
        runner = self.runner()
        order = self.order()
        runner.wait = Mock(return_value=True)
        runner.wait_tile = Mock(side_effect=[(1, 1, 10, 10), dream.DreamError("生产未确认")])
        runner.click = Mock()
        runner.submit_production = Mock(return_value=5)
        runner.choose_category = Mock(side_effect=order.tags)
        with patch.object(dream.time, "monotonic", return_value=10):
            with self.assertRaisesRegex(dream.DreamError, "生产未确认"):
                runner.schedule((0, 0), order, deadline=610)
        self.assertEqual((order.scheduled, order.harvested), (0, 0))
        self.assertEqual(runner.pending, {})
        runner.submit_production.assert_called_once_with(deadline=40)
        self.assertEqual(runner.wait_tile.call_args_list[-1].kwargs["confirmations"], 1)

    def test_next_round_starts_only_after_entire_previous_round_is_harvested(self):
        runner = self.runner()
        tiles = ((0, 0), (2, 3))
        boxes = {tile: (650 + tile[1] * 141, 240 + tile[0] * 164, 80, 25) for tile in tiles}
        states = {tile: ("空闲中", boxes[tile]) for tile in tiles}
        events, now = [], [10.0]
        order = self.order(15)
        runner.tile_states = Mock(side_effect=lambda image, expected=None, **kwargs: {
            tile: state for tile, state in states.items()
            if (expected is None or state[0] in ((expected,) if isinstance(expected, str) else expected))
            and (kwargs.get("tiles") is None or tile in kwargs["tiles"])
        })
        def tile_state(tile, image, expected=None, **kwargs):
            state = states.get(tile)
            choices = (expected,) if isinstance(expected, str) else expected
            return state if state and (choices is None or state[0] in choices) else None
        runner.tile_state = Mock(side_effect=tile_state)
        def schedule(tile, actual_order, **kwargs):
            events.append(("schedule", tile))
            states[tile] = ("生产中", boxes[tile])
            actual_order.scheduled += 5
            runner.pending[tile] = dream.ProductionBatch(actual_order, 5)
        def click(box):
            tile = next(tile for tile in tiles if boxes[tile] == box)
            events.append(("harvest", tile))
            states[tile] = ("空闲中", box)
        def sleep(seconds):
            now[0] += 1
            for tile, (state, box) in list(states.items()):
                if state == "生产中":
                    states[tile] = ("可收获", box)
        runner.schedule = Mock(side_effect=schedule)
        runner.click = Mock(side_effect=click)
        with (patch.object(dream.time, "monotonic", side_effect=lambda: now[0]),
              patch.object(dream.time, "sleep", side_effect=sleep)):
            runner.produce([order], False)
        self.assertEqual(events, [("schedule", tiles[0]), ("schedule", tiles[1]),
                                  ("harvest", tiles[0]), ("harvest", tiles[1]),
                                  ("schedule", tiles[0]), ("harvest", tiles[0])])
        self.assertEqual((order.scheduled, order.harvested), (15, 15))
        self.assertEqual(runner.pending, {})

    def test_missing_recorded_tile_is_not_treated_as_finished(self):
        runner = self.runner()
        order = self.order()
        order.scheduled = 5
        runner.pending[(0, 0)] = dream.ProductionBatch(order, 5)
        runner.tile_states = Mock(return_value={(2, 3): ("空闲中", (1080, 580, 80, 25))})
        runner.harvest = Mock()
        now = [10.0]
        with (patch.object(dream.time, "monotonic", side_effect=lambda: now[0]),
              patch.object(dream.time, "sleep", side_effect=lambda seconds: now.__setitem__(0, now[0] + 1))):
            with self.assertRaisesRegex(dream.DreamError, "等待超时"):
                runner.finish_tiles(((0, 0), (2, 3)), False, 15)
        self.assertEqual(order.harvested, 0)
        self.assertIs(runner.pending[(0, 0)].order, order)

    def test_orders_drain_existing_batch_before_reading_new_demand(self):
        runner = self.runner()
        trace = Mock()
        for name in ("require_order_images", "enter_management", "finish_tiles", "enter_orders",
                     "read_orders", "button", "produce"):
            action = Mock(return_value=[self.order()] if name == "read_orders" else None)
            setattr(runner, name, action)
            trace.attach_mock(action, name)
        runner.wait = Mock(return_value={(0, 0): ("生产中", (1, 1, 10, 10))})
        runner.deliver_available = Mock(side_effect=[(0, 1), (1, 1)])
        runner.orders(False)
        names = [call[0] for call in trace.method_calls]
        self.assertLess(names.index("finish_tiles"), names.index("read_orders"))
        self.assertLess(names.index("read_orders"), names.index("produce"))

    def test_all_rest_stops_unfinished_order_without_clicking(self):
        runner = self.runner()
        runner.wait = Mock(return_value={(0, 0): ("休息中", (1, 1, 10, 10))})
        runner.schedule = Mock()
        with self.assertRaisesRegex(dream.DreamError, "订单需求尚未完成"):
            runner.produce([self.order()], False)
        runner.schedule.assert_not_called()


if __name__ == "__main__":
    unittest.main()
