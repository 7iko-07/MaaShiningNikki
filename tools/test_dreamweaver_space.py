"""Offline behavior tests; screenshots and timings still require device validation."""

import importlib.util
import json
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from maa.agent.agent_server import AgentServer
from maa.define import Rect
from tools.validate_schema import load_jsonc


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location(
    "dream_under_test", ROOT / "agent/custom/dreamweaver_space.py"
)
dream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dream
with patch.object(AgentServer, "register_custom_action", return_value=True):
    spec.loader.exec_module(dream)


class DreamTests(unittest.TestCase):
    def runner(self):
        context = Mock()
        context.tasker.stopping = False
        return dream.DreamRunner(context)

    def order(self, quantity=15):
        return dream.Order(dream.Card(1, (490, 285, 68, 54)), quantity, ("连衣裙", "布艺", "绿色系"))

    @contextmanager
    def selection_clock(self):
        now = [10.0]

        def sleep(seconds):
            now[0] += seconds

        with patch.object(dream.time, "monotonic", side_effect=lambda: now[0]), \
                patch.object(dream.time, "sleep", side_effect=sleep):
            yield now

    def test_progress_uses_colon_field_and_completed_numerator(self):
        for text, expected in (
            ("今日可完成订单：0/5", (0, 5)),
            ("今日可完成订单:５ / ５", (5, 5)),
            ("今日可完成订单：1/5", (1, 5)),
        ):
            self.assertEqual(dream.parse_progress(text), expected)
        for text in ("0/5", "库存：0/5", "今日可完成订单：6/5", "今日可完成订单：0/0", "今日可完成订单：1/5 2/5"):
            self.assertIsNone(dream.parse_progress(text))

    def test_quantity_and_tags_reject_unknown_and_conflicting_requirements(self):
        self.assertEqual(dream.parse_quantity("数量：１５件"), 15)
        self.assertIsNone(dream.parse_quantity("库存15件"))
        self.assertEqual(dream.parse_tags("素材标签：连衣裙布艺绿色系"), ("连衣裙", "布艺", "绿色系"))
        self.assertEqual(dream.parse_tags("素材标签：特殊"), (None, "特殊", None))
        self.assertEqual(dream.parse_tags("连衣裙素材标签：布艺绿色系"), ("连衣裙", "布艺", "绿色系"))
        for text in ("素材标签：", "素材标签：连衣裙上装", "素材标签：布艺紫色系", "素材标签：连衣裙???"):
            self.assertIsNone(dream.parse_tags(text))

    def test_scheme_uses_category_labels_despite_ocr_baseline_order(self):
        self.assertEqual(dream.parse_scheme("连衣裙布艺方案使用素材:蓝色系"), ("连衣裙", "布艺", "蓝色系"))
        self.assertIsNone(dream.parse_scheme("连衣裙布艺方案使用素材:紫色系"))

    def test_production_does_not_deduct_stock_and_rounds_up_batches(self):
        for quantity, target in ((1, 1), (5, 5), (6, 6), (15, 15)):
            self.assertEqual(self.order(quantity).production_target, target)

    def test_overlapping_stars_are_deduplicated_but_same_country_cards_remain(self):
        a = dream.Card(3, (100, 100, 65, 90))
        b = dream.Card(1, (101, 101, 64, 50))
        c = dream.Card(2, (180, 100, 65, 70))
        d = dream.Card(1, (260, 100, 65, 50))
        self.assertEqual(dream.deduplicate_cards([b, a, c, d]), [d, c, a])

    def test_counted_stars_take_priority_over_an_uncertain_template_match(self):
        confirmed = dream.Card(2, (100, 100, 70, 77), True)
        uncertain = dream.Card(3, (100, 78, 68, 99))
        self.assertEqual(dream.deduplicate_cards([uncertain, confirmed]), [confirmed])

    def test_map_counts_extra_stars_above_a_one_star_template_match(self):
        image = dream.np.zeros((720, 1280, 3), dtype=dream.np.uint8)
        box = (1082, 451, 70, 53)
        self.assertIsNone(dream.map_star_count(image, box))
        image[462:482, 1090:1114] = (50, 160, 255)
        self.assertEqual(dream.map_star_count(image, box), 1)
        image[440:460, 1090:1114] = (50, 160, 255)
        self.assertEqual(dream.map_star_count(image, box), 2)
        image[416:436, 1090:1114] = (50, 160, 255)
        self.assertEqual(dream.map_star_count(image, box), 3)
        self.assertIsNone(dream.map_star_count(image, (1270, 451, 70, 53)))

    def test_detail_counts_one_to_three_stars_and_rejects_bad_spacing(self):
        image = dream.np.zeros((720, 1280, 3), dtype=dream.np.uint8)
        self.assertIsNone(dream.detail_star_count(image))
        for count, x in enumerate((966, 1014, 1062), 1):
            image[152:186, x:x + 32] = (50, 160, 255)
            self.assertEqual(dream.detail_star_count(image), count)
        image[:, 1014:1046] = 0
        self.assertIsNone(dream.detail_star_count(image))

    def test_unreadable_detail_stars_do_not_extend_the_page_wait(self):
        runner = self.runner()
        image = dream.np.zeros((720, 1280, 3), dtype=dream.np.uint8)
        runner.hit = Mock(return_value=True)
        self.assertEqual(runner.order_detail(image), (None,))
        runner.hit.return_value = False
        self.assertIsNone(runner.order_detail(image))

    def test_map_corrects_partial_one_star_hits_and_keeps_unknown_candidates(self):
        runner = self.runner()
        boxes = [(1082, 451, 70, 53), (556, 503, 70, 53)]
        def reco(node, image):
            items = [SimpleNamespace(box=box) for box in boxes] if node == "织梦一星订单" else []
            return SimpleNamespace(hit=bool(items), filtered_results=items)
        runner.reco = Mock(side_effect=reco)
        with patch.object(dream, "map_star_count", side_effect=[2, None]):
            cards = runner.cards(object())
        self.assertEqual(cards, [dream.Card(1, boxes[1]), dream.Card(2, boxes[0], True)])

    def test_read_orders_corrects_detail_stars_or_continues_when_unknown(self):
        for map_confirmed, detail_stars in ((False, 2), (False, None), (True, None)):
            runner = self.runner()
            card = dream.Card(1, (490, 285, 68, 54), map_confirmed)
            runner.cards = Mock(return_value=[card])
            runner.image = Mock(return_value=object())
            runner.completed = Mock(return_value=False)
            runner.box = Mock(return_value=None)
            runner.click = Mock()
            tags = (None, "特殊", "灰色系")
            runner.wait = Mock(side_effect=[(detail_stars,), (10, tags), (4, 5)])
            orders = runner.read_orders(1)
            self.assertEqual((orders[0].quantity, orders[0].tags), (10, tags))
            self.assertEqual(orders[0].card.stars, detail_stars or card.stars)
            self.assertEqual(orders[0].card.stars_confirmed, detail_stars is not None or map_confirmed)
            self.assertEqual(runner.wait.call_count, 3)
            self.assertEqual(runner.wait.call_args_list[0].args[0], runner.order_detail)
            self.assertEqual(runner.click.call_args_list[0].args, (card.box,))

    def test_tick_roi_is_relative_lower_right_and_clipped(self):
        self.assertEqual(dream.check_roi((490, 285, 68, 54), 1280, 720), [524, 298, 68, 68])
        self.assertEqual(dream.check_roi((1220, 670, 60, 50), 1280, 720), [1250, 682, 30, 38])
        runner = self.runner()
        image = SimpleNamespace(shape=(720, 1280, 3))
        runner.reco = Mock(return_value=SimpleNamespace(
            hit=True, filtered_results=[SimpleNamespace(box=(540, 320, 42, 32))],
        ))
        self.assertTrue(runner.completed(self.order().card, image))
        runner.reco.assert_called_once_with("织梦订单完成对号", image)

    def test_full_screen_ticks_only_complete_their_own_card(self):
        runner = self.runner()
        image = SimpleNamespace(shape=(720, 1280, 3))
        card = dream.Card(1, (968, 508, 70, 53))
        for boxes, expected in (
            ([(599, 522, 42, 32)], False),
            ([(599, 522, 42, 32), (1011, 527, 42, 32)], True),
            ([(1011, 527, 0, 32)], False),
            ([], False),
        ):
            runner.reco = Mock(return_value=SimpleNamespace(
                hit=bool(boxes), filtered_results=[SimpleNamespace(box=box) for box in boxes],
            ))
            self.assertEqual(runner.completed(card, image), expected)
            self.assertFalse(runner.completed(dream.Card(2, (1083, 430, 70, 77)), image))

    def test_order_labels_stars_and_ticks_use_full_screen_recognition(self):
        pipeline = load_jsonc(ROOT / "assets/resource/pipeline/dreamweaver_space.json")
        for node in ("织梦可交付订单", "织梦一星订单", "织梦二星订单", "织梦三星订单", "织梦订单完成对号"):
            self.assertNotIn("roi", pipeline[node]["recognition"]["param"], node)

    def test_ocr_reads_all_tags_once_in_display_order(self):
        runner = self.runner()
        items = [
            SimpleNamespace(box=Rect(x, 360, 50, 20), text=text)
            for x, text in (
                (1110, "绿色系"),
                (850, "素材标签："),
                (1030, "布艺"),
                (960, "连衣裙"),
            )
        ]
        runner.reco = Mock(
            return_value=SimpleNamespace(
                hit=True, filtered_results=items, best_result=items[0]
            )
        )
        self.assertEqual(runner.text("标签", object()), "素材标签：连衣裙布艺绿色系")

    def test_status_bands_keep_same_tile_for_all_three_states(self):
        runner = self.runner()
        harvested = SimpleNamespace(text="可收获！", box=Rect(800, 153, 90, 31))
        producing = SimpleNamespace(text="生产中", box=Rect(669, 244, 81, 23))
        idle = SimpleNamespace(text="空闲中", box=Rect(943, 261, 69, 28))

        def result(items):
            return SimpleNamespace(hit=bool(items), filtered_results=items, best_result=None)

        def reco(node, image, roi):
            x, y, w, h = roi
            return result([item for item in (harvested, producing, idle)
                           if node == "织梦工位状态" and x <= item.box[0] < x + w
                           and y <= item.box[1] < y + h])
        runner.reco = Mock(side_effect=reco)
        states = runner.tile_states(object())
        self.assertEqual(
            {tile: value[0] for tile, value in states.items()},
            {(0, 0): "生产中", (0, 1): "可收获", (0, 2): "空闲中"},
        )
        self.assertEqual(
            [call.args[2] for call in runner.reco.call_args_list[:2]],
            [[640, 145, 137, 48], [640, 232, 137, 68]],
        )

    def test_all_twelve_unlocked_tiles_are_recognized_including_lower_rows(self):
        runner = self.runner()
        status_items = []
        expected = {}
        for row in range(3):
            for col in range(4):
                text = ("可收获", "空闲中", "生产中", "可收获")[col]
                expected[(row, col)] = text
                y = (153 if text == "可收获" else 244) + row * 164
                item = SimpleNamespace(text=text, box=Rect(665 + col * 141, y, 80, 25))
                status_items.append(item)
        def reco(node, image, roi):
            x, y, w, h = roi
            found = [item for item in status_items if node == "织梦工位状态"
                     and x <= item.box[0] < x + w and y <= item.box[1] < y + h]
            return SimpleNamespace(hit=bool(found), filtered_results=found)
        runner.reco = Mock(side_effect=reco)
        states = runner.tile_states(object())
        self.assertEqual({tile: value[0] for tile, value in states.items()}, expected)
        self.assertEqual(len(states), 12)
        self.assertEqual(runner.reco.call_args_list[-1].args[2], [1063, 473, 137, 48])

    def test_full_roi_false_hit_and_neighbor_box_never_allow_click(self):
        runner = self.runner()
        def result(text, box):
            return SimpleNamespace(hit=True, filtered_results=[SimpleNamespace(text=text, box=box)])
        runner.reco = Mock(side_effect=[result("可收获", (781, 145, 137, 48)),
                                       result("空闲中", (803, 261, 69, 28))])
        self.assertEqual(runner.tile_state((0, 1), object())[0], "空闲中")
        runner.reco = Mock(side_effect=[result("可收获", (940, 153, 90, 31)),
                                       SimpleNamespace(hit=False), SimpleNamespace(hit=False)])
        self.assertIsNone(runner.tile_state((0, 1), object()))
        runner.context.run_action_direct.assert_not_called()

    def test_no_hit_and_invalid_click_do_not_click(self):
        runner = self.runner()
        runner.reco = Mock(return_value=SimpleNamespace(hit=False, box=Rect(10, 10, 50, 20)))
        self.assertIsNone(runner.box("按钮", object()))
        with self.assertRaises(dream.DreamError):
            runner.click((1, 1, 0, 20))
        runner.context.run_action_direct.assert_not_called()

    def test_wait_uses_one_deadline_and_requires_two_state_observations(self):
        runner = self.runner()
        now = [0.0]
        runner.image = Mock(return_value=object())
        predicate = Mock(side_effect=[True, False, True, True])
        with (
            patch.object(dream.time, "monotonic", side_effect=lambda: now[0]),
            patch.object(
                dream.time,
                "sleep",
                side_effect=lambda dt: now.__setitem__(0, now[0] + dt),
            ),
        ):
            self.assertTrue(runner.wait(predicate, "结果", deadline=1))
            self.assertEqual(predicate.call_count, 4)
            with self.assertRaises(dream.DreamError):
                runner.wait(lambda image: False, "结果", deadline=1)
            self.assertLess(now[0], 1.3)

    def test_cancellation_stops_before_screenshot(self):
        runner = self.runner()
        runner.context.tasker.stopping = True
        with self.assertRaises(dream.DreamError):
            runner.image()
        runner.context.tasker.controller.post_screencap.assert_not_called()

    def test_failed_screenshot_does_not_reuse_cached_old_image(self):
        runner = self.runner()
        runner.context.tasker.controller.post_screencap.return_value.wait.return_value = (
            SimpleNamespace(succeeded=False)
        )
        runner.context.tasker.controller.cached_image = object()
        with self.assertRaisesRegex(dream.DreamError, "不使用上一帧"):
            runner.image()

    def test_tick_recognition_not_started_is_an_error_not_uncompleted(self):
        runner = self.runner()
        runner.context.run_recognition.return_value = None
        with self.assertRaisesRegex(dream.DreamError, "识别未能启动"):
            runner.completed(self.order().card, SimpleNamespace(shape=(720, 1280, 3)))

    def test_sign_in_waits_for_each_popup_and_closes_each_only_once(self):
        runner = self.runner()
        runner.button = Mock()
        runner.click = Mock()
        runner.wait = Mock(return_value=True)
        runner.sign_in()
        self.assertEqual(runner.click.call_args_list[0].args[0], [550, 625, 180, 55])
        self.assertEqual(runner.click.call_count, 2)
        runner.button.assert_called_once_with("织梦今日签到图标")

    def test_popup_background_does_not_count_as_space_return(self):
        runner = self.runner()
        runner.hit = Mock(side_effect=lambda node, image: node in ("导航在织梦空间页面", "织梦获得奖励文字"))
        self.assertFalse(runner.space_ready(object()))

    def test_delivery_refreshes_available_after_each_completed_order(self):
        runner = self.runner()
        frames = iter(("first", "inventory_changed"))
        runner.image = Mock(side_effect=lambda: next(frames))
        runner.box = Mock(
            side_effect=lambda node, image: (
                (742, 369, 50, 23) if image == "first" else None
            )
        )
        runner.click = Mock()
        runner.button = Mock()
        progress = [(0, 5), (1, 5)]
        def wait(predicate, label, **kwargs):
            if label == "订单地图":
                return progress.pop(0)
            if label == "产品自动选满":
                runner.hit = Mock(return_value=True)
                runner.text = Mock(return_value="已选择:15/15")
                self.assertTrue(predicate("selected"))
            if label == "交付后今日完成数增加一次":
                runner.map_ready = Mock(return_value=(1, 5))
                self.assertEqual(predicate("returned"), (1, 5))
            return True
        runner.wait = Mock(side_effect=wait)
        self.assertEqual(runner.deliver_available(), (1, 5))
        self.assertEqual(
            [c.args[1] for c in runner.box.call_args_list],
            ["first", "inventory_changed"],
        )
        self.assertEqual(runner.click.call_count, 2)  # 可交付 + 固定产品框。
        # 失败日志中的标签挨着三星订单；目标应落在二星卡片 (716,379,70,77) 内。
        target_x, target_y, target_w, target_h = runner.click.call_args_list[0].args[0]
        self.assertTrue(716 <= target_x < 786 and 379 <= target_y < 456)
        self.assertEqual((target_w, target_h), (1, 1))
        self.assertEqual(
            [c.args[0] for c in runner.button.call_args_list],
            ["织梦详情交付按钮", "织梦产品交付按钮"],
        )

    def test_delivery_at_daily_cap_does_not_click(self):
        runner = self.runner()
        runner.wait = Mock(return_value=(5, 5))
        runner.click = Mock()
        self.assertEqual(runner.deliver_available(), (5, 5))
        runner.click.assert_not_called()

    def test_incomplete_selection_never_clicks_delivery_again(self):
        runner = self.runner()
        runner.image = Mock(return_value="frame")
        runner.box = Mock(return_value=(100, 100, 30, 20))
        runner.click = Mock()
        runner.button = Mock()
        runner.hit = Mock(return_value=True)
        runner.text = Mock(return_value="已选择:0/15")
        def wait(predicate, label, **kwargs):
            if label == "订单地图":
                return (0, 5)
            if label == "产品自动选满":
                self.assertFalse(predicate("frame"))
                raise dream.DreamError("超时")
            return True
        runner.wait = Mock(side_effect=wait)
        with self.assertRaises(dream.DreamError):
            runner.deliver_available()
        runner.button.assert_called_once_with("织梦详情交付按钮")

    def test_completed_order_is_rechecked_and_never_clicked(self):
        runner = self.runner()
        card = self.order().card
        runner.cards = Mock(return_value=[card])
        runner.image = Mock(return_value=object())
        runner.completed = Mock(side_effect=[False, True])
        runner.click = Mock()
        with self.assertRaises(dream.DreamError):
            runner.read_orders(1)
        runner.click.assert_not_called()

    def test_missing_order_images_stop_before_management_clicks(self):
        runner = self.runner()
        runner.enter_management = Mock()
        with patch.object(dream.Path, "is_file", return_value=False):
            with self.assertRaisesRegex(dream.DreamError, "订单完成对号.png"):
                runner.orders(False)
        runner.enter_management.assert_not_called()

    def test_schedule_counts_only_after_same_tile_production_confirmed(self):
        runner = self.runner()
        order = self.order()
        tile = (0, 1)
        runner.click = Mock()
        runner.submit_production = Mock(return_value=5)
        runner.image = Mock(return_value=object())
        runner.wait = Mock(return_value=True)
        runner.choose_category = Mock(side_effect=order.tags)
        def wait_tile(actual_tile, state, **kwargs):
            self.assertEqual(actual_tile, tile)
            self.assertEqual(order.scheduled, 0)
            self.assertNotIn(tile, runner.pending)
            return (1, 1, 10, 10)
        runner.wait_tile = Mock(side_effect=wait_tile)
        runner.schedule(tile, order)
        self.assertEqual(order.scheduled, 5)
        self.assertIs(runner.pending[tile].order, order)
        self.assertEqual(runner.wait_tile.call_args_list[-1].args, (tile, "生产中"))
        self.assertEqual(runner.wait_tile.call_args_list[-1].kwargs["confirmations"], 1)

    def test_failed_production_click_does_not_reserve_or_count_batch(self):
        runner = self.runner()
        order = self.order()
        runner.click = Mock()
        runner.submit_production = Mock(side_effect=dream.DreamError("点击失败"))
        runner.image = Mock(return_value=object())
        runner.wait = Mock(return_value=True)
        runner.choose_category = Mock(side_effect=order.tags)
        runner.wait_tile = Mock(return_value=(1, 1, 10, 10))
        with self.assertRaises(dream.DreamError):
            runner.schedule((0, 0), order)
        self.assertEqual(order.scheduled, 0)
        self.assertEqual(runner.pending, {})

    def test_harvest_counts_only_owned_slot_after_idle_confirmed(self):
        runner = self.runner()
        order = self.order()
        owned, other = (0, 0), (0, 1)
        runner.pending[owned] = dream.ProductionBatch(order, 5)
        runner.image = Mock(return_value=object())
        runner.wait = Mock(return_value=True)
        runner.wait_tile = Mock(return_value=(1, 1, 10, 10))
        runner.click = Mock()
        runner.tile_states = Mock(
            side_effect=[
                {other: ("可收获", (1, 1, 10, 10)), owned: ("可收获", (1, 1, 10, 10))},
                {},
            ]
        )
        runner.harvest()
        self.assertEqual(order.harvested, 5)
        self.assertEqual(runner.pending, {})

    def test_acceleration_is_clicked_once_then_waits_for_tile(self):
        runner = self.runner()
        runner.wait_tile = Mock(return_value=(1, 1, 10, 10))
        runner.wait = Mock(return_value=True)
        runner.click = Mock()
        runner.button = Mock()
        runner.harvest = Mock()
        runner.accelerate((0, 0), 600)
        self.assertEqual(runner.wait_tile.call_args_list[0].kwargs["confirmations"], 1)
        runner.button.assert_called_once_with("织梦立即完成按钮", deadline=600)
        self.assertEqual(runner.wait_tile.call_args_list[-1].args, ((0, 0), "可收获"))

    def test_acceleration_result_timeout_never_repeats_stone_click(self):
        runner = self.runner()
        runner.wait_tile = Mock(side_effect=[(1, 1, 10, 10), dream.DreamError("动画未结束")])
        runner.wait = Mock(return_value=True)
        runner.click = Mock()
        runner.button = Mock()
        with self.assertRaises(dream.DreamError):
            runner.accelerate((0, 0), 600)
        runner.button.assert_called_once_with("织梦立即完成按钮", deadline=600)

    def test_natural_wait_is_ten_minutes_not_reset_by_polling(self):
        runner = self.runner()
        order = self.order(5)
        order.scheduled = 5
        runner.pending[(0, 0)] = dream.ProductionBatch(order, 5)
        runner.image = Mock(return_value=object())
        runner.harvest = Mock()
        runner.tile_states = Mock(side_effect=lambda image, expected=None, **kwargs:
                                  {(0, 0): ("生产中", (1, 1, 10, 10))} if expected is None else {})
        runner.management_ready = Mock(return_value=True)
        now = [10.0]
        def wait(predicate, label, **kwargs):
            if "deadline" not in kwargs:
                return predicate(object())
            self.assertEqual(kwargs["deadline"], 610)
            if label == "织梦经营工位状态":
                return predicate(object())
            self.assertFalse(predicate(object()))
            now[0] = kwargs["deadline"]
            raise dream.DreamError("超时")
        runner.wait = Mock(side_effect=wait)
        with (
            patch.object(dream.time, "monotonic", side_effect=lambda: now[0]),
            self.assertRaises(dream.DreamError),
        ):
            runner.produce([order], False)
        self.assertEqual(order.harvested, 0)

    def test_production_fills_idle_slots_and_tracks_two_rounds(self):
        runner = self.runner()
        order = self.order(15)
        states = {(0, 0): ("空闲中", (1, 1, 10, 10)), (0, 1): ("空闲中", (1, 1, 10, 10))}
        runner.image = Mock(return_value=object())
        runner.tile_states = Mock(side_effect=lambda image, expected=None, **kwargs: dict(states))
        runner.management_ready = Mock(return_value=True)
        deadlines = []
        def schedule(tile, actual_order, **kwargs):
            deadlines.append(kwargs["deadline"])
            states[tile] = ("生产中", (1, 1, 10, 10))
            runner.pending[tile] = dream.ProductionBatch(actual_order, 5)
            actual_order.scheduled += 5
        def harvest(**kwargs):
            for tile, (state, box) in list(states.items()):
                if state == "可收获":
                    runner.pending.pop(tile).order.harvested += 5
                    states[tile] = ("空闲中", box)
        def finish(tiles, use_stone, deadline):
            self.assertEqual(deadline, 610)
            for tile, (state, box) in list(states.items()):
                if state == "生产中":
                    states[tile] = ("可收获", box)
            harvest(deadline=deadline)
        def wait(predicate, description, **kwargs):
            return predicate(object())
        runner.schedule = Mock(side_effect=schedule)
        runner.harvest = Mock(side_effect=harvest)
        runner.finish_tiles = Mock(side_effect=finish)
        runner.wait = Mock(side_effect=wait)
        with patch.object(dream.time, "monotonic", return_value=10):
            runner.produce([order], False)
        self.assertEqual((order.scheduled, order.harvested), (15, 15))
        self.assertEqual(deadlines, [610, 610, 610])
        self.assertEqual(runner.pending, {})

    def test_fully_produced_but_undeliverable_does_not_plan_again(self):
        runner = self.runner()
        runner.require_order_images = Mock()
        runner.enter_management = Mock()
        runner.harvest = Mock()
        runner.enter_orders = Mock()
        runner.deliver_available = Mock(side_effect=[(4, 5), (4, 5)])
        runner.read_orders = Mock(return_value=[self.order()])
        runner.button = Mock()
        runner.wait = Mock(return_value={(0, 0): ("空闲中", (1, 1, 10, 10))})
        runner.produce = Mock()
        with self.assertRaisesRegex(dream.DreamError, "避免重复生产"):
            runner.orders(False)
        runner.read_orders.assert_called_once_with(1)
        runner.produce.assert_called_once()

    def test_exact_required_material_does_not_fall_back_to_any(self):
        runner = self.runner()
        runner.reco = Mock(
            return_value=SimpleNamespace(
                hit=True,
                filtered_results=[
                    SimpleNamespace(text="布艺", box=Rect(800, 380, 40, 20))
                ],
            )
        )
        runner.hit = Mock(return_value=True)
        runner.text = Mock(return_value="方案使用素材：连衣裙XX蓝色系")
        runner.image = Mock(return_value=object())
        runner.click = Mock()
        with self.selection_clock(), self.assertRaisesRegex(dream.DreamError, "皮革.*已选中"):
            runner.choose_category(object(), "材质", dream.MATERIALS, "皮革", deadline=11)
        runner.click.assert_not_called()
        runner.context.run_action_direct.assert_not_called()

    def test_absent_category_uses_recognized_option(self):
        runner = self.runner()
        runner.reco = Mock(
            return_value=SimpleNamespace(
                hit=True,
                filtered_results=[
                    SimpleNamespace(text="上装", box=Rect(900, 180, 40, 20))
                ],
            )
        )
        runner.hit = Mock(return_value=True)
        runner.text = Mock(side_effect=["方案使用素材：XXXXXX", "方案使用素材：上装XXXX"])
        runner.image = Mock(return_value=object())
        runner.click = Mock()
        with self.selection_clock():
            self.assertEqual(runner.choose_category(object(), "版型", dream.TYPES, None, deadline=11), "上装")
        runner.click.assert_called_once_with((920, 190, 1, 1))

    def test_special_material_swipe_uses_confirmed_coordinates(self):
        runner = self.runner()
        special = SimpleNamespace(
            hit=True, filtered_results=[SimpleNamespace(text="特殊", box=Rect(1100, 380, 40, 20))]
        )
        runner.reco = Mock(side_effect=[SimpleNamespace(hit=False), special, special])
        runner.hit = Mock(return_value=True)
        runner.text = Mock(side_effect=["方案使用素材：XXXXXX", "方案使用素材：XXXXXX", "方案使用素材：特殊XXXX"])
        runner.image = Mock(return_value=object())
        runner.context.run_action_direct.return_value = SimpleNamespace(success=True)
        runner.click = Mock()
        with self.selection_clock():
            self.assertEqual(runner.choose_category(object(), "材质", dream.MATERIALS, "特殊", deadline=11), "特殊")
        runner.context.run_action_direct.assert_called_once()
        params = runner.context.run_action_direct.call_args.args[1]
        self.assertEqual(params.begin, [1063, 342])
        self.assertEqual(params.end, [[945, 345]])
        runner.click.assert_called_once_with((1120, 390, 1, 1))

    def test_option_missed_click_retries_fresh_center_until_selected(self):
        runner = self.runner()
        runner.hit = Mock(return_value=True)
        runner.image = Mock(return_value=object())
        runner.click = Mock()
        runner.text = Mock(side_effect=lambda *args: (
            "方案使用素材：连衣裙布艺蓝色系" if runner.click.call_count == 2
            else "方案使用素材：连衣裙蓝色系xX"
        ))
        with self.selection_clock() as now:
            def reco(node, image):
                box = (804, 375, 44, 30) if now[0] == 10 else (810, 370, 44, 30)
                return SimpleNamespace(hit=True, filtered_results=[SimpleNamespace(text="布艺", box=box)])

            runner.reco = Mock(side_effect=reco)
            self.assertEqual(runner.choose_category(object(), "材质", dream.MATERIALS, None, deadline=12), "布艺")
        self.assertEqual([call.args[0] for call in runner.click.call_args_list],
                         [(826, 390, 1, 1), (832, 385, 1, 1)])
        self.assertGreaterEqual(runner.image.call_count, 4)

    def test_already_selected_category_is_not_clicked_again(self):
        runner = self.runner()
        runner.hit = Mock(return_value=True)
        runner.reco = Mock(return_value=SimpleNamespace(
            hit=True, filtered_results=[SimpleNamespace(text="布艺", box=(804, 375, 44, 30))]
        ))
        runner.text = Mock(return_value="方案使用素材：连衣裙布艺XX")
        runner.click = Mock()
        with self.selection_clock():
            self.assertEqual(runner.choose_category(object(), "材质", dream.MATERIALS, "布艺", deadline=11), "布艺")
        runner.click.assert_not_called()

    def test_option_disappears_after_click_never_reuses_previous_box(self):
        runner = self.runner()
        runner.hit = Mock(return_value=True)
        runner.reco = Mock(side_effect=lambda *args: SimpleNamespace(
            hit=not runner.click.called,
            filtered_results=[SimpleNamespace(text="布艺", box=(804, 375, 44, 30))],
        ))
        runner.text = Mock(return_value="方案使用素材：连衣裙XX蓝色系")
        runner.image = Mock(return_value=object())
        runner.click = Mock()
        with self.selection_clock(), self.assertRaisesRegex(dream.DreamError, "布艺.*已选中"):
            runner.choose_category(object(), "材质", dream.MATERIALS, None, deadline=11)
        runner.click.assert_called_once_with((826, 390, 1, 1))

    def test_selection_checks_design_page_before_recognition_and_click(self):
        runner = self.runner()
        runner.hit = Mock(return_value=False)
        runner.reco = Mock()
        runner.text = Mock()
        runner.image = Mock(return_value=object())
        runner.click = Mock()
        with self.selection_clock(), self.assertRaises(dream.DreamError):
            runner.choose_category(object(), "材质", dream.MATERIALS, "布艺", deadline=11)
        runner.reco.assert_not_called()
        runner.text.assert_not_called()
        runner.click.assert_not_called()

    def test_cancelled_selection_stops_before_any_click(self):
        runner = self.runner()
        runner.context.tasker.stopping = True
        runner.click = Mock()
        with self.selection_clock(), self.assertRaisesRegex(dream.DreamError, "任务已取消"):
            runner.choose_category(object(), "材质", dream.MATERIALS, "布艺", deadline=11)
        runner.click.assert_not_called()

    def test_unselected_material_times_out_without_submitting_or_counting(self):
        runner = self.runner()
        runner.hit = Mock(return_value=True)
        runner.reco = Mock(return_value=SimpleNamespace(
            hit=True, filtered_results=[SimpleNamespace(text="布艺", box=(804, 375, 44, 30))]
        ))
        runner.text = Mock(return_value="方案使用素材：连衣裙XX蓝色系")
        runner.image = Mock(return_value=object())
        runner.click = Mock()
        runner.wait_tile = Mock(return_value=(1, 1, 10, 10))
        runner.wait = Mock(return_value=True)
        runner.submit_production = Mock()
        order = self.order()
        with self.selection_clock(), self.assertRaisesRegex(dream.DreamError, "布艺.*已选中"):
            runner.schedule((0, 1), order, deadline=11)
        runner.submit_production.assert_not_called()
        self.assertEqual((order.scheduled, order.harvested), (0, 0))
        self.assertEqual(runner.pending, {})

    def test_action_runs_selected_sign_before_orders_and_failure_propagates(self):
        context = Mock()
        flags = {"织梦选择今日签到", "织梦选择经营", "织梦选择订单驿站", "织梦选择能量石加速"}
        context.get_node_data.side_effect = lambda node: {"enabled": node in flags}
        with patch.object(dream, "DreamRunner") as factory:
            self.assertTrue(dream.DreamweaverSpaceAction().run(context, SimpleNamespace()))
            self.assertEqual(
                [c[0] for c in factory.return_value.method_calls],
                ["sign_in", "orders"],
            )
            factory.return_value.orders.assert_called_once_with(True)
            factory.return_value.orders.side_effect = dream.DreamError("未完成")
            self.assertFalse(dream.DreamweaverSpaceAction().run(context, SimpleNamespace()))

    def test_task_entry_and_choice_overrides_exist(self):
        pipeline = load_jsonc(ROOT / "assets/resource/pipeline/dreamweaver_space.json")
        interface = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
        task = next(task for task in interface["task"] if task["entry"] == "织梦空间任务链入口")
        self.assertIn(task["entry"], pipeline)
        self.assertEqual(pipeline["织梦空间任务链准备"]["next"], ["织梦空间执行子任务"])
        for option in ("织梦空间子任务", "织梦经营子任务", "织梦订单生产完成方式"):
            for case in interface["option"][option]["cases"]:
                for node in case.get("pipeline_override", {}):
                    self.assertIn(node, pipeline)
        self.assertEqual(
            [c["name"] for c in interface["option"]["织梦空间子任务"]["cases"]],
            ["七日妙想", "今日签到", "织梦经营", "清理成衣作品"],
        )
        self.assertEqual(
            [c["name"] for c in interface["option"]["织梦经营子任务"]["cases"]],
            ["每日能量晶矿", "订单驿站", "售卖橱窗"],
        )


if __name__ == "__main__":
    unittest.main()
