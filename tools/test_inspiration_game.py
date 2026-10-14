"""灵感拾趣：环形选路、身体增长、转向反馈及单局返回。"""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools.test_dreamweaver_space import ROOT, dream
from tools.validate_schema import load_jsonc
COLS, ROWS = dream.COLS, dream.ROWS
Tracker, TrackingError = dream.Tracker, dream.TrackingError
advance, cell_at, center = dream.advance, dream.cell_at, dream.center
choose_direction, displacement, step = dream.choose_direction, dream.displacement, dream.step


class PlannerTests(unittest.TestCase):
    def test_wrapping_and_screenshot_geometry(self):
        self.assertEqual(step((12, 7), (1, 0)), (0, 7))
        self.assertEqual(step((0, 0), (0, -1)), (0, 7))
        for x in range(COLS):
            for y in range(ROWS):
                self.assertEqual(cell_at(center((x, y))), (x, y))
        self.assertEqual(cell_at((521, 570)), (4, 6))
        self.assertEqual(cell_at((414, 487)), (2, 5))
        self.assertAlmostEqual(displacement(center((12, 0)), center((0, 0)))[0], 84.5)

    def test_tail_can_move_but_growth_prevents_entering_tail(self):
        body = ((1, 1), (1, 2), (0, 2), (0, 1))
        self.assertEqual(advance(body, (0, 1), frozenset())[0],
                         ((0, 1), (1, 1), (1, 2), (0, 2)))
        self.assertIsNone(advance(body, (0, 1), frozenset({(0, 1)})))
        self.assertIsNone(advance(body, (1, 2), frozenset()))
        moved, remaining = advance(body, (2, 1), frozenset({(2, 1)}))
        self.assertEqual(len(moved), 5)
        self.assertFalse(remaining)

    def test_food_route_uses_wrapped_shortest_path(self):
        self.assertEqual(choose_direction(((0, 3),), (0, 1), {(12, 3)}), (-1, 0))
        self.assertEqual(choose_direction(((5, 0),), (1, 0), {(5, 7)}), (0, -1))

    def test_no_reverse_no_body_collision_and_no_target_under_joystick(self):
        body = ((5, 5), (5, 6), (6, 6), (6, 5))
        direction = choose_direction(body, (0, -1), {(6, 5), (4, 5)})
        self.assertNotEqual(direction, (0, 1))
        self.assertIsNotNone(advance(body, step(body[0], direction), frozenset({(4, 5)})))
        direction = choose_direction(((10, 6),), (0, 1), {(11, 6)})
        self.assertNotEqual(step((10, 6), direction), (11, 6))

    def test_new_objects_during_simulated_round_are_collected_without_self_collision(self):
        rng = random.Random(12)
        body, direction, food = ((1, 6),), (1, 0), frozenset()
        collected = 0
        for tick in range(130):
            if tick % 4 == 0:
                free = [(x, y) for x in range(COLS) for y in range(ROWS) if (x, y) not in body]
                food = food | {rng.choice(free)}
            chosen = choose_direction(body, direction, food)
            self.assertIsNotNone(chosen)
            target = step(body[0], chosen)
            collected += target in food
            moved = advance(body, target, food)
            self.assertIsNotNone(moved)
            body, food = moved
            direction = chosen
        self.assertGreater(collected, 10)
        self.assertEqual(len(body), collected + 1)


class TrackerTests(unittest.TestCase):
    def tracker(self, cell=(1, 6), direction=(1, 0)):
        tracker = Tracker(center(cell), 0)
        tracker.heading = direction
        return tracker

    def test_countdown_does_not_invent_direction(self):
        tracker = Tracker(center((1, 6)), 0)
        tracker.observe(center((1, 6)), 1)
        self.assertIsNone(tracker.heading)
        self.assertFalse(tracker.near_center())
        point = center((1, 6))
        tracker.observe((point[0] + 7, point[1]), 1.1)
        self.assertEqual(tracker.heading, (1, 0))

    def test_multiple_cells_preserve_growth_and_body_order(self):
        tracker = self.tracker()
        tracker.see_food({(2, 6), (3, 6)})
        tracker.observe(center((4, 6)), 0.6)
        self.assertEqual(tracker.body, ((4, 6), (3, 6), (2, 6)))
        self.assertFalse(tracker.food)
        self.assertEqual(tracker.eaten, {(2, 6), (3, 6)})

    def test_stationary_countdown_is_excluded_from_speed_calibration(self):
        tracker = Tracker(center((1, 6)), 0)
        tracker.observe(center((1, 6)), 2)
        tracker.observe(center((1, 6)), 3)
        point = center((1, 6))
        tracker.observe((point[0] + 10, point[1]), 3.05)
        self.assertAlmostEqual(tracker.period, 0.4225)
        tracker.observe(center((3, 6)), 3.9)
        self.assertEqual(tracker.body[0], (3, 6))

    def test_body_history_wraps_at_edges(self):
        tracker = self.tracker((12, 3))
        tracker.see_food({(0, 3)})
        tracker.observe(center((0, 3)), 0.2)
        self.assertEqual(tracker.body, ((0, 3), (12, 3)))

    def test_immediate_and_queued_turn_are_verified_from_observed_position(self):
        for target, expected in (((1, 5), ((1, 5),)), ((2, 5), ((2, 5),))):
            with self.subTest(target=target):
                tracker = self.tracker()
                tracker.request((0, -1))
                tracker.observe(center(target), 0.4)
                self.assertEqual(tracker.body, expected)
                self.assertEqual(tracker.heading, (0, -1))
                self.assertIsNone(tracker.pending)
        tracker = self.tracker()
        tracker.request((0, -1))
        tracker.observe(center((2, 6)), 0.2)
        self.assertIsNotNone(tracker.pending)
        self.assertEqual(tracker.heading, (1, 0))
        tracker.observe(center((2, 5)), 0.4)
        self.assertIsNone(tracker.pending)

    def test_no_second_turn_before_first_is_confirmed(self):
        tracker = self.tracker()
        tracker.request((0, -1))
        with self.assertRaises(TrackingError):
            tracker.request((-1, 0))
        tracker = self.tracker()
        with self.assertRaises(TrackingError):
            tracker.request((-1, 0))

    def test_unknown_jump_or_uncommanded_turn_is_not_reset_to_one_segment(self):
        tracker = self.tracker()
        tracker.body = ((1, 6), (0, 6))
        for point in (center((7, 6)), center((1, 3))):
            with self.assertRaises(TrackingError):
                tracker.observe(point, 0.2)
            self.assertEqual(tracker.body, ((1, 6), (0, 6)))

    def test_temporarily_occluded_food_is_kept_and_respawn_can_grow_again(self):
        tracker = self.tracker()
        tracker.see_food({(2, 6)})
        tracker.see_food(set())
        tracker.observe(center((2, 6)), 0.2)
        self.assertEqual(len(tracker.body), 2)
        tracker.observe(center((4, 6)), 0.6)
        tracker.see_food({(2, 6)})
        self.assertIn((2, 6), tracker.food)
        self.assertNotIn((2, 6), tracker.eaten)


class GameFlowTests(unittest.TestCase):
    def runner(self):
        context = Mock()
        context.tasker.stopping = False
        return dream.DreamRunner(context)

    def clock(self, now):
        return (patch.object(dream.time, "monotonic", side_effect=lambda: now[0]),
                patch.object(dream.time, "sleep", side_effect=lambda dt: now.__setitem__(0, now[0] + dt)))

    def test_mirrored_duplicates_are_one_head_but_multiple_distinct_heads_are_rejected(self):
        runner = self.runner()
        runner.reco = Mock(return_value=SimpleNamespace(hit=True, filtered_results=[
            SimpleNamespace(box=(402, 441, 24, 34)), SimpleNamespace(box=(403, 441, 24, 34))]))
        self.assertEqual(runner.inspiration_head(object()), (414, 487))
        runner.reco.return_value.filtered_results.append(SimpleNamespace(box=(509, 524, 24, 34)))
        self.assertIsNone(runner.inspiration_head(object()))
        runner.reco.return_value.hit = False
        self.assertIsNone(runner.inspiration_head(object()))

    def test_result_needs_all_three_labels_in_the_result_regions(self):
        runner = self.runner()
        for values, expected in (([False], False), ([True, False], False),
                                 ([True, True, False], False), ([True, True, True], True)):
            runner.hit = Mock(side_effect=values)
            self.assertEqual(runner.inspiration_result(object()), expected)

    def test_short_swipe_has_real_success_check_and_cancellation(self):
        runner = self.runner()
        runner.context.run_action_direct.return_value = SimpleNamespace(success=True)
        runner.inspiration_turn((0, -1))
        name, action = runner.context.run_action_direct.call_args.args
        self.assertEqual(name, "Swipe")
        self.assertEqual(action.begin, [1157, 597, 1, 1])
        self.assertEqual(action.end, [[1157, 537, 1, 1]])
        self.assertEqual(action.duration, [80])
        runner.context.run_action_direct.return_value.success = False
        with self.assertRaises(dream.DreamError):
            runner.inspiration_turn((1, 0))
        runner.context.tasker.stopping = True
        runner.context.run_action_direct.reset_mock()
        with self.assertRaises(dream.DreamError):
            runner.inspiration_turn((1, 0))
        runner.context.run_action_direct.assert_not_called()

    def test_active_round_reads_new_food_rechecks_head_and_turns_before_result(self):
        runner = self.runner()
        frames = iter(["countdown", "moving", "fresh", "turn-confirmed", "result"])
        runner.image = Mock(side_effect=lambda: next(frames))
        runner.inspiration_head = Mock(side_effect=lambda frame: {
            "countdown": center((1, 6)), "moving": center((2, 6)),
            "fresh": center((2, 6)), "turn-confirmed": center((2, 5)), "result": None}[frame])
        runner.hit = Mock(return_value=True)
        runner.inspiration_food = Mock(return_value={(2, 5)})
        runner.inspiration_turn = Mock()
        runner.inspiration_result = Mock(side_effect=lambda frame: frame == "result")
        runner.wait = Mock(return_value=True)
        now = [0.0]
        with self.clock(now)[0], self.clock(now)[1]:
            runner.inspiration_play()
        runner.inspiration_food.assert_called_once_with("moving")
        runner.inspiration_turn.assert_called_once_with((0, -1))
        runner.wait.assert_called_once()
        self.assertEqual(runner.wait.call_args.args[0], runner.inspiration_result)

    def test_countdown_and_midgame_without_initialization_never_turn(self):
        for countdown in (True, False):
            runner = self.runner()
            runner.image = Mock(side_effect=["game", "game", "game", "result"])
            runner.inspiration_head = Mock(side_effect=lambda frame: None if frame == "result" else center((1, 6)))
            runner.hit = Mock(return_value=countdown)
            runner.inspiration_food = Mock(return_value=set())
            runner.inspiration_turn = Mock()
            runner.inspiration_result = Mock(return_value=True)
            runner.wait = Mock(return_value=True)
            now = [0.0]
            with self.clock(now)[0], self.clock(now)[1]:
                runner.inspiration_play()
            runner.inspiration_turn.assert_not_called()

    def test_unknown_trajectory_stops_controls_but_finishes_same_round(self):
        runner = self.runner()
        runner.image = Mock(side_effect=["countdown", "jump", "next", "result"])
        runner.inspiration_head = Mock(side_effect=[center((1, 6)), center((7, 6)), center((8, 6)), None])
        runner.hit = Mock(return_value=True)
        runner.inspiration_turn = Mock()
        runner.inspiration_food = Mock()
        runner.inspiration_result = Mock(return_value=True)
        runner.wait = Mock(return_value=True)
        now = [0.0]
        with self.clock(now)[0], self.clock(now)[1], patch.object(dream.logger, "warning") as warning:
            runner.inspiration_play()
        warning.assert_called_once()
        runner.inspiration_turn.assert_not_called()
        runner.inspiration_food.assert_not_called()

    def test_missing_result_has_one_deadline_and_never_restarts_game(self):
        runner = self.runner()
        runner.image = Mock(return_value="unknown")
        runner.inspiration_head = Mock(return_value=None)
        runner.inspiration_result = Mock(return_value=False)
        runner.button = Mock()
        now = [0.0]
        with self.clock(now)[0], self.clock(now)[1], self.assertRaises(dream.DreamError):
            runner.inspiration_play()
        self.assertLess(now[0], 90.1)
        runner.button.assert_not_called()

    def test_task_returns_via_space_and_reopens_seven_before_marking_completed(self):
        runner = self.runner()
        trace = Mock()
        for method in ("click", "wait", "button", "inspiration_play"):
            setattr(runner, method, Mock(return_value=(30, 32, 54, 55)))
            trace.attach_mock(getattr(runner, method), method)
        runner.seven_inspiration()
        self.assertEqual([call.args[0] for call in runner.button.call_args_list],
                         ["灵感拾趣入口", "七日妙想入口"])
        self.assertEqual([call[0] for call in trace.method_calls],
                         ["wait", "click", "wait", "button", "inspiration_play", "wait",
                          "click", "wait", "button", "wait"])
        runner.enter_seven = Mock()
        runner.seven_inspiration = Mock()
        completed = []
        candidates = iter([("拾趣", 0, None), ("全部完成", None, None)])
        runner.seven_current = Mock(side_effect=lambda done: completed.append(set(done)) or next(candidates))
        runner.seven_open_task = Mock()
        runner.seven()
        self.assertEqual(completed, [set(), {"拾趣"}])
        runner.seven_open_task.assert_not_called()

    def test_failed_return_does_not_continue_next_card(self):
        runner = self.runner()
        runner.enter_seven = Mock()
        runner.seven_current = Mock(return_value=("拾趣", 0, None))
        runner.seven_inspiration = Mock(side_effect=dream.DreamError("未返回织梦空间"))
        with self.assertRaises(dream.DreamError):
            runner.seven()
        runner.seven_current.assert_called_once()

    def test_return_arrow_requires_result_and_space_wait_rejects_loading(self):
        runner = self.runner()
        runner.wait = Mock(return_value=(30, 32, 54, 55))
        runner.click = Mock()
        runner.button = Mock()
        runner.inspiration_play = Mock()
        runner.seven_inspiration()
        result_back = runner.wait.call_args_list[2].args[0]
        runner.inspiration_result = Mock(return_value=False)
        runner.box = Mock(return_value=(30, 32, 54, 55))
        self.assertIsNone(result_back("game"))
        runner.box.assert_not_called()
        runner.inspiration_result.return_value = True
        self.assertEqual(result_back("result"), (30, 32, 54, 55))
        runner.box.assert_called_once_with("灵感拾趣返回按钮", "result")
        space_wait = runner.wait.call_args_list[3].args[0]
        runner.hit = Mock(return_value=True)
        runner.space_ready = Mock(return_value=True)
        self.assertFalse(space_wait("loading"))
        runner.space_ready.assert_not_called()
        runner.hit.return_value = False
        self.assertTrue(space_wait("space"))

    def test_assets_and_recognition_omit_explicit_defaults(self):
        pipeline = load_jsonc(ROOT / "assets/resource/pipeline/dreamweaver_space.json")
        for name, node in pipeline.items():
            if not name.startswith("灵感拾趣"):
                continue
            param = node["recognition"]["param"]
            self.assertNotIn("threshold", param)
            self.assertNotIn("index", param)
            self.assertNotIn("post_delay", node)
            for template in param.get("template", []):
                self.assertTrue((ROOT / "assets/resource/image" / template).is_file())


if __name__ == "__main__":
    unittest.main()
