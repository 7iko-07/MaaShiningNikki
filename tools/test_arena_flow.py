"""竞技场结束、错误传播及页面交接的离线回归测试。"""

import copy
import json
import subprocess
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from maa.controller import CustomController
from maa.custom_action import CustomAction
from maa.custom_recognition import CustomRecognition
from maa.define import MaaStatusEnum, Status, TaskDetail, LoggingLevelEnum
from maa.resource import Resource
from maa.tasker import Tasker

from tools.test_arena_weekly import ROOT, arena, load_action
from tools.validate_schema import load_jsonc

navigation = load_action("page_navigation")
ARENA = json.loads((ROOT / "assets/resource/pipeline/arena.json").read_text(encoding="utf-8"))


def task_detail(succeeded):
    return TaskDetail(1, "test", [], Status(
        MaaStatusEnum.succeeded if succeeded else MaaStatusEnum.failed))


class ArenaActionTests(unittest.TestCase):
    def setUp(self):
        self.action = arena.ArenaCompareAction()
        self.context = Mock()
        self.context.tasker.stopping = False
        self.context.get_node_data.return_value = {}
        self.context.override_next.return_value = True
        self.controller = self.context.tasker.controller
        self.controller.post_screencap.return_value.wait.return_value.succeeded = True
        self.controller.post_click.return_value.wait.return_value.succeeded = True
        self.context.run_recognition.return_value = SimpleNamespace(hit=False)
        self.context.run_task.return_value = task_detail(True)
        for patcher in (patch.object(arena, "in_settlement", return_value=False),
                        patch.object(arena.time, "sleep")):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.params = copy.deepcopy(ARENA["竞技场战力对比"]["action"]["param"]["custom_action_param"])
        self.params["max_rounds"] = 0

    def run_action(self, node="竞技场战力对比"):
        return self.action.run(self.context, SimpleNamespace(
            node_name=node, custom_action_param=json.dumps(self.params)))

    def test_both_strategies_finish_all_challenges_without_outer_wait(self):
        for node in ("竞技场战力对比", "竞技场稳守段位"):
            with self.subTest(node=node):
                self.action._read_challenge_times = Mock(return_value=3)
                self.action._challenge_once = Mock(return_value=arena.ChallengeOutcome.CHALLENGED)
                self.context.run_task.reset_mock()
                self.assertTrue(self.run_action(node))
                self.assertEqual(self.context.run_task.call_count, 3)
                self.context.override_next.assert_called_with(node, [])

    def test_zero_count_is_normal_stop(self):
        with patch.object(arena, "read_ocr_number", return_value=0):
            self.assertTrue(self.run_action())
        self.controller.post_click.assert_not_called()
        self.context.run_task.assert_not_called()

    def test_unknown_count_never_challenges_and_retries_are_finite(self):
        with patch.object(arena, "read_ocr_number", return_value=None) as read:
            self.assertFalse(self.run_action())
        self.assertEqual(read.call_count, 3)
        self.controller.post_click.assert_not_called()
        self.context.run_task.assert_not_called()

    def test_transient_count_ocr_failure_recovers(self):
        with patch.object(arena, "read_ocr_number", side_effect=[None, None, 0]) as read:
            self.assertTrue(self.run_action())
        self.assertEqual(read.call_count, 3)

    def test_missing_count_roi_fails_before_click(self):
        del self.params["剩余次数"]
        self.assertFalse(self.run_action())
        self.controller.post_click.assert_not_called()

    def test_known_stronger_opponents_stop_but_unreadable_power_fails(self):
        self.action._read_challenge_times = Mock(return_value=1)
        for own, opponent, expected in ((100, 200, True), (None, 200, False), (100, None, False)):
            with self.subTest(own=own, opponent=opponent):
                self.action._read_player_power = Mock(return_value=(own, object()))
                with patch.object(arena, "read_ocr_number", return_value=opponent):
                    self.assertEqual(self.run_action(), expected)
                self.controller.post_click.assert_not_called()
                self.context.run_task.assert_not_called()

    def test_override_failure_is_not_success_on_any_normal_exit(self):
        self.context.override_next.return_value = False
        self.action._read_challenge_times = Mock(return_value=0)
        self.assertFalse(self.run_action())
        self.action._read_challenge_times.return_value = 1
        for outcome in (arena.ChallengeOutcome.CHALLENGED, arena.ChallengeOutcome.STOPPED):
            self.action._challenge_once = Mock(return_value=outcome)
            self.assertFalse(self.run_action())
        with patch.object(arena, "in_settlement", return_value=True):
            self.assertFalse(self.run_action())

    def test_popup_outcomes_use_real_task_status(self):
        self.assertIs(self.action._handle_optional_ok_popup(self.context, self.controller, "ok", 0),
                      arena.PopupOutcome.ABSENT)
        self.context.run_task.assert_not_called()
        self.context.run_recognition.return_value.hit = True
        for result, expected in ((None, arena.PopupOutcome.FAILED),
                                 (task_detail(False), arena.PopupOutcome.FAILED),
                                 (task_detail(True), arena.PopupOutcome.HANDLED)):
            self.context.run_task.return_value = result
            self.assertIs(self.action._handle_optional_ok_popup(self.context, self.controller, "ok", 0),
                          expected)

    def test_failed_popup_stops_without_entering_result_flow(self):
        self.action._read_challenge_times = Mock(return_value=1)
        self.action._read_player_power = Mock(return_value=(100, object()))
        self.context.run_recognition.return_value.hit = True
        self.context.run_task.return_value = task_detail(False)
        with patch.object(arena, "read_ocr_number", return_value=10):
            self.assertFalse(self.run_action())
        self.context.run_task.assert_called_once_with("点击OK-竞技场")

    def test_failed_result_flow_stops_next_challenge(self):
        self.action._read_challenge_times = Mock(return_value=3)
        self.action._challenge_once = Mock(return_value=arena.ChallengeOutcome.CHALLENGED)
        self.context.run_task.return_value = task_detail(False)
        self.assertFalse(self.run_action())
        self.assertEqual(self.action._challenge_once.call_count, 1)

    def test_failed_opponent_click_never_enters_result_flow(self):
        self.action._read_challenge_times = Mock(return_value=1)
        self.action._read_player_power = Mock(return_value=(100, object()))
        self.controller.post_click.return_value.wait.return_value.succeeded = False
        with patch.object(arena, "read_ocr_number", return_value=10):
            self.assertFalse(self.run_action())
        self.context.run_task.assert_not_called()

    def configure_comparison(self, node="竞技场战力对比", force=False, rounds=1):
        self.params = copy.deepcopy(ARENA[node]["action"]["param"]["custom_action_param"])
        self.params["max_rounds"] = rounds
        self.context.get_node_data.return_value = {
            "attach": {"challenge_after_refresh_limit": force}}
        self.action._read_challenge_times = Mock(return_value=1)
        self.action._read_player_power = Mock(return_value=(100, object()))
        self.action._read_opponent_power = Mock(return_value=200)
        self.action._click_opponent = Mock(return_value=True)
        self.action._click_roi = Mock(return_value=True)
        self.action._handle_optional_ok_popup = Mock(return_value=arena.PopupOutcome.ABSENT)

    def test_switch_off_keeps_both_strategies_refresh_limit_and_never_counts_search(self):
        for node in ("竞技场战力对比", "竞技场稳守段位"):
            for power in (100, 200):
                with self.subTest(node=node, power=power):
                    self.configure_comparison(node, rounds=2)
                    self.action._read_opponent_power.return_value = power
                    with patch.object(arena, "logger") as log:
                        self.assertTrue(self.run_action(node))
                    self.assertEqual(self.action._click_roi.call_count, 2)
                    self.assertEqual(self.action._read_opponent_power.call_count,
                                     3 * len(self.params["area_b"]))
                    self.action._click_opponent.assert_not_called()
                    log.info.assert_called_once_with(
                        "竞技场挑战统计：自身战力大于对手 0 次，小于 0 次，相等 0 次")

    def test_enabled_switch_forces_last_rightmost_only_after_exact_refresh_limit(self):
        for node in ("竞技场战力对比", "竞技场稳守段位"):
            for power, symbol, summary in ((200, "<", "小于 1 次，相等 0 次"),
                                           (100, "=", "小于 0 次，相等 1 次")):
                with self.subTest(node=node, power=power):
                    self.configure_comparison(node, force=True, rounds=2)
                    self.action._read_opponent_power.return_value = power
                    with patch.object(arena, "logger") as log:
                        self.assertTrue(self.run_action(node))
                    self.assertEqual(self.action._click_roi.call_count, 2)
                    self.assertEqual(self.action._read_opponent_power.call_count,
                                     3 * len(self.params["area_b"]))
                    self.action._click_opponent.assert_called_once_with(
                        self.controller, self.params["area_b"][-1])
                    self.assertEqual([call.args[0] for call in log.info.call_args_list], [
                        f"自身战力 100 {symbol} 最右选手战力 {power}",
                        f"竞技场挑战统计：自身战力大于对手 0 次，{summary}"])

    def test_weaker_opponent_preserves_original_order_with_switch_on_or_off(self):
        for force in (False, True):
            for index, position in enumerate(("最左", "中间", "最右")):
                with self.subTest(force=force, position=position):
                    self.configure_comparison(force=force)
                    self.action._read_opponent_power.side_effect = [200] * index + [50]
                    with patch.object(arena, "logger") as log:
                        self.assertTrue(self.run_action())
                    self.action._click_roi.assert_not_called()
                    self.action._click_opponent.assert_called_once_with(
                        self.controller, self.params["area_b"][index])
                    log.info.assert_any_call(f"自身战力 100 > {position}选手战力 50")
                    log.info.assert_called_with(
                        "竞技场挑战统计：自身战力大于对手 1 次，小于 0 次，相等 0 次")

    def test_final_batch_still_prefers_weaker_leftmost_over_forced_rightmost(self):
        self.configure_comparison(force=True)
        self.action._read_opponent_power.side_effect = [200, 200, 200, 50]
        self.assertTrue(self.run_action())
        self.action._click_roi.assert_called_once()
        self.action._click_opponent.assert_called_once_with(self.controller, self.params["area_b"][0])

    def test_each_challenge_resets_refresh_budget_and_counts_only_launched_opponents(self):
        self.configure_comparison("竞技场稳守段位", force=True)
        self.action._read_challenge_times.return_value = 3
        self.action._read_opponent_power.side_effect = [200, 200, 200, 100, 200, 50]
        with patch.object(arena, "logger") as log:
            self.assertTrue(self.run_action("竞技场稳守段位"))
        self.assertEqual(self.action._click_roi.call_count, 3)
        self.assertEqual(self.action._click_opponent.call_count, 3)
        self.assertEqual(self.context.run_task.call_count, 3)
        log.info.assert_called_with(
            "竞技场挑战统计：自身战力大于对手 1 次，小于 1 次，相等 1 次")

    def test_ok_popup_retries_do_not_count_or_reset_refresh_budget(self):
        self.configure_comparison("竞技场稳守段位", force=True)
        self.action._handle_optional_ok_popup.side_effect = [
            arena.PopupOutcome.HANDLED, arena.PopupOutcome.ABSENT]
        with patch.object(arena, "logger") as log:
            self.assertTrue(self.run_action("竞技场稳守段位"))
        self.action._click_roi.assert_called_once()
        self.assertEqual(self.action._click_opponent.call_count, 2)
        self.assertEqual(len(log.info.call_args_list), 2)
        log.info.assert_called_with(
            "竞技场挑战统计：自身战力大于对手 0 次，小于 1 次，相等 0 次")

    def test_forced_challenge_never_bypasses_unreadable_power_or_failed_actions(self):
        for failure in ("own", "opponent", "refresh", "click", "popup"):
            with self.subTest(failure=failure):
                self.configure_comparison(force=True)
                if failure == "own":
                    self.action._read_player_power.return_value = (None, None)
                elif failure == "opponent":
                    self.action._read_opponent_power.return_value = None
                elif failure == "refresh":
                    self.action._click_roi.return_value = False
                elif failure == "click":
                    self.action._click_opponent.return_value = False
                else:
                    self.action._handle_optional_ok_popup.return_value = arena.PopupOutcome.FAILED
                with patch.object(arena, "logger") as log:
                    self.assertFalse(self.run_action())
                log.info.assert_called_once_with(
                    "竞技场挑战统计：自身战力大于对手 0 次，小于 0 次，相等 0 次")

    def test_cancel_or_settlement_during_last_power_read_prevents_forced_click(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                self.context.tasker.stopping = False
                self.configure_comparison(force=True, rounds=0)
                def read(*args):
                    self.context.tasker.stopping = cancel
                    return 200
                self.action._read_opponent_power.side_effect = read
                with patch.object(arena, "in_settlement", side_effect=[False, False, True]):
                    self.assertEqual(self.run_action(), not cancel)
                self.action._click_opponent.assert_not_called()

    def test_failure_cancellation_and_exception_after_launch_preserve_counts(self):
        for ending in ("failure", "cancel", "exception"):
            with self.subTest(ending=ending):
                self.context.tasker.stopping = False
                self.configure_comparison(force=True, rounds=0)
                self.action._read_challenge_times.return_value = 3
                def finish(*args):
                    if ending == "exception":
                        raise RuntimeError("test result failure")
                    self.context.tasker.stopping = ending == "cancel"
                    return task_detail(ending == "cancel")
                self.context.run_task.side_effect = finish
                with patch.object(arena, "logger") as log:
                    self.assertFalse(self.run_action())
                self.action._click_opponent.assert_called_once()
                log.info.assert_called_with(
                    "竞技场挑战统计：自身战力大于对手 0 次，小于 1 次，相等 0 次")

    def test_reentry_does_not_reuse_counts_from_previous_run(self):
        self.configure_comparison(force=True, rounds=0)
        self.assertTrue(self.run_action())
        self.action._read_challenge_times.return_value = 0
        with patch.object(arena, "logger") as log:
            self.assertTrue(self.run_action())
        log.info.assert_called_with(
            "竞技场挑战统计：自身战力大于对手 0 次，小于 0 次，相等 0 次")


class ArenaReturnActionTests(unittest.TestCase):
    def setUp(self):
        self.context = Mock()
        self.context.tasker.stopping = False
        self.controller = self.context.tasker.controller
        self.controller.post_screencap.return_value.wait.return_value.succeeded = True
        self.controller.post_click.return_value.wait.return_value.succeeded = True
        self.now = 0.0
        self.visible = lambda node: node == "返回竞技场战力对比页面"
        self.context.run_recognition.side_effect = self.recognize
        for patcher in (patch.object(arena.time, "monotonic", side_effect=lambda: self.now),
                        patch.object(arena.time, "sleep", side_effect=self.sleep)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def sleep(self, seconds):
        self.now += seconds

    def recognize(self, node, image):
        return SimpleNamespace(hit=self.visible(node), box=SimpleNamespace(
            x=int(self.now * 100), y=50, w=36, h=38))

    def run_action(self):
        return arena.ArenaReturnToCompareAction().run(self.context, SimpleNamespace(
            node_name="返回竞技场战力对比页面"))

    def test_retries_fresh_box_centers_until_destination_then_stops(self):
        self.visible = lambda node: node == "返回竞技场战力对比页面" or (
            node == "竞技场搭配完成确认" and self.now >= 2.5)
        self.assertTrue(self.run_action())
        points = [call.args for call in self.controller.post_click.call_args_list]
        self.assertEqual(len(points), 3)
        self.assertEqual(points[0], (18, 69))
        self.assertTrue(all(y == 69 for x, y in points))
        self.assertTrue(all(b[0] - a[0] >= 100 for a, b in zip(points, points[1:])))
        # 已到达时不再识别或点击仍可见的返回按钮。
        self.assertEqual(self.context.run_recognition.call_args.args[0], "竞技场搭配完成确认")
        self.context.run_task.assert_not_called()

    def test_destination_already_visible_never_clicks(self):
        self.visible = lambda node: True
        self.assertTrue(self.run_action())
        self.controller.post_click.assert_not_called()

    def test_ok_popup_precedes_return_and_shares_deadline(self):
        self.visible = lambda node: node != "竞技场搭配完成确认"
        self.assertFalse(self.run_action())
        self.assertEqual(self.now, 30)
        self.assertGreater(self.controller.post_click.call_count, 1)
        nodes = [call.args[0] for call in self.context.run_recognition.call_args_list]
        self.assertNotIn("返回竞技场战力对比页面", nodes)

    def test_no_recognized_button_never_blind_clicks_and_times_out(self):
        self.visible = lambda node: False
        self.assertFalse(self.run_action())
        self.assertEqual(self.now, 30)
        self.controller.post_click.assert_not_called()

    def test_ok_popup_closes_then_destination_completes(self):
        self.visible = lambda node: (
            node == "点击OK-竞技场" if self.now < 0.3
            else node == "竞技场搭配完成确认")
        self.assertTrue(self.run_action())
        self.controller.post_click.assert_called_once_with(18, 69)

    def test_capture_click_recognition_and_box_failures_stop(self):
        for failure in ("capture", "click", "recognition", "box"):
            with self.subTest(failure=failure):
                self.setUp_failure(failure)
                self.assertFalse(self.run_action())
                self.assertEqual(self.now, 0)

    def setUp_failure(self, failure):
        self.controller.post_screencap.return_value.wait.return_value.succeeded = failure != "capture"
        self.controller.post_click.return_value.wait.return_value.succeeded = failure != "click"
        self.context.run_recognition.side_effect = self.recognize
        if failure in ("recognition", "box"):
            self.context.run_recognition.side_effect = lambda node, image: (
                None if failure == "recognition" else SimpleNamespace(
                    hit=node == "返回竞技场战力对比页面", box=SimpleNamespace(w=0, h=0)))

    def test_stop_or_recognition_exceeding_deadline_never_clicks(self):
        self.context.tasker.stopping = True
        self.assertFalse(self.run_action())
        self.context.tasker.stopping = False
        def slow_recognition(node, image):
            self.now = 31
            return SimpleNamespace(hit=True)
        self.context.run_recognition.side_effect = slow_recognition
        self.assertFalse(self.run_action())
        self.controller.post_click.assert_not_called()


class RankPopupActionTests(unittest.TestCase):
    def setUp(self):
        self.context = Mock()
        self.context.tasker.stopping = False
        self.controller = self.context.tasker.controller
        self.controller.post_screencap.return_value.wait.return_value.succeeded = True
        self.controller.post_click.return_value.wait.return_value.succeeded = True
        self.now = 0.0
        self.frame = -1
        self.visible = lambda: True
        self.promotion_visible = lambda: False
        self.reward_visible = lambda: False
        self.entry = "竞技场排名提升页面"
        self.ready = True
        self.click_times = []
        self.clicked_nodes = []
        self.controller.post_screencap.side_effect = self.capture
        self.controller.post_click.side_effect = self.click
        self.context.run_recognition.side_effect = self.recognize
        for patcher in (patch.object(arena.time, "monotonic", side_effect=lambda: self.now),
                        patch.object(arena.time, "sleep", side_effect=self.sleep)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def sleep(self, seconds):
        self.now += seconds

    def capture(self):
        self.frame += 1
        return self.controller.post_screencap.return_value

    def click(self, x, y):
        self.click_times.append(self.now)
        self.clicked_nodes.append(self.last_popup)
        return self.controller.post_click.return_value

    def recognize(self, node, image):
        if node == "竞技场排名提升页面":
            hit = self.visible()
        elif node == "竞技场段位晋级页面":
            hit = self.promotion_visible()
        elif node == "竞技场挑战奖励页面":
            hit = self.reward_visible()
        else:
            hit = self.ready
        if hit and node in arena.ArenaCloseRankPopupAction.POPUP_NODES:
            self.last_popup = node
        return SimpleNamespace(
            hit=hit,
            box=SimpleNamespace(x=100 + self.frame, y=200, w=40, h=30))

    def run_action(self):
        return arena.ArenaCloseRankPopupAction().run(self.context, SimpleNamespace(
            node_name=self.entry,
            custom_action_param=json.dumps({"timeout": 10000, "max_clicks": 3})))

    def test_waits_for_animation_after_three_clicks(self):
        self.visible = lambda: self.now < 4
        self.assertTrue(self.run_action())
        self.assertEqual(len(self.click_times), 3)
        self.assertGreater(self.now, 4)
        self.assertTrue(all(b - a >= 1 for a, b in zip(self.click_times, self.click_times[1:])))
        # Each click follows the freshly recognized moving text, not the initial box.
        points = [call.args for call in self.controller.post_click.call_args_list]
        self.assertEqual(points[0], (120, 215))
        self.assertGreater(points[1][0], points[0][0])

    def test_transient_ocr_misses_do_not_finish_or_click(self):
        frames = [True, False, False, True, False, False, False]
        self.visible = lambda: frames[min(self.frame, len(frames) - 1)]
        self.assertTrue(self.run_action())
        self.assertEqual(self.frame, 6)
        self.assertEqual(len(self.click_times), 1)

    def test_natural_disappearance_needs_no_click(self):
        self.visible = lambda: False
        self.assertTrue(self.run_action())
        self.assertEqual(self.frame, 2)
        self.controller.post_click.assert_not_called()

    def test_persistent_text_times_out_after_only_three_clicks(self):
        self.assertFalse(self.run_action())
        self.assertEqual(len(self.click_times), 3)
        self.assertAlmostEqual(self.now, 10)

    def test_disappearance_without_arena_is_not_success(self):
        self.visible = lambda: False
        self.ready = False
        self.assertFalse(self.run_action())
        self.controller.post_click.assert_not_called()
        self.assertAlmostEqual(self.now, 10)

    def test_recognition_error_is_not_disappearance(self):
        self.context.run_recognition.side_effect = None
        self.context.run_recognition.return_value = None
        self.assertFalse(self.run_action())
        self.controller.post_click.assert_not_called()

    def test_failed_capture_or_click_propagates(self):
        self.controller.post_screencap.return_value.wait.return_value.succeeded = False
        self.assertFalse(self.run_action())
        self.controller.post_click.assert_not_called()
        self.controller.post_screencap.return_value.wait.return_value.succeeded = True
        self.controller.post_click.return_value.wait.return_value.succeeded = False
        self.assertFalse(self.run_action())
        self.assertEqual(len(self.click_times), 1)

    def test_cancellation_prevents_click(self):
        self.context.tasker.stopping = True
        self.assertFalse(self.run_action())
        self.controller.post_click.assert_not_called()

    def test_promotion_alone_uses_same_disappearance_check(self):
        self.entry = "竞技场段位晋级页面"
        self.visible = lambda: False
        self.promotion_visible = lambda: self.now < 2
        self.assertTrue(self.run_action())
        self.assertEqual(set(self.clicked_nodes), {"竞技场段位晋级页面"})
        self.assertGreaterEqual(self.now, 2.6)

    def test_rank_then_promotion_does_not_exit_on_visible_background(self):
        self.visible = lambda: self.now < 1
        self.promotion_visible = lambda: 1 <= self.now < 3
        self.assertTrue(self.run_action())
        self.assertEqual(set(self.clicked_nodes), set(arena.ArenaCloseRankPopupAction.POPUP_NODES))
        self.assertGreaterEqual(self.now, 3.6)

    def test_promotion_then_rank_does_not_exit_on_visible_background(self):
        self.entry = "竞技场段位晋级页面"
        self.promotion_visible = lambda: self.now < 1
        self.visible = lambda: 1 <= self.now < 3
        self.assertTrue(self.run_action())
        self.assertEqual(self.clicked_nodes[0], "竞技场段位晋级页面")
        self.assertIn("竞技场排名提升页面", self.clicked_nodes)
        self.assertGreaterEqual(self.now, 3.6)

    def test_alternating_popups_share_deadline_and_per_type_click_limits(self):
        self.visible = lambda: int(self.now / 1.2) % 2 == 0
        self.promotion_visible = lambda: not self.visible()
        self.assertFalse(self.run_action())
        self.assertAlmostEqual(self.now, 20)
        for node in arena.ArenaCloseRankPopupAction.POPUP_NODES:
            self.assertLessEqual(self.clicked_nodes.count(node), 3)
        self.assertTrue(all(b - a >= 1 for a, b in zip(self.click_times, self.click_times[1:])))

    def test_other_popup_recognition_error_is_not_clear_screen(self):
        self.visible = lambda: False
        recognize = self.recognize
        self.context.run_recognition.side_effect = lambda node, image: (
            None if node == "竞技场段位晋级页面" else recognize(node, image))
        self.assertFalse(self.run_action())
        self.controller.post_click.assert_not_called()

    def test_reward_after_either_animation_exits_without_arena_or_reward_click(self):
        for entry in arena.ArenaCloseRankPopupAction.POPUP_NODES:
            with self.subTest(entry=entry):
                self.entry = entry
                self.visible = lambda: False
                self.promotion_visible = lambda: False
                self.reward_visible = lambda: True
                self.ready = False
                self.frame = -1
                self.context.run_recognition.reset_mock()
                self.assertTrue(self.run_action())
                self.assertEqual(self.frame, 2)
                self.controller.post_click.assert_not_called()
                self.context.run_task.assert_not_called()
                nodes = [call.args[0] for call in self.context.run_recognition.call_args_list]
                self.assertEqual(nodes[-1], "竞技场挑战奖励页面")
                self.assertNotIn("竞技场挑战返回页面", nodes)

    def test_visible_animation_is_not_skipped_for_reward(self):
        self.reward_visible = lambda: True
        self.assertFalse(self.run_action())
        self.assertEqual(len(self.click_times), 3)
        nodes = [call.args[0] for call in self.context.run_recognition.call_args_list]
        self.assertNotIn("竞技场挑战奖励页面", nodes)

    def test_reward_recognition_error_is_not_clear_screen(self):
        self.visible = lambda: False
        recognize = self.recognize
        self.context.run_recognition.side_effect = lambda node, image: (
            None if node == "竞技场挑战奖励页面" else recognize(node, image))
        self.assertFalse(self.run_action())
        self.controller.post_click.assert_not_called()


class ScreenController(CustomController):
    """仅生成空白截图，通过识别回调模拟页面，不连接设备。"""
    def __init__(self):
        super().__init__()
        self.screen = ""
        self.on_click = lambda: None
        self.on_capture = lambda: None

    def connect(self):
        return True

    def request_uuid(self):
        return "arena-offline"

    def get_features(self):
        return 0

    def screencap(self):
        self.on_capture()
        return np.zeros((1280, 720, 3), dtype=np.uint8)

    def click(self, x, y):
        self.on_click()
        return True

    def unused(self, *args):
        return False

    start_app = stop_app = swipe = touch_down = touch_move = touch_up = unused
    click_key = input_text = key_down = key_up = unused


class ScreenRecognition(CustomRecognition):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def analyze(self, context, argv):
        if argv.node_name == self.controller.screen:
            return (10, 10, 20, 20)
        return None


class ScreenAction(CustomAction):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def run(self, context, argv):
        return self.callback(context, argv)


class ArenaPipelineTests(unittest.TestCase):
    """执行原生 Pipeline 调度，保留真实 next / JumpBack / Or，仅模拟画面和点击。"""
    def setUp(self):
        Tasker.set_stdout_level(LoggingLevelEnum.Off)
        self.resource = Resource()
        self.controller = ScreenController()
        self.assertTrue(self.controller.post_connection().wait().succeeded)
        self.tasker = Tasker()
        self.assertTrue(self.tasker.bind(self.resource, self.controller))
        self.assertTrue(self.resource.register_custom_recognition("screen", ScreenRecognition(self.controller)))
        self.assertTrue(self.resource.register_custom_action("screen_action", ScreenAction(self.advance)))
        self.transitions = {}
        self.executed = []
        self.nodes = {}
        for path in (ROOT / "assets/resource/pipeline").rglob("*.json"):
            self.nodes.update(load_jsonc(path))
        # Keep composite recognition and edges; replace image-dependent leaves.
        for name, node in self.nodes.items():
            recognition = node.get("recognition", "DirectHit")
            kind = recognition if isinstance(recognition, str) else recognition["type"]
            if kind not in ("DirectHit", "Or"):
                node["recognition"] = {"type": "Custom", "param": {"custom_recognition": "screen"}}
            node["action"] = {"type": "Custom", "param": {"custom_action": "screen_action"}}
            node["pre_delay"] = node["post_delay"] = 0
            node["rate_limit"] = 10
            node["timeout"] = 150
        self.assertTrue(self.resource.override_pipeline(self.nodes))

    def advance(self, context, argv):
        self.executed.append(argv.node_name)
        if argv.node_name in self.transitions:
            self.controller.screen = self.transitions[argv.node_name]
        return True

    def execute(self, entry, overrides=None):
        job = self.tasker.post_task(entry, overrides or {})
        deadline = time.monotonic() + 5
        while not job.done and time.monotonic() < deadline:
            time.sleep(.01)
        if not job.done:
            self.tasker.post_stop().wait()
            self.fail("Pipeline 未在测试期限内退出")
        return job.get()

    def test_challenge_leaves_finish_without_loading_wait(self):
        for node in ("竞技场战力对比", "竞技场稳守段位"):
            self.controller.screen = node
            self.assertTrue(self.execute(node).status.succeeded)

    def test_native_strategy_refresh_and_switch_overrides_remain_independent(self):
        interface = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
        option_name = "竞技场刷新上限后仍挑战"
        option = interface["option"][option_name]
        self.assertEqual(option["type"], "switch")
        self.assertEqual(option["default_case"], "No")
        task = next(task for task in interface["task"] if task["name"] == "竞技场")
        self.assertIn(option_name, task["option"])
        for strategy in interface["option"]["竞技场挑战策略"]["cases"]:
            for case in option["cases"]:
                for maximum in (1, 5, 99):
                    refresh = json.loads(json.dumps(
                        interface["option"]["竞技场刷新次数"]["pipeline_override"]
                    ).replace('"{max_rounds}"', str(maximum)))
                    for reverse in (False, True):
                        with self.subTest(strategy=strategy["name"], case=case["name"],
                                          maximum=maximum, reverse=reverse):
                            self.assertTrue(self.resource.override_pipeline(ARENA))
                            overrides = [strategy["pipeline_override"], refresh, case["pipeline_override"]]
                            for override in reversed(overrides) if reverse else overrides:
                                self.assertTrue(self.resource.override_pipeline(override))
                            for node in ("竞技场战力对比", "竞技场稳守段位"):
                                data = self.resource.get_node_data(node)
                                self.assertEqual(data["attach"]["challenge_after_refresh_limit"],
                                                 case["name"] == "Yes")
                                expected = copy.deepcopy(ARENA[node]["action"]["param"]["custom_action_param"])
                                expected["max_rounds"] = maximum
                                self.assertEqual(data["action"]["param"]["custom_action_param"], expected)
                            routed = self.resource.get_node_data("竞技场挑战策略分流")["next"][0]
                            target = routed["name"] if isinstance(routed, dict) else routed
                            self.assertEqual(target, strategy["pipeline_override"]["竞技场挑战策略分流"]["next"][0])

    def test_native_action_reads_switch_from_context_and_logs_summary(self):
        interface = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
        action = arena.ArenaCompareAction()
        self.assertTrue(self.resource.register_custom_action("arena_compare", action))
        for node in ("竞技场战力对比", "竞技场稳守段位"):
            for enabled in (False, True):
                with self.subTest(node=node, enabled=enabled):
                    self.assertTrue(self.resource.override_pipeline(ARENA))
                    self.assertTrue(self.resource.override_pipeline({node: {
                        "recognition": "DirectHit", "pre_delay": 0, "post_delay": 0,
                        "attach": {"challenge_after_refresh_limit": enabled}}}))
                    with patch.object(arena, "in_settlement", return_value=False), \
                         patch.object(arena.time, "sleep"), \
                         patch.object(arena, "logger") as log, \
                         patch.object(action, "_read_challenge_times", return_value=1), \
                         patch.object(action, "_read_player_power", return_value=(100, None)), \
                         patch.object(action, "_read_opponent_power", return_value=200), \
                         patch.object(action, "_click_roi", return_value=True) as refresh, \
                         patch.object(action, "_click_opponent", return_value=True) as click, \
                         patch.object(action, "_handle_optional_ok_popup", return_value=arena.PopupOutcome.ABSENT), \
                         patch.object(action, "_run_result_flow", return_value=True):
                        self.assertTrue(self.execute(node).status.succeeded)
                    self.assertEqual(refresh.call_count, 5)
                    self.assertEqual(click.call_count, int(enabled))
                    log.info.assert_called_with(
                        f"竞技场挑战统计：自身战力大于对手 0 次，小于 {int(enabled)} 次，相等 0 次")

    def test_ok_leaf_returns_to_parent_and_does_not_wait_for_loading(self):
        self.controller.screen = "点击OK-竞技场"
        self.transitions["点击OK-竞技场"] = "竞技场挑战返回页面"
        overrides = {"test_parent": {"recognition": "DirectHit", "pre_delay": 0, "post_delay": 0,
                      "next": ["[JumpBack]点击OK-竞技场", "竞技场挑战返回页面"], "timeout": 150}}
        self.assertTrue(self.execute("test_parent", overrides).status.succeeded)
        self.assertEqual(self.executed, ["点击OK-竞技场", "竞技场挑战返回页面"])

    def test_persistent_loading_times_out(self):
        self.controller.screen = "在加载页面"
        self.assertFalse(self.execute("竞技场挑战结果处理").status.succeeded)
        self.assertNotIn("在加载页面", self.executed)

    def test_outfit_next_waits_for_popup_while_next_remains_visible(self):
        for index in (1, 2, 3):
            with self.subTest(index=index):
                self.executed.clear()
                entry = f"确认推荐搭配下一步{index}"
                popup = f"确定竞技场影召{index}"
                self.controller.screen = entry
                clicks, captures = [], [0]
                self.controller.on_click = lambda: clicks.append(True)
                def capture():
                    if clicks:
                        captures[0] += 1
                        if captures[0] >= 3:
                            self.controller.screen = popup
                self.controller.on_capture = capture
                overrides = {
                    # 下一步始终可识别，弹窗出现后也不会消失。
                    entry: {"recognition": "DirectHit", "action": ARENA[entry]["action"]},
                    popup: {"next": []},
                }
                self.assertTrue(self.execute(entry, overrides).status.succeeded)
                self.assertEqual(len(clicks), 1)
                self.assertEqual(self.executed, [popup])
                self.assertGreaterEqual(captures[0], 3)

    def test_outfit_missing_popup_fails_without_reclicking_next(self):
        entry = "确认推荐搭配下一步1"
        self.controller.screen = entry
        clicks = []
        self.controller.on_click = lambda: clicks.append(True)
        overrides = {entry: {"recognition": "DirectHit", "action": ARENA[entry]["action"]}}
        self.assertFalse(self.execute(entry, overrides).status.succeeded)
        self.assertEqual(len(clicks), 1)
        self.assertNotIn("确定竞技场影召1", self.executed)

    def test_result_pages_finish_at_arena(self):
        self.controller.screen = "竞技场快速挑战结果页面"
        self.transitions = {"竞技场快速挑战结果页面": "竞技场挑战奖励页面",
                            "竞技场挑战奖励页面": "竞技场挑战返回页面"}
        self.assertTrue(self.execute("竞技场挑战结果处理").status.succeeded)
        self.assertEqual(self.executed, ["竞技场挑战结果处理", "竞技场快速挑战结果页面",
                                         "竞技场挑战奖励页面", "竞技场挑战返回页面"])

    def test_rank_popup_precedes_visible_arena_after_rewards(self):
        self.controller.screen = "竞技场挑战奖励页面"
        self.transitions = {"竞技场挑战奖励页面": "竞技场排名提升页面",
                            "竞技场排名提升页面": "竞技场挑战返回页面"}
        # The arena can remain visible behind the translucent rank popup.
        overrides = {"竞技场挑战返回页面": {"recognition": "DirectHit"}}
        self.assertTrue(self.execute("竞技场挑战奖励页面", overrides).status.succeeded)
        self.assertEqual(self.executed, ["竞技场挑战奖励页面", "竞技场排名提升页面",
                                         "竞技场挑战返回页面"])

    def test_rank_popup_action_failure_stops_result_flow(self):
        self.controller.screen = "竞技场挑战奖励页面"
        self.transitions = {"竞技场挑战奖励页面": "竞技场排名提升页面"}
        self.assertTrue(self.resource.register_custom_action(
            "failed_close", ScreenAction(lambda context, argv: False)))
        overrides = {
            "竞技场排名提升页面": {"action": {"type": "Custom", "param": {
                "custom_action": "failed_close"}}},
            "竞技场挑战返回页面": {"recognition": "DirectHit"},
        }
        self.assertFalse(self.execute("竞技场挑战奖励页面", overrides).status.succeeded)
        self.assertNotIn("竞技场挑战返回页面", self.executed)

    def test_promotion_precedes_visible_arena_after_rewards(self):
        self.controller.screen = "竞技场挑战奖励页面"
        self.transitions = {"竞技场挑战奖励页面": "竞技场段位晋级页面",
                            "竞技场段位晋级页面": "竞技场挑战返回页面"}
        overrides = {"竞技场挑战返回页面": {"recognition": "DirectHit"}}
        self.assertTrue(self.execute("竞技场挑战奖励页面", overrides).status.succeeded)
        self.assertEqual(self.executed, ["竞技场挑战奖励页面", "竞技场段位晋级页面",
                                         "竞技场挑战返回页面"])

    def test_reward_after_either_animation_precedes_visible_arena(self):
        for entry in arena.ArenaCloseRankPopupAction.POPUP_NODES:
            with self.subTest(entry=entry):
                self.executed = []
                self.controller.screen = entry
                self.transitions = {entry: "竞技场挑战奖励页面",
                                    "竞技场挑战奖励页面": "竞技场挑战返回页面"}
                overrides = {"竞技场挑战返回页面": {"recognition": "DirectHit"}}
                self.assertTrue(self.execute(entry, overrides).status.succeeded)
                self.assertEqual(self.executed, [entry, "竞技场挑战奖励页面", "竞技场挑战返回页面"])

    def test_reward_close_failure_after_animation_stops_flow(self):
        self.controller.screen = "竞技场段位晋级页面"
        self.transitions = {"竞技场段位晋级页面": "竞技场挑战奖励页面"}
        self.assertTrue(self.resource.register_custom_action(
            "failed_reward_close", ScreenAction(lambda context, argv: False)))
        overrides = {
            "竞技场挑战奖励页面": {"action": {"type": "Custom", "param": {
                "custom_action": "failed_reward_close"}}},
            "竞技场挑战返回页面": {"recognition": "DirectHit"},
        }
        self.assertFalse(self.execute("竞技场段位晋级页面", overrides).status.succeeded)
        self.assertNotIn("竞技场挑战返回页面", self.executed)

    def test_delayed_weekly_page_is_processed_before_dispatch(self):
        captures = [0]
        def capture():
            captures[0] += 1
            if captures[0] == 3:
                self.controller.screen = "竞技场上周成绩页面"
        self.controller.on_capture = capture
        self.transitions = {"竞技场上周成绩页面": "竞技场每周奖励页面",
                            "竞技场每周奖励页面": "竞技场本周搭配主题页面",
                            "竞技场本周搭配主题页面": "竞技场挑战返回页面"}
        overrides = {"竞技场搭配分流": {"next": []}}
        self.assertTrue(self.execute("竞技场每周页面处理", overrides).status.succeeded)
        self.assertGreater(self.executed.index("竞技场搭配分流"),
                           self.executed.index("竞技场本周搭配主题页面"))

    def test_unready_weekly_page_does_not_dispatch(self):
        self.controller.screen = "unknown"
        self.assertFalse(self.execute("竞技场每周页面处理").status.succeeded)
        self.assertNotIn("竞技场搭配分流", self.executed)

    def test_navigation_hands_off_each_weekly_page_before_closing_it(self):
        self.assertTrue(self.resource.register_custom_action("page_navigate", navigation.PageNavigateAction()))
        self.assertTrue(self.resource.register_custom_action("navigation_click", navigation.NavigationClickAction()))
        for page in ("竞技场上周成绩页面", "竞技场每周奖励页面", "竞技场本周搭配主题页面"):
            with self.subTest(page=page):
                self.executed.clear()
                self.controller.screen = "导航在独自页面"
                self.controller.on_click = lambda: setattr(self.controller, "screen", page)
                overrides = {
                    "竞技场任务链准备": {"action": ARENA["竞技场任务链准备"]["action"], "next": []},
                    "竞技场导航点击入口": {"action": ARENA["竞技场导航点击入口"]["action"]},
                    "导航点击钻石竞技场": {"recognition": "DirectHit"},
                }
                self.assertTrue(self.execute("竞技场任务链准备", overrides).status.succeeded)
                self.assertEqual(self.controller.screen, page)
                self.assertNotIn(page, self.executed)


class NativePipelineProcessTests(unittest.TestCase):
    def test_native_pipeline_in_separate_client_process(self):
        # Importing maa.agent selects server mode. Native Tasker tests need a
        # fresh client process, independent of other offline test imports.
        code = (
            "from maa.library import Library; Library.version(); "
            "import unittest, sys; "
            "from tools.test_arena_flow import ArenaPipelineTests; "
            "suite = unittest.defaultTestLoader.loadTestsFromTestCase(ArenaPipelineTests); "
            "result = unittest.TextTestRunner(verbosity=2).run(suite); "
            "sys.exit(not result.wasSuccessful())"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, encoding="utf-8", errors="replace", timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([
        loader.loadTestsFromTestCase(ArenaActionTests),
        loader.loadTestsFromTestCase(ArenaReturnActionTests),
        loader.loadTestsFromTestCase(RankPopupActionTests),
        loader.loadTestsFromTestCase(NativePipelineProcessTests),
    ])


if __name__ == "__main__":
    unittest.main()
