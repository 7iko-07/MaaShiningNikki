"""结算跨界、无红点分流及分享路径隔离的离线回归。"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.define import Rect
from maa.tasker import Tasker
from tools.test_time_corridor_challenge import ROOT, module as clothes


def load(name):
    package = "_workflow_guard_tests"
    if package not in sys.modules:
        parent = types.ModuleType(package)
        parent.__path__ = [str(ROOT / "agent/custom")]
        sys.modules[package] = parent
    spec = importlib.util.spec_from_file_location(f"{package}.{name}", ROOT / f"agent/custom/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with patch.object(AgentServer, "register_custom_action", return_value=True):
        spec.loader.exec_module(module)
    return module


stair = load("memory_stairway")
manicure = load("manicure")


class SettlementTests(unittest.TestCase):
    def test_beijing_boundaries_and_utc_equivalents(self):
        for value, expected in [("2026-10-18T22:59:59+08:00", False),
                                ("2026-10-18T23:00:00+08:00", True),
                                ("2026-10-19T04:59:59+08:00", True),
                                ("2026-10-19T05:00:00+08:00", False),
                                ("2026-10-20T00:00:00+08:00", False)]:
            now = datetime.fromisoformat(value)
            self.assertEqual(stair.in_settlement(now), expected)
            self.assertEqual(stair.in_settlement(now.astimezone(timezone.utc)), expected)

    def test_entry_reentry_restores_navigation_and_override_failure_propagates(self):
        context = Mock()
        context.tasker.stopping = False
        context.override_next.return_value = True
        argv = SimpleNamespace(node_name="心阶任务链入口")
        action = stair.MemoryStairwayGuardAction()
        for settling, successors in [(True, []), (False, ["心阶任务链准备"])]:
            with patch.object(stair, "in_settlement", return_value=settling):
                self.assertTrue(action.run(context, argv))
                context.override_next.assert_called_with(argv.node_name, successors)
        context.override_next.return_value = False
        with patch.object(stair, "in_settlement", return_value=True):
            self.assertFalse(action.run(context, argv))

    def test_crossing_settlement_before_retry_does_not_click_again(self):
        action = stair.MemoryStairwayChallengeAction()
        context = Mock()
        context.tasker.stopping = False
        context.override_next.return_value = True
        argv = SimpleNamespace(node_name="点击心阶-开始挑战", box=Rect(10, 20, 30, 40),
                               custom_action_param=json.dumps({"timeout": 0, "max_retries": 3,
                                                               "after_click_delay": 0}))
        with patch.object(stair, "in_settlement", side_effect=[False, False, True]), \
             patch.object(action, "_screencap", return_value=np.zeros((4, 4, 3))), \
             patch.object(action, "_image_changed", return_value=(False, 0)), \
             patch("time.sleep"):
            self.assertTrue(action.run(context, argv))
        context.tasker.controller.post_click.assert_called_once_with(25, 40)
        context.override_next.assert_called_with(argv.node_name, [])

    def test_already_settling_or_cancelled_never_starts_challenge(self):
        context = Mock()
        context.tasker.stopping = False
        context.override_next.return_value = True
        argv = SimpleNamespace(node_name="点击心阶-开始挑战")
        with patch.object(stair, "in_settlement", return_value=True):
            self.assertTrue(stair.MemoryStairwayChallengeAction().run(context, argv))
        context.tasker.controller.post_click.assert_not_called()
        context.tasker.stopping = True
        self.assertFalse(stair.MemoryStairwayChallengeAction().run(context, argv))


class StairwayPopupTests(unittest.TestCase):
    TITLE = "导航在心阶页面"
    SEASON = "心阶新赛季开始弹窗"
    KNOWN = "心阶我知道了弹窗"
    PARTS = "心阶部件制作提示"
    REWARD = "心阶快速挑战奖励弹窗"
    READY = "点击真我之境"

    def simulate(self, frames, *, fail_node=None, click_success=True, cancel=False,
                 slow=False, timeout=2000, capture_success=True, after_quick_challenge=False):
        context = Mock()
        context.tasker.stopping = False
        now = [0.0]
        captures, clicks = [], []
        controller = context.tasker.controller

        def capture():
            frame = frames[min(len(captures), len(frames) - 1)]
            captures.append(frame)
            controller.cached_image = frame
            return SimpleNamespace(wait=lambda: SimpleNamespace(status=SimpleNamespace(succeeded=capture_success)))

        def recognize(node, image):
            if slow:
                now[0] += .6
            if node == fail_node:
                return None
            return SimpleNamespace(hit=node in image)

        def click(kind, param):
            self.assertEqual(kind, "Click")
            self.assertEqual(param.target, (613, 48, 1, 1))
            clicks.append((now[0], controller.cached_image))
            if cancel:
                context.tasker.stopping = True
            return SimpleNamespace(success=click_success)

        controller.post_screencap.side_effect = capture
        context.run_recognition.side_effect = recognize
        context.run_action_direct.side_effect = click
        argv = SimpleNamespace(custom_action_param=json.dumps({
            "timeout": timeout, "after_quick_challenge": after_quick_challenge}))
        with patch.object(stair.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(stair.time, "sleep", side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)):
            success = stair.MemoryStairwayPopupsAction().run(context, argv)
        return success, context, captures, clicks, now[0]

    def test_both_popup_labels_in_one_frame_produce_one_blank_click(self):
        page = {self.TITLE}
        success, _, captures, clicks, _ = self.simulate([page | {self.SEASON, self.KNOWN}, page, page])
        self.assertTrue(success)
        self.assertEqual(len(clicks), 1)
        self.assertEqual(len(captures), 3)

    def test_sequential_popups_are_rechecked_and_clicks_stay_one_second_apart(self):
        page = {self.TITLE}
        success, _, _, clicks, _ = self.simulate(
            [page | {self.SEASON}] + [page | {self.KNOWN}] * 4 + [page, page])
        self.assertTrue(success)
        self.assertEqual(len(clicks), 2)
        self.assertGreaterEqual(clicks[1][0] - clicks[0][0], 1)
        self.assertIn(self.SEASON, clicks[0][1])
        self.assertIn(self.KNOWN, clicks[1][1])

    def test_clear_requires_two_new_consecutive_frames(self):
        page = {self.TITLE}
        success, _, captures, clicks, _ = self.simulate([page, page | {self.KNOWN}, page, page])
        self.assertTrue(success)
        self.assertEqual(len(captures), 4)
        self.assertEqual(len(clicks), 1)

    def test_no_popup_never_clicks_and_does_not_use_visible_quick_challenge_as_evidence(self):
        success, context, captures, clicks, _ = self.simulate([{self.TITLE, "点击快速挑战"}])
        self.assertTrue(success)
        self.assertEqual(len(captures), 2)
        self.assertEqual(clicks, [])
        self.assertNotIn("点击快速挑战", [call.args[0] for call in context.run_recognition.call_args_list])

    def test_loading_or_missing_title_resets_clear_count_and_never_clicks(self):
        page = {self.TITLE}
        for obscured in (set(), {self.TITLE, "在加载页面"}):
            success, _, captures, clicks, _ = self.simulate([page, obscured, page, page])
            self.assertTrue(success)
            self.assertEqual(len(captures), 4)
            self.assertEqual(clicks, [])
        for blocked in ({self.KNOWN}, {self.TITLE, self.KNOWN, "在加载页面"}):
            success, _, _, clicks, _ = self.simulate([blocked])
            self.assertFalse(success)
            self.assertEqual(clicks, [])

    def test_persistent_popup_times_out_without_extending_the_deadline(self):
        success, _, _, clicks, elapsed = self.simulate([{self.TITLE, self.KNOWN}])
        self.assertFalse(success)
        self.assertEqual(len(clicks), 2)
        self.assertAlmostEqual(elapsed, 2)

    def test_capture_recognition_click_failure_cancel_and_slow_recognition_fail_closed(self):
        for kwargs in ({"fail_node": self.KNOWN}, {"capture_success": False}, {"slow": True},
                       {"click_success": False}, {"cancel": True}):
            with self.subTest(kwargs=kwargs):
                success, _, _, clicks, _ = self.simulate([{self.TITLE, self.KNOWN}], **kwargs)
                self.assertFalse(success)
                self.assertLessEqual(len(clicks), 1)
                if "fail_node" in kwargs or "slow" in kwargs or "capture_success" in kwargs:
                    self.assertEqual(clicks, [])

    def test_quick_challenge_stacked_popups_close_before_entering_next_stage(self):
        page = {self.TITLE, self.READY}
        success, _, captures, clicks, _ = self.simulate(
            [page | {self.PARTS, self.REWARD}] + [page | {self.REWARD}] * 4 + [page, page],
            after_quick_challenge=True)
        self.assertTrue(success)
        self.assertEqual(len(clicks), 2)
        self.assertGreaterEqual(clicks[1][0] - clicks[0][0], 1)
        self.assertIn(self.PARTS, clicks[0][1])
        self.assertNotIn(self.PARTS, clicks[1][1])
        self.assertEqual(captures[-2:], [page, page])

    def test_quick_challenge_optional_popups_and_no_rewards(self):
        page = {self.TITLE, self.READY}
        for popup in (self.PARTS, self.REWARD, None):
            with self.subTest(popup=popup):
                frames = [page | {popup}, page, page] if popup else [page, page]
                success, _, _, clicks, _ = self.simulate(frames, after_quick_challenge=True)
                self.assertTrue(success)
                self.assertEqual(len(clicks), int(popup is not None))

    def test_quick_challenge_missing_ready_entry_waits_without_blind_clicks(self):
        success, _, _, clicks, elapsed = self.simulate([{self.TITLE}], after_quick_challenge=True)
        self.assertFalse(success)
        self.assertEqual(clicks, [])
        self.assertAlmostEqual(elapsed, 2)

    def test_quick_challenge_popup_failure_and_timeout_stop_the_task(self):
        for kwargs in ({}, {"fail_node": self.REWARD}, {"click_success": False},
                       {"capture_success": False}, {"cancel": True}):
            with self.subTest(kwargs=kwargs):
                success, _, _, _, _ = self.simulate(
                    [{self.TITLE, self.REWARD}], after_quick_challenge=True, **kwargs)
                self.assertFalse(success)

    def test_quick_challenge_late_popup_resets_clear_confirmation(self):
        page = {self.TITLE, self.READY}
        success, _, captures, clicks, _ = self.simulate(
            [{self.TITLE}, page, page | {self.REWARD}, page, page], after_quick_challenge=True)
        self.assertTrue(success)
        self.assertEqual(len(captures), 5)
        self.assertEqual(len(clicks), 1)


class ManicureDispatchTests(unittest.TestCase):
    def simulate(self, frames, *, stable=True, failed_red=False, slow=False, custom_param='{"timeout":1000}'):
        context = Mock()
        context.tasker.stopping = False
        context.override_next.return_value = True
        now = [0.0]
        captures = []
        controller = context.tasker.controller

        def capture():
            frame = frames[min(len(captures), len(frames) - 1)]
            captures.append(frame)
            controller.cached_image = frame
            return SimpleNamespace(wait=lambda: SimpleNamespace(status=SimpleNamespace(succeeded=True)))

        def recognize(node, frame):
            if slow:
                now[0] += .6
            if failed_red and node == "点击话题任务":
                return None
            return SimpleNamespace(hit=node in frame)

        context.wait_freezes.return_value = stable
        controller.post_screencap.side_effect = capture
        context.run_recognition.side_effect = recognize
        argv = SimpleNamespace(node_name="美甲话题任务分流", custom_action_param=custom_param)
        with patch.object(manicure.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(manicure.time, "sleep", side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)):
            success = manicure.ManicureTopicDispatchAction().run(context, argv)
        controller.post_click.assert_not_called()
        context.run_action_direct.assert_not_called()
        return success, context, captures

    def test_no_red_requires_two_consecutive_ready_frames(self):
        page = {"美甲话题页面就绪"}
        success, context, captures = self.simulate([page, {"在加载页面"}, page, page])
        self.assertTrue(success)
        self.assertEqual(len(captures), 4)
        context.override_next.assert_called_with("美甲话题任务分流", [])

    def test_red_has_priority_even_after_one_absent_frame(self):
        page = {"美甲话题页面就绪"}
        success, context, captures = self.simulate([page, page | {"点击话题任务"}])
        self.assertTrue(success)
        self.assertEqual(len(captures), 2)
        context.override_next.assert_called_with("美甲话题任务分流", ["点击话题任务"])

    def test_framework_null_custom_parameters_use_defaults(self):
        success, context, captures = self.simulate([{"美甲话题页面就绪"}], custom_param="null")
        self.assertTrue(success)
        self.assertEqual(len(captures), 2)
        context.override_next.assert_called_with("美甲话题任务分流", [])

    def test_unknown_loading_unstable_and_failed_recognition_are_not_success(self):
        for frames, kwargs in [([set()], {}), ([{"在加载页面", "美甲话题页面就绪"}], {}),
                               ([{"美甲周期奖励确认", "美甲话题页面就绪"}], {}),
                               ([{"美甲话题页面就绪"}], {"stable": False}),
                               ([{"美甲话题页面就绪"}], {"failed_red": True}),
                               ([{"美甲话题页面就绪"}], {"slow": True})]:
            with self.subTest(kwargs=kwargs, frames=frames):
                success, context, _ = self.simulate(frames, **kwargs)
                self.assertFalse(success)
                self.assertNotIn(unittest.mock.call("美甲话题任务分流", []), context.override_next.call_args_list)


class ShareFlowTests(unittest.TestCase):
    def test_left_and_right_share_chains_have_independent_endings(self):
        share = json.loads((ROOT / "assets/resource/pipeline/share.json").read_text(encoding="utf-8"))
        heart = json.loads((ROOT / "assets/resource/pipeline/hearts_door.json").read_text(encoding="utf-8"))
        self.assertEqual(share["点击分享"]["recognition"]["param"]["roi"], [2, 97, 223, 1183])
        self.assertEqual(share["识别到无分享奖励"]["recognition"]["param"]["roi"], [2, 97, 223, 1183])
        self.assertEqual(heart["心之门点击分享"]["recognition"]["param"]["roi"], [445, 1107, 275, 173])
        self.assertEqual(share["识别到无分享奖励"]["next"], ["stop"])
        node = "心之门点击分享"
        visited = []
        while node != "stop":
            self.assertNotIn(node, visited)
            visited.append(node)
            self.assertNotIn(node, share)
            node = heart[node]["next"][0]
        self.assertEqual(visited, ["心之门点击分享", "心之门点击qq", "心之门关闭qq",
                                   "心之门点击分享奖励-空白", "点击谜之海-感应第二次-空白"])
        interface = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
        cases = interface["option"]["是否在心之门这里分享"]["cases"]
        disabled = next(case for case in cases if case["name"] == "No")
        self.assertNotIn("心之门点击分享", disabled["pipeline_override"]["点击谜之海-感应第二次"]["next"])


class FractionTests(unittest.TestCase):
    def detail(self, texts, box_type=Rect):
        return SimpleNamespace(hit=True, all_results=[SimpleNamespace(
            text=text, box=box_type(x, y, w, h) if box_type is Rect else box_type((x, y, w, h)), score=score)
            for text, x, y, w, h, score in texts])

    def test_complete_split_and_over_capacity_stamina(self):
        for texts, expected in [([("55/140", 133, 7, 67, 23, .99)], 55),
                                ([("55", 133, 10, 27, 17, .99), ("140", 162, 7, 38, 23, .99)], 55),
                                ([("193", 130, 10, 25, 20, .99), ("/", 155, 10, 7, 20, .99),
                                  ("140", 162, 7, 38, 23, .99)], 193),
                                ([("93", 132, 10, 28, 17, .99), ("/140", 160, 7, 41, 23, .99)], 93),
                                ([("93/", 132, 7, 35, 23, .99), ("140", 167, 7, 38, 23, .99)], 93),
                                ([("０／１４０", 133, 7, 67, 23, .99)], 0)]:
            for box_type in (Rect, list, tuple):
                with self.subTest(texts=texts, box_type=box_type):
                    self.assertEqual(clothes._fraction_value(self.detail(texts, box_type)), expected)

    def test_missing_digits_wrong_rows_ambiguous_and_low_confidence_are_rejected(self):
        for texts in [[("5", 134, 10, 12, 17, .99), ("厂", 146, 13, 16, 9, .8), ("140", 162, 7, 38, 23, .99)],
                      [("55140", 133, 7, 67, 23, .99)],
                      [("55", 133, 10, 27, 17, .99), ("140", 162, 40, 38, 23, .99)],
                      [("55", 133, 10, 27, 17, .4), ("140", 162, 7, 38, 23, .99)],
                      [("9", 132, 10, 10, 17, .99), ("3", 142, 10, 10, 17, .99),
                       ("/140", 160, 7, 41, 23, .99)],
                      [("93", 132, 10, 28, 17, .99), ("/0", 160, 7, 41, 23, .99)],
                      [("55/140", 133, 7, 67, 23, .99), ("60/140", 240, 7, 67, 23, .99)]]:
            for box_type in (Rect, list, tuple):
                with self.subTest(texts=texts, box_type=box_type):
                    self.assertIsNone(clothes._fraction_value(self.detail(texts, box_type)))

    def test_invalid_box_returns_no_value(self):
        for box in (None, [], [1, 2, 3], [1, 2, 3, 0], [1, 2, 3, -1],
                    [-1, 2, 3, 4], [1, 2, "3", 4], [1, 2, 3.5, 4], [1, 2, True, 4], object()):
            with self.subTest(box=box):
                detail = self.detail([("55", 133, 10, 27, 17, .99), ("140", 162, 7, 38, 23, .99)])
                detail.all_results[0].box = box
                self.assertIsNone(clothes._fraction_value(detail))

    def test_logged_split_boxes_use_sdk_parser_and_whole_line_fallback(self):
        # The actual 02:40:18 result has list boxes and a one-pixel overlap.
        raw = {"all": [{"text": "93", "box": [132, 10, 28, 17], "score": .998662},
                       {"text": "/140", "box": [159, 7, 41, 23], "score": .991208}]}
        all_results, _, _ = Tasker._parse_recognition_raw_detail(None, "OCR", raw)
        self.assertIsInstance(all_results[0].box, list)
        split = SimpleNamespace(hit=True, all_results=all_results)
        self.assertIsNone(clothes._fraction_value(split))
        for text, expected in (("93/140", 93), ("93140", None)):
            with self.subTest(whole_line=text):
                context = Mock()
                image = object()
                context.run_recognition_direct.side_effect = [split,
                    self.detail([(text, 82, 0, 145, 40, .99)], list)]
                self.assertEqual(clothes._read_fraction(context, image, [82, 0, 145, 40]), expected)
                calls = context.run_recognition_direct.call_args_list
                self.assertFalse(calls[0].args[1].only_rec)
                self.assertTrue(calls[1].args[1].only_rec)
                self.assertIs(calls[0].args[2], image)
                self.assertIs(calls[1].args[2], image)

    def test_failed_detection_can_use_whole_line_without_guessing(self):
        context = Mock()
        context.run_recognition_direct.side_effect = [SimpleNamespace(hit=False),
            self.detail([("55/140", 82, 0, 145, 40, .99)])]
        self.assertEqual(clothes._read_fraction(context, None, [82, 0, 145, 40]), 55)
        calls = context.run_recognition_direct.call_args_list
        self.assertFalse(calls[0].args[1].only_rec)
        self.assertTrue(calls[1].args[1].only_rec)


class PreparedClothingValuesTests(unittest.TestCase):
    def setUp(self):
        self.node = "批量挑战心灵迷宫制衣材料"
        pipeline = json.loads((ROOT / "assets/resource/pipeline/Make_the_target_clothes.json").read_text(encoding="utf-8"))
        self.params = pipeline[self.node]["action"]["param"]["custom_action_param"]
        self.context = Mock()
        self.context.tasker.stopping = False
        self.context.override_pipeline.return_value = True
        self.context.override_next.return_value = True
        self.action = clothes.MakeTargetClothesBatchChallengeAction()
        self.action.clear_task_state(54321, "heart_maze")
        self.addCleanup(self.action.clear_task_state, 54321, "heart_maze")
        self.params["prepared_values"] = {"task_id": 54321, "mode": "heart_maze",
                                          "values": [55, 8, 30], "created": 100}

    def run_action(self):
        argv = SimpleNamespace(node_name=self.node, task_detail=SimpleNamespace(task_id=54321),
                               custom_action_param=json.dumps(self.params))
        return self.action.run(self.context, argv)

    def test_initial_values_are_consumed_once_and_quantity_is_checked_only_after_challenge(self):
        with patch.object(clothes.time, "monotonic", return_value=100.1), \
             patch.object(self.action, "_read_round_values") as read_initial, \
             patch.object(self.action, "_has_sufficient_quantity", return_value=True) as quantity, \
             patch.object(self.action, "_read_required_number", return_value=7) as after, \
             patch.object(self.action, "_click_roi") as click, patch.object(clothes.time, "sleep"):
            self.assertTrue(self.run_action())
        read_initial.assert_not_called()
        quantity.assert_called_once()
        after.assert_called_once()
        self.assertEqual(click.call_args_list[0].args[1], self.params["multi_button"])
        cleared = self.context.override_pipeline.call_args.args[0][self.node]["action"]["param"]["custom_action_param"]
        self.assertIsNone(cleared["prepared_values"])
        self.assertEqual(cleared["mode"], "heart_maze")
        self.assertEqual(cleared["remaining_count_roi"], self.params["remaining_count_roi"])

    def test_other_task_other_mode_and_expired_values_cannot_drive_a_click(self):
        for field, value in [("task_id", 54322), ("mode", "main_story"), ("created", 97)]:
            with self.subTest(field=field):
                prepared = dict(self.params["prepared_values"], **{field: value})
                self.params["prepared_values"], original = prepared, self.params["prepared_values"]
                with patch.object(clothes.time, "monotonic", return_value=100.1), \
                     patch.object(self.action, "_read_round_values", return_value=None) as read, \
                     patch.object(self.action, "_click_roi") as click:
                    self.assertFalse(self.run_action())
                    read.assert_called_once()
                    click.assert_not_called()
                self.params["prepared_values"] = original

    def test_zero_stamina_or_remaining_count_never_clicks(self):
        for values in ([0, 8, 30], [55, 8, 0]):
            self.params["prepared_values"]["values"] = values
            with patch.object(clothes.time, "monotonic", return_value=100.1), \
                 patch.object(self.action, "_read_round_values") as read, \
                 patch.object(self.action, "_has_sufficient_quantity") as quantity, \
                 patch.object(self.action, "_click_roi") as click:
                self.assertTrue(self.run_action())
                read.assert_not_called()
                quantity.assert_not_called()
                click.assert_not_called()

    def test_failed_clear_stops_before_consuming_or_clicking(self):
        self.context.override_pipeline.return_value = False
        with patch.object(self.action, "_read_round_values") as read, patch.object(self.action, "_click_roi") as click:
            self.assertFalse(self.run_action())
            read.assert_not_called()
            click.assert_not_called()


def native_manicure_no_red():
    from maa.custom_recognition import CustomRecognition
    from maa.resource import Resource
    from maa.tasker import Tasker
    from tools.test_make_target_clothes_readiness import ScreenController

    class Recognition(CustomRecognition):
        def analyze(self, context, argv):
            return self.AnalyzeResult(box=(10, 10, 30, 30), detail={}) if argv.node_name == "美甲话题页面就绪" else None

    pipeline = json.loads((ROOT / "assets/resource/pipeline/manicure.json").read_text(encoding="utf-8"))
    nodes = {"美甲话题任务分流": pipeline["美甲话题任务分流"]}
    for name in ("美甲话题页面就绪", "在加载页面", "美甲周期奖励确认", "点击话题任务"):
        nodes[name] = {"recognition": {"type": "Custom", "param": {"custom_recognition": "snapshot"}},
                       "action": "DoNothing"}
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory)
        (bundle / "pipeline").mkdir()
        (bundle / "pipeline/test.json").write_text(json.dumps(nodes, ensure_ascii=False), encoding="utf-8")
        resource = Resource()
        resource.register_custom_recognition("snapshot", Recognition())
        resource.register_custom_action("manicure_topic_dispatch", manicure.ManicureTopicDispatchAction())
        assert resource.post_bundle(bundle).wait().status.succeeded
        controller = ScreenController()
        assert controller.post_connection().wait().status.succeeded
        tasker = Tasker()
        assert tasker.bind(resource, controller)
        assert tasker.post_task("美甲话题任务分流").wait().status.succeeded
        assert not controller.clicks
        del tasker
        del resource


def native_stairway_popups(after_quick_challenge=False):
    import copy
    import time
    from maa.custom_action import CustomAction
    from maa.custom_recognition import CustomRecognition
    from maa.resource import Resource
    from maa.tasker import Tasker
    from tools.test_make_target_clothes_readiness import ScreenController

    class Controller(ScreenController):
        def __init__(self):
            super().__init__()
            self.click_times = []

        def click(self, x, y):
            assert (x, y) == (613, 48)
            self.click_times.append(time.monotonic())
            return super().click(x, y)

    controller = Controller()
    clear_frames, entered = [], []
    first_popup, second_popup = (("心阶部件制作提示", "心阶快速挑战奖励弹窗") if after_quick_challenge
                                 else ("心阶新赛季开始弹窗", "心阶我知道了弹窗"))
    entry, successor = (("点击心阶页面-空白2", "点击真我之境") if after_quick_challenge
                        else ("点击心阶页面-空白", "点击快速挑战"))

    class Recognition(CustomRecognition):
        def analyze(self, context, argv):
            hit = argv.node_name == "导航在心阶页面"
            if argv.node_name == first_popup:
                hit = len(controller.clicks) == 0
            elif argv.node_name == second_popup:
                hit = len(controller.clicks) == 1
                if len(controller.clicks) == 2:
                    clear_frames.append(controller.frames)
            elif argv.node_name == "点击真我之境":
                hit = len(controller.clicks) == 2
            return self.AnalyzeResult(box=(10, 10, 30, 30), detail={}) if hit else None

    class QuickChallenge(CustomAction):
        def run(self, context, argv):
            entered.append(controller.frames)
            return len(controller.clicks) == 2 and len(set(clear_frames)) >= 2

    source = json.loads((ROOT / "assets/resource/pipeline/memory_stairway.json").read_text(encoding="utf-8"))
    nodes = {entry: copy.deepcopy(source[entry]),
             successor: {"recognition": "DirectHit", "action": {"type": "Custom", "param": {
                 "custom_action": "quick",
             }}, "post_delay": 0}}
    recognition_nodes = ("导航在心阶页面", "在加载页面", first_popup, second_popup)
    if after_quick_challenge:
        recognition_nodes += ("点击真我之境",)
    for name in recognition_nodes:
        nodes.setdefault(name, {})["recognition"] = {
            "type": "Custom", "param": {"custom_recognition": "snapshot"}}
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory)
        (bundle / "pipeline").mkdir()
        (bundle / "pipeline/test.json").write_text(json.dumps(nodes, ensure_ascii=False), encoding="utf-8")
        resource = Resource()
        assert resource.register_custom_recognition("snapshot", Recognition())
        assert resource.register_custom_action("memory_stairway_popups", stair.MemoryStairwayPopupsAction())
        assert resource.register_custom_action("quick", QuickChallenge())
        assert resource.post_bundle(bundle).wait().status.succeeded
        assert controller.post_connection().wait().status.succeeded
        tasker = Tasker()
        assert tasker.bind(resource, controller)
        assert tasker.post_task(entry).wait().status.succeeded
        assert len(entered) == 1 and len(controller.clicks) == 2
        assert controller.click_times[1] - controller.click_times[0] >= 1
        del tasker
        del resource


class NativeManicureTests(unittest.TestCase):
    def test_framework_default_arguments_and_successful_leaf_exit(self):
        result = subprocess.run([sys.executable, "-X", "utf8", "-c",
            "from maa.library import Library; Library.version(); "
            "from tools.test_workflow_guards import native_manicure_no_red; native_manicure_no_red()"],
            cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Exception ignored on calling ctypes callback", result.stderr)


class NativeStairwayTests(unittest.TestCase):
    def test_both_popup_handlers_finish_before_quick_challenge(self):
        result = subprocess.run([sys.executable, "-X", "utf8", "-c",
            "from maa.library import Library; Library.version(); "
            "from tools.test_workflow_guards import native_stairway_popups; native_stairway_popups()"],
            cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Exception ignored on calling ctypes callback", result.stderr)

    def test_reward_popup_handlers_finish_before_true_self_entry(self):
        result = subprocess.run([sys.executable, "-X", "utf8", "-c",
            "from maa.library import Library; Library.version(); "
            "from tools.test_workflow_guards import native_stairway_popups; native_stairway_popups(True)"],
            cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Exception ignored on calling ctypes callback", result.stderr)


if __name__ == "__main__":
    unittest.main()
