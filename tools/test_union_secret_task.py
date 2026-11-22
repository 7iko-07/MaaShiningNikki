"""Offline tests for secret-task remaining-count routing and pipeline wiring."""

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from maa.agent.agent_server import AgentServer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location(
    "union_secret_task_under_test",
    ROOT / "agent/custom/union_secret_task.py",
)
union_secret_task = importlib.util.module_from_spec(spec)
with patch.object(AgentServer, "register_custom_recognition", return_value=True):
    with patch.object(AgentServer, "register_custom_action", return_value=True):
        spec.loader.exec_module(union_secret_task)


class UnionSecretTaskTests(unittest.TestCase):
    def test_parse_execution_progress(self):
        cases = {
            "今日执行次数：0/5": (0, 5),
            "今日执行次数: 5 / 5": (5, 5),
            "今日执行次数：１０／１０+": (10, 10),
            " 6/5 ": (6, 5),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(
                    union_secret_task.parse_execution_progress(text),
                    expected,
                )

    def test_completion_requires_zero_remaining(self):
        for text in ("今日执行次数：1/5", "4/5", "5/5", "没有次数"):
            with self.subTest(text=text):
                self.assertFalse(union_secret_task.is_execution_complete(text))
        for text in ("0/5", "0/0", "０／５"):
            with self.subTest(text=text):
                self.assertTrue(union_secret_task.is_execution_complete(text))

    def test_execution_button_node_depends_on_remaining_count(self):
        self.assertIsNone(
            union_secret_task.execution_button_node_for_remaining(0)
        )
        self.assertEqual(
            union_secret_task.execution_button_node_for_remaining(1),
            "机密任务执行一次按钮OCR",
        )
        for remaining in (2, 4, 5):
            with self.subTest(remaining=remaining):
                self.assertEqual(
                    union_secret_task.execution_button_node_for_remaining(remaining),
                    "机密任务执行多次按钮OCR",
                )

    def test_pipeline_waits_for_explicit_completion(self):
        pipeline = json.loads(
            (ROOT / "assets/resource/pipeline/union.json").read_text(encoding="utf-8")
        )
        for node_name in ("联盟机密任务入口分流", "机密任务手动执行后分流"):
            node = pipeline[node_name]
            self.assertIn("机密任务已完成", node["next"])
            self.assertNotIn("点击返回联盟", node["next"])
            self.assertEqual(node["timeout"], 30000)

        self.assertLess(
            pipeline["联盟机密任务入口分流"]["next"].index("机密任务已完成"),
            pipeline["联盟机密任务入口分流"]["next"].index("点击执行多次"),
        )
        self.assertLess(
            pipeline["机密任务手动执行后分流"]["next"].index("机密任务已完成"),
            pipeline["机密任务手动执行后分流"]["next"].index("点击执行多次"),
        )

        complete = pipeline["机密任务已完成"]
        self.assertEqual(
            complete["recognition"]["param"]["custom_recognition"],
            "union_secret_task_complete",
        )
        self.assertEqual(complete["next"], ["点击返回联盟"])

    def test_dynamic_execution_action_uses_count_specific_button_ocr(self):
        pipeline = json.loads(
            (ROOT / "assets/resource/pipeline/union.json").read_text(encoding="utf-8")
        )
        node = pipeline["点击执行多次"]
        self.assertEqual(
            node["recognition"]["param"]["custom_recognition"],
            "union_secret_task_ready",
        )
        self.assertEqual(
            node["action"]["param"]["custom_action"],
            "union_secret_task_execute",
        )
        params = node["action"]["param"]["custom_action_param"]
        self.assertEqual(
            params["single_button_node"],
            "机密任务执行一次按钮OCR",
        )
        self.assertEqual(
            params["multiple_button_node"],
            "机密任务执行多次按钮OCR",
        )
        self.assertEqual(
            pipeline[params["single_button_node"]]["recognition"]["param"],
            {"roi": [75, 1135, 200, 100], "expected": ["执行"]},
        )
        self.assertEqual(
            pipeline[params["multiple_button_node"]]["recognition"]["param"],
            {"roi": [215, 1135, 190, 100], "expected": ["执行"]},
        )
        self.assertEqual(params["completed_next"], "点击返回联盟")
        self.assertIn(params["purchase_popup_node"], pipeline)
        self.assertIn(params["cancel_purchase_node"], pipeline)

    def test_start_execution_button_remains_separate(self):
        pipeline = json.loads(
            (ROOT / "assets/resource/pipeline/union.json").read_text(encoding="utf-8")
        )
        node = pipeline["点击开始执行"]
        self.assertEqual(node["recognition"]["param"]["expected"], ["开始执行"])
        self.assertNotEqual(
            node["action"]["param"]["custom_action"],
            "union_secret_task_execute",
        )

    def simulate_execution(self, frames, *, timeout=6000, override=True, click_success=True,
                           action_success=True, capture_success=True, cancel_at=None, popup_error=False):
        context, controller, argv = self._action_fixture(5)
        context.tasker.stopping = False
        context.override_next.return_value = override
        context.run_action.return_value = SimpleNamespace(success=action_success)
        controller.post_click.return_value.wait.return_value = SimpleNamespace(succeeded=click_success)
        now = [0.0]
        current = [None]
        sequence = iter(frames)
        last = [frames[-1]]
        def screenshot():
            last[0] = next(sequence, last[0])
            current[0] = last[0]
            if cancel_at is not None and now[0] >= cancel_at:
                context.tasker.stopping = True
            return SimpleNamespace(succeeded=capture_success)
        controller.post_screencap.return_value.wait.side_effect = screenshot
        def recognize(node, image):
            if popup_error and node == '机密任务购买次数弹窗':
                raise RuntimeError('recognition failed')
            frame = current[0]
            hit = frame == {'关闭机密任务周奖励弹窗': 'weekly',
                            '联盟执行多次奖励': 'reward',
                            '机密任务购买次数弹窗': 'purchase'}.get(node)
            if node == '机密任务今日执行次数OCR' and isinstance(frame, int):
                return SimpleNamespace(hit=True, best_result=SimpleNamespace(text=f'{frame}/5'),
                                       filtered_results=[], all_results=[], raw_detail=None)
            if node in ('机密任务执行一次按钮OCR', '机密任务执行多次按钮OCR'):
                hit = isinstance(frame, int) and frame > 0
                box = [121, 1160, 88, 42] if frame == 1 else [251, 1162, 90, 40]
                return SimpleNamespace(hit=hit, best_result=SimpleNamespace(text='执行', box=box),
                                       filtered_results=[], all_results=[])
            return SimpleNamespace(hit=hit)
        context.run_recognition.side_effect = recognize
        argv.custom_action_param = json.dumps({'timeout': timeout})
        with patch.object(union_secret_task.time, 'monotonic', side_effect=lambda: now[0]), \
             patch.object(union_secret_task.time, 'sleep', side_effect=lambda dt: now.__setitem__(0, now[0] + dt)):
            ok = union_secret_task.UnionSecretTaskExecuteAction().run(context, argv)
        context.run_task.assert_not_called()
        return ok, context, controller, now[0]

    def test_action_clicks_count_specific_target_and_confirms_zero(self):
        for remaining, expected_click in ((1, (165, 1181)), (2, (296, 1182))):
            ok, context, controller, _ = self.simulate_execution([remaining, remaining, 'reward', 0, 0])
            self.assertTrue(ok)
            controller.post_click.assert_called_once_with(*expected_click)
            context.override_next.assert_called_once_with('点击执行多次', ['点击返回联盟'])

    def test_late_weekly_reward_is_closed_then_execution_retried(self):
        ok, context, controller, _ = self.simulate_execution(
            [5, 5] + ['weekly'] * 5 + [5] * 3 + ['reward'] * 4 + [0, 0])
        self.assertTrue(ok)
        self.assertEqual(controller.post_click.call_count, 2)
        self.assertEqual(context.run_action.call_count, 2)
        self.assertTrue(all(c.args[0] == '机密任务关闭奖励弹窗' for c in context.run_action.call_args_list))

    def test_zero_requires_two_frames_without_click(self):
        ok, context, controller, _ = self.simulate_execution([0, 'transition', 0, 0])
        self.assertTrue(ok)
        controller.post_click.assert_not_called()
        self.assertEqual(controller.post_screencap.call_count, 4)

    def test_unknown_page_never_clicks_and_times_out(self):
        ok, context, controller, elapsed = self.simulate_execution(['transition'])
        self.assertFalse(ok)
        controller.post_click.assert_not_called()
        context.override_next.assert_not_called()
        self.assertEqual(elapsed, 6)

    def test_unchanged_count_retries_but_never_reports_success(self):
        ok, context, controller, elapsed = self.simulate_execution([5])
        self.assertFalse(ok)
        self.assertGreater(controller.post_click.call_count, 1)
        context.override_next.assert_not_called()
        self.assertEqual(elapsed, 6)

    def test_execution_reward_blocks_retry_until_count_refreshes(self):
        ok, context, controller, _ = self.simulate_execution([5, 5, 'reward', 'reward', 'reward', 5])
        self.assertFalse(ok)
        controller.post_click.assert_called_once()
        context.override_next.assert_not_called()

    def test_count_decrease_continues_with_single_button_until_zero(self):
        ok, context, controller, _ = self.simulate_execution([5, 5, 1, 1, 1, 1, 0, 0])
        self.assertTrue(ok)
        self.assertEqual([c.args for c in controller.post_click.call_args_list], [(296, 1182), (165, 1181)])

    def test_purchase_popup_cancels_and_fails_without_executing(self):
        ok, context, controller, _ = self.simulate_execution(['purchase'])
        self.assertFalse(ok)
        context.run_action.assert_called_once_with('取消购买机密任务次数')
        controller.post_click.assert_not_called()

    def test_failed_capture_click_close_or_override_cannot_succeed(self):
        for frames, options in (([5], {'capture_success': False}),
                                ([5], {'click_success': False}),
                                (['weekly'], {'action_success': False}),
                                ([0], {'override': False})):
            with self.subTest(options=options):
                ok, _, _, _ = self.simulate_execution(frames, **options)
                self.assertFalse(ok)

    def test_cancellation_stops_without_new_click(self):
        ok, context, controller, _ = self.simulate_execution([5], cancel_at=0.25)
        self.assertFalse(ok)
        controller.post_click.assert_not_called()
        context.override_next.assert_not_called()

    def test_popup_recognition_error_cannot_be_treated_as_no_popup(self):
        ok, context, controller, _ = self.simulate_execution([5], popup_error=True)
        self.assertFalse(ok)
        controller.post_click.assert_not_called()
        context.override_next.assert_not_called()

    def test_reward_closer_is_a_single_action_without_successors(self):
        pipeline = json.loads((ROOT / 'assets/resource/pipeline/union.json').read_text(encoding='utf-8'))
        closer = pipeline['机密任务关闭奖励弹窗']
        self.assertEqual(closer['action']['type'], 'Click')
        self.assertNotIn('next', closer)
        self.assertEqual(pipeline['点击执行多次']['next'], ['点击返回联盟'])

    def test_ready_recognition_retries_transitions_without_acting(self):
        context, controller, _ = self._action_fixture(2)
        recognize = context.run_recognition.side_effect
        stage = 0

        def transition(node, image):
            if stage == 0:
                return SimpleNamespace(hit=False)
            if stage == 1 and node == "机密任务执行多次按钮OCR":
                return SimpleNamespace(hit=False)
            return recognize(node, image)

        context.run_recognition.side_effect = transition
        ready = union_secret_task.UnionSecretTaskReadyRecognition()
        argv = SimpleNamespace(image=controller.cached_image)
        self.assertIsNone(ready.analyze(context, argv))
        stage = 1
        self.assertIsNone(ready.analyze(context, argv))
        stage = 2
        result = ready.analyze(context, argv)
        self.assertEqual(result.box, (251, 1162, 90, 40))
        controller.post_click.assert_not_called()
        context.run_task.assert_not_called()

    def test_ready_recognition_selects_single_button_and_leaves_zero_to_completion(self):
        ready = union_secret_task.UnionSecretTaskReadyRecognition()
        for remaining in (0, 1):
            with self.subTest(remaining=remaining):
                context, controller, _ = self._action_fixture(remaining)
                result = ready.analyze(
                    context, SimpleNamespace(image=controller.cached_image)
                )
                if remaining == 0:
                    self.assertIsNone(result)
                    self.assertEqual(context.run_recognition.call_count, 1)
                else:
                    self.assertEqual(result.box, (121, 1160, 88, 42))

    def test_dispatch_returns_do_not_leave_jumpback_continuations(self):
        pipeline = json.loads(
            (ROOT / "assets/resource/pipeline/union.json").read_text(encoding="utf-8")
        )
        for handler, dispatch in (
            ("关闭机密任务周奖励弹窗", "联盟机密任务入口分流"),
            ("点击前往当前目标", "联盟机密任务入口分流"),
            ("手动执行后点击前往当前目标", "机密任务手动执行后分流"),
        ):
            with self.subTest(handler=handler):
                self.assertEqual(pipeline[handler]["next"], [dispatch])

    def test_battle_wait_has_one_bounded_recognition_loop(self):
        pipeline = json.loads(
            (ROOT / "assets/resource/pipeline/union.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            pipeline["点击竞技场-下一步"]["next"], ["等待机密任务比拼中"]
        )
        wait = pipeline["等待机密任务比拼中"]
        self.assertEqual(wait["next"], ["等待竞技场比拼结束"])
        self.assertEqual(wait["timeout"], 120000)
        self.assertFalse(wait.get("on_error"))
        # With only an OCR candidate, no successful DirectHit/JumpBack can
        # restart the native next-list timer during battle or loading.
        self.assertEqual(
            pipeline[wait["next"][0]]["recognition"]["type"], "OCR"
        )

    def _action_fixture(self, remaining, purchase_popup=False, button_found=True):
        controller = MagicMock()
        controller.cached_image.copy.return_value = MagicMock(name="image")
        controller.post_screencap.return_value.wait.return_value = None
        controller.post_click.return_value.wait.return_value = None

        progress_result = SimpleNamespace(
            hit=True,
            best_result=SimpleNamespace(text=f"今日执行次数：{remaining}/5"),
            filtered_results=[],
            all_results=[],
            raw_detail=None,
        )
        miss = SimpleNamespace(hit=False)
        hit = SimpleNamespace(hit=True)
        single_button = SimpleNamespace(
            hit=True,
            best_result=SimpleNamespace(text="执行一次", box=[121, 1160, 88, 42]),
            filtered_results=[],
            all_results=[],
        )
        multiple_button = SimpleNamespace(
            hit=True,
            best_result=SimpleNamespace(text="执行二次", box=[251, 1162, 90, 40]),
            filtered_results=[],
            all_results=[],
        )

        def recognize(node_name, _image):
            if node_name == "机密任务今日执行次数OCR":
                return progress_result
            if node_name == "机密任务执行一次按钮OCR":
                return single_button if button_found else miss
            if node_name == "机密任务执行多次按钮OCR":
                return multiple_button if button_found else miss
            if node_name == "机密任务购买次数弹窗":
                return hit if purchase_popup else miss
            if node_name == "联盟执行多次奖励":
                return hit
            raise AssertionError(f"unexpected recognition node: {node_name}")

        context = MagicMock()
        context.tasker.controller = controller
        context.run_recognition.side_effect = recognize
        context.run_task.return_value = SimpleNamespace(success=True)
        argv = SimpleNamespace(
            node_name="点击执行多次",
            custom_action_param=json.dumps(
                {
                    "popup_timeout": 0,
                }
            ),
        )
        return context, controller, argv

    def test_interface_overrides_check_completion_before_execution(self):
        interface = json.loads(
            (ROOT / "assets/interface.json").read_text(encoding="utf-8")
        )
        cases = interface["option"]["机密任务执行方式"]["cases"]
        for case in cases:
            with self.subTest(case=case["name"]):
                next_nodes = case["pipeline_override"]["联盟机密任务入口分流"][
                    "next"
                ]
                completion_index = next_nodes.index("机密任务已完成")
                for execution_node in ("点击开始执行", "点击执行多次", "机密任务刷爆发", "机密任务每周检查"):
                    if execution_node in next_nodes:
                        self.assertLess(
                            completion_index,
                            next_nodes.index(execution_node),
                        )


class SecretTaskBurstTests(unittest.TestCase):
    def watch(self, frames, required, total=2):
        action = union_secret_task.UnionSecretTaskBurstAction()
        context = SimpleNamespace(tasker=SimpleNamespace(stopping=False))
        now = [0.]
        with patch.object(action, "_capture", side_effect=frames), \
             patch.object(action, "_read_part", side_effect=lambda _, frame: frame[0]), \
             patch.object(action, "_ocr", side_effect=lambda _, frame, *args: SimpleNamespace(
                 hit=bool(frame[1]), filtered_results=[SimpleNamespace(text=frame[1])],
             )), \
             patch.object(union_secret_task.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(union_secret_task.time, "sleep", side_effect=lambda dt: now.__setitem__(0, now[0] + dt)):
            return action._watch(context, required, total, 60)

    def test_three_selected_parts_must_trigger_in_one_battle(self):
        parts = {"裙子", "发型", "鞋袜"}
        frames = [
            ("裙子", "魅力爆发"), ("裙子", "魅力爆发"), ("裙子", ""),
            ("发型", ""), ("发型", "魅力爆发"), ("发型", ""),
            ("鞋袜", ""), ("鞋袜", "魅力爆发"),
        ]
        self.assertEqual(self.watch(frames, parts, 3), "matched")

    def test_missing_required_round_retries_after_confirmed_transition(self):
        frames = [
            ("发型", ""), ("发型", "闪耀瞬间"), ("面饰", ""), ("面饰", ""),
        ]
        with patch.object(union_secret_task.logger, "info") as info:
            self.assertEqual(self.watch(frames, {"裙子", "发型"}), "retry")
        info.assert_called_once_with("发型：未爆发")

    def test_v6_reads_both_known_and_unlisted_round_labels(self):
        action = union_secret_task.UnionSecretTaskBurstAction()
        for text, part in (("面饰评分", "面饰"), ("颈饰评分", "项圈"), ("特殊饰品评分", "特殊饰品")):
            with patch.object(action, "_ocr", return_value=SimpleNamespace(
                hit=True, best_result=SimpleNamespace(text=text),
            )):
                self.assertEqual(action._read_part(None, None), part)

    def test_any_pair_ignores_single_burst_that_spans_part_labels(self):
        frames = [
            ("裙子", ""), ("裙子", ""),
            ("耳饰", "魅力爆发"), ("耳饰", "魅力爆发"),
            ("项圈", "魅力爆发"), ("项圈", "魅力爆发"),
            ("项圈", ""), ("发型", ""), ("发型", "魅力爆发"),
        ]
        with patch.object(union_secret_task.logger, "info") as info:
            self.assertEqual(self.watch(frames, set()), "matched")
        messages = [call.args[0] for call in info.call_args_list]
        self.assertEqual([msg for msg in messages if msg.endswith("：已爆发")],
                         ["耳饰：已爆发", "发型：已爆发"])

    def test_three_bursts_may_include_one_or_two_required_parts(self):
        frames = [
            ("裙子", "魅力爆发"), ("裙子", ""), ("裙子", ""),
            ("发型", "魅力爆发"), ("发型", ""), ("发型", ""),
            ("面饰", "魅力爆发"),
        ]
        for required in (set(), {"发型"}, {"裙子", "发型"}):
            with self.subTest(required=required):
                self.assertEqual(self.watch(frames, required, 3), "matched")

    def test_last_round_waits_for_skill_before_retry(self):
        frames = [("发型", "")] * 2 + [("鞋袜", "")] * 3 + [("鞋袜", "轻盈脚步")] * 8
        self.assertEqual(self.watch(frames, set()), "retry")

    def test_missing_required_round_stops_after_last_round(self):
        frames = [("发型", "魅力爆发")] + [("鞋袜", "轻盈脚步")] * 9
        with patch.object(union_secret_task.logger, "info") as info:
            self.assertEqual(self.watch(frames, {"发型", "耳饰"}, 2), "unavailable")
        messages = [call.args[0] for call in info.call_args_list]
        self.assertTrue(any("未识别到必爆部位（耳饰）" in msg for msg in messages))
        self.assertNotIn("耳饰：未爆发", messages)

    def make_runner(self, **options):
        action = union_secret_task.UnionSecretTaskBurstAction()
        context = MagicMock()
        context.get_node_data.return_value = {"attach": options}
        context.tasker.stopping = False
        context.override_next.return_value = True
        start_task = context.clone.return_value.run_task.return_value
        start_task.status.succeeded = True
        start_task.node_id_list = [123]
        context.tasker.get_node_detail.return_value = SimpleNamespace(
            name="等待机密任务比拼中", completed=True,
        )
        return action, context, SimpleNamespace(node_name="机密任务刷爆发")

    def test_user_limit_exits_once_and_does_not_enter_settlement(self):
        action, context, argv = self.make_runner(total=2, minutes="37")
        with patch.object(action, "_wait_progress", return_value=(5, 5)), \
             patch.object(action, "_watch", return_value="limit") as watch, \
             patch.object(action, "_exit_battle", return_value=True) as exit_battle, \
             patch.object(union_secret_task.time, "monotonic", side_effect=[100, 105, 2321]):
            self.assertTrue(action.run(context, argv))
        watch.assert_called_once_with(context, set(), 2, 2320)
        exit_battle.assert_called_once_with(context, (5, 5))
        context.clone.return_value.run_task.assert_called_once()
        context.override_next.assert_called_once_with(argv.node_name, [])

    def test_empty_or_zero_limit_executes_remaining_without_farming(self):
        for minutes in ("", "0", 0):
            with self.subTest(minutes=minutes):
                action, context, argv = self.make_runner(total=2, minutes=minutes)
                with patch.object(action, "_wait_progress") as wait_progress:
                    self.assertTrue(action.run(context, argv))
                wait_progress.assert_not_called()
                context.clone.assert_not_called()
                context.override_next.assert_called_once_with(argv.node_name, ["点击执行多次"])

    def test_timeout_continue_hands_current_battle_to_settlement(self):
        action, context, argv = self.make_runner(total=3, minutes=1, continue_on_timeout=True)
        with patch.object(action, "_wait_progress", return_value=(5, 5)), \
             patch.object(action, "_watch", return_value="limit"), \
             patch.object(action, "_exit_battle") as exit_battle:
            self.assertTrue(action.run(context, argv))
        context.clone.return_value.run_task.assert_called_once()
        exit_battle.assert_not_called()
        context.override_next.assert_called_once_with(argv.node_name, ["等待机密任务比拼中"])

    def test_timeout_between_battles_with_continue_starts_one_final_battle(self):
        action, context, argv = self.make_runner(total=3, minutes=1, continue_on_timeout=True)
        with patch.object(action, "_wait_progress", return_value=(5, 5)), \
             patch.object(action, "_watch", side_effect=["retry", "limit"]), \
             patch.object(action, "_exit_battle", return_value=True) as exit_battle, \
             patch.object(union_secret_task.time, "monotonic", side_effect=[100, 105, 165]):
            self.assertTrue(action.run(context, argv))
        self.assertEqual(context.clone.return_value.run_task.call_count, 2)
        exit_battle.assert_called_once()
        context.override_next.assert_called_once_with(argv.node_name, ["等待机密任务比拼中"])

    def test_manual_stop_does_not_continue_to_settlement(self):
        action, context, argv = self.make_runner(total=3, minutes=1, continue_on_timeout=True)
        with patch.object(action, "_wait_progress", return_value=(5, 5)), \
             patch.object(action, "_watch", return_value="stopped"), \
             patch.object(action, "_exit_battle") as exit_battle:
            self.assertFalse(action.run(context, argv))
        exit_battle.assert_not_called()
        context.override_next.assert_not_called()

    def test_outfit_timeout_resumes_donation_only_after_return(self):
        for recovered in (True, False):
            with self.subTest(recovered=recovered):
                action, context, argv = self.make_runner(total=2, minutes=1, continue_on_timeout=True)
                context.clone.return_value.run_task.return_value.status.succeeded = recovered
                context.tasker.get_node_detail.return_value = SimpleNamespace(
                    name="返回联盟页面", completed=recovered,
                )
                with patch.object(action, "_wait_progress", return_value=(5, 5)), \
                     patch.object(action, "_watch") as watch:
                    self.assertEqual(action.run(context, argv), recovered)
                watch.assert_not_called()
                context.clone.return_value.run_task.assert_called_once()
                if recovered:
                    context.override_next.assert_called_once_with(argv.node_name, ["点击联盟捐献"])
                else:
                    context.override_next.assert_not_called()

    def test_unavailable_part_exits_without_starting_another_battle(self):
        action, context, argv = self.make_runner(total=3, minutes=60, 发型=True)
        with patch.object(action, "_wait_progress", return_value=(5, 5)), \
             patch.object(action, "_watch", return_value="unavailable") as watch, \
             patch.object(action, "_exit_battle", return_value=True) as exit_battle:
            self.assertTrue(action.run(context, argv))
        self.assertEqual(watch.call_args.args[1:3], ({"发型"}, 3))
        exit_battle.assert_called_once_with(context, (5, 5))
        context.clone.return_value.run_task.assert_called_once()
        context.override_next.assert_called_once_with(argv.node_name, [])

    def test_invalid_selection_stops_before_entering_battle(self):
        action, context, argv = self.make_runner(total=2, minutes=60, 裙子=True, 发型=True, 鞋袜=True)
        with patch.object(action, "_wait_progress") as progress:
            self.assertFalse(action.run(context, argv))
        progress.assert_not_called()
        context.clone.assert_not_called()

    def test_exit_requires_unchanged_remaining_count(self):
        action, context, _ = self.make_runner()
        exit_button = SimpleNamespace(hit=True, box=(20, 150, 60, 22))
        context.run_action_direct.return_value.success = True
        with patch.object(action, "_capture"), \
             patch.object(action, "_ocr", side_effect=[
                 SimpleNamespace(hit=False), SimpleNamespace(hit=False), exit_button,
                 SimpleNamespace(hit=True),
             ]), \
             patch.object(action, "_progress", return_value=(4, 5)), \
             patch.object(union_secret_task.time, "sleep"):
            with self.assertRaisesRegex(ValueError, "剩余次数发生变化"):
                action._exit_battle(context, (5, 5))
        context.run_action_direct.assert_called_once()

    def test_exit_retries_visible_button_and_clicks_exact_center(self):
        action, context, _ = self.make_runner()
        now = [0.0]
        context.run_action_direct.return_value.success = True
        miss = SimpleNamespace(hit=False)
        exit_button = SimpleNamespace(hit=True, box=(1, 115, 92, 18))
        popup = SimpleNamespace(hit=True)
        confirm = SimpleNamespace(hit=True, box=(456, 684, 64, 39))

        def recognize(_context, _image, roi, _expected):
            clicks = context.run_action_direct.call_count
            if roi[0] == 160:
                return SimpleNamespace(hit=clicks == 3)
            if roi[0] == 70:
                return popup if clicks == 2 else miss
            return confirm if roi[0] == 350 else exit_button

        with patch.object(action, "_capture"), \
             patch.object(action, "_ocr", side_effect=recognize), \
             patch.object(action, "_progress", return_value=(5, 5)), \
             patch.object(union_secret_task.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(union_secret_task.time, "sleep", side_effect=lambda dt: now.__setitem__(0, now[0] + dt)):
            self.assertTrue(action._exit_battle(context, (5, 5)))
        calls = context.run_action_direct.call_args_list
        self.assertEqual([call.args[1].target for call in calls],
                         [(47, 124), (47, 124), (488, 703)])
        self.assertTrue(all(len(call.args) == 2 for call in calls))


class SecretTaskOutfitNextTests(unittest.TestCase):
    def setUp(self):
        self.action = union_secret_task.UnionSecretTaskOutfitNextAction()
        self.context = MagicMock()
        self.context.tasker.stopping = False
        self.controller = self.context.tasker.controller
        self.controller.post_screencap.return_value.wait.return_value.succeeded = True
        self.context.clone.return_value.run_task.return_value.status.succeeded = True
        self.context.run_recognition.side_effect = self.recognize
        self.context.clone.return_value.run_recognition.side_effect = self.recognize
        self.controller.post_click.side_effect = self.click_next
        self.node = "点击竞技场-下一步"
        self.now = 0.0
        self.next_clicks = []
        self.confirm_at = float("inf")
        self.single_line_at = float("inf")
        self.loading_at = float("inf")

    def recognize(self, node, image, override):
        if node == union_secret_task.OUTFIT_CONFIRM_NODE:
            if override:
                self.assertTrue(override[node]["recognition"]["param"]["only_rec"])
            at = self.single_line_at if override else self.confirm_at
            return SimpleNamespace(hit=self.now >= at, box=(323, 1091, 73, 39))
        if node == self.node:
            return SimpleNamespace(hit=True, box=(62, 1180, 158, 45))
        self.assertEqual(node, "在加载页面")
        return SimpleNamespace(hit=self.now >= self.loading_at)

    def click_next(self, x, y):
        self.assertEqual((x, y), (141, 1202))
        self.next_clicks.append(self.now)
        return self.controller.post_screencap.return_value

    def run_action(self):
        with patch.object(union_secret_task.time, "monotonic", side_effect=lambda: self.now), \
             patch.object(union_secret_task.time, "sleep", side_effect=self.sleep):
            return self.action.run(self.context, SimpleNamespace(node_name=self.node))

    def sleep(self, seconds):
        self.now += seconds

    def test_confirm_takes_priority_without_waiting_for_retry(self):
        self.confirm_at = 0.6
        self.assertTrue(self.run_action())
        self.assertEqual(self.next_clicks, [0])
        self.assertLess(self.now, 2.2)
        self.context.override_next.assert_called_once_with(self.node, ["等待机密任务比拼中"])

    def test_single_line_confirm_prevents_retry_even_when_next_is_visible(self):
        self.single_line_at = 2
        self.assertTrue(self.run_action())
        self.assertEqual(self.next_clicks, [0])
        self.assertGreaterEqual(self.now, 2.2)
        confirm, override = self.context.clone.return_value.run_task.call_args.args
        self.assertEqual(confirm, "点击竞技场-确定影召")
        self.assertEqual(override[confirm]["action"]["param"]["target"], [323, 1091, 73, 39])

    def test_missed_click_retries_after_grace_period(self):
        self.confirm_at = 2.6
        self.assertTrue(self.run_action())
        self.assertEqual(len(self.next_clicks), 2)
        self.assertGreaterEqual(self.next_clicks[1] - self.next_clicks[0], 2.2)

    def test_loading_pauses_clicks_and_timeout_routes_to_alliance(self):
        self.loading_at = 0.2
        self.assertTrue(self.run_action())
        self.assertEqual(self.next_clicks, [0])
        self.assertEqual(self.now, 15)
        self.context.clone.return_value.run_task.assert_not_called()
        self.context.override_next.assert_called_once_with(self.node, ["返回联盟页面"])


class SecretTaskWeeklyTests(unittest.TestCase):
    def setUp(self):
        self.state = union_secret_task.state
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        patcher = patch.object(self.state, "STATE_PATH", Path(self.temp.name) / "weekly.sqlite3")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.state._sessions.clear()
        self.now = datetime(2026, 11, 18, 10, tzinfo=self.state.BEIJING)
        clock = patch.object(self.state, "now_beijing", side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.action = union_secret_task.UnionSecretTaskWeeklyAction()
        self.context = MagicMock()
        self.context.tasker.controller.uuid = "union-device"
        self.context.tasker.stopping = False
        self.entry = {"mode": "start", "weekly": True}
        self.burst = {"total": 2, "minutes": 60}
        self.context.get_node_data.side_effect = lambda node: {
            "attach": self.burst if node == "机密任务刷爆发" else self.entry,
        }

    def invoke(self, operation, task_id=20):
        return self.action.run(self.context, SimpleNamespace(
            node_name="联盟任务链入口" if operation == "enter" else operation,
            task_detail=SimpleNamespace(task_id=task_id),
            custom_action_param=json.dumps({"operation": operation}),
        ))

    def branch(self):
        return self.context.override_next.call_args.args[1]

    def new_run(self):
        self.action = union_secret_task.UnionSecretTaskWeeklyAction()
        self.assertTrue(self.invoke("enter"))
        self.assertTrue(self.invoke("choose"))

    def finish(self):
        self.new_run()
        self.assertTrue(self.invoke("complete"))

    def test_week_resets_on_monday_five_in_beijing(self):
        before = datetime(2026, 11, 23, 4, 59, 59, tzinfo=self.state.BEIJING)
        after = before.replace(hour=5, minute=0, second=0)
        self.assertEqual(union_secret_task.secret_task_week_key(before), "2026-11-16T05:00:00+08:00")
        self.assertEqual(union_secret_task.secret_task_week_key(after.astimezone(timezone.utc)),
                         "2026-11-23T05:00:00+08:00")

    def test_failed_run_retries_and_completion_persists_after_restart(self):
        self.new_run()
        self.assertEqual(self.branch(), ["点击开始执行"])
        self.new_run()
        self.assertEqual(self.branch(), ["点击开始执行"])
        self.assertTrue(self.invoke("complete"))
        self.new_run()
        self.assertEqual(self.branch(), ["点击执行多次"])

    def test_new_settings_replace_previous_completion(self):
        self.finish()
        self.entry["mode"] = "burst"
        self.new_run()
        self.assertEqual(self.branch(), ["机密任务刷爆发"])
        self.invoke("complete")
        for change in ({"total": 3}, {"发型": True}, {"minutes": 30}, {"continue_on_timeout": True}):
            self.burst.update(change)
            self.new_run()
            self.assertEqual(self.branch(), ["机密任务刷爆发"])
            self.invoke("complete")
            self.new_run()
            self.assertEqual(self.branch(), ["点击执行多次"])
        self.entry["mode"] = "start"
        self.new_run()
        self.assertEqual(self.branch(), ["点击开始执行"])

    def test_disabled_weekly_still_executes_and_records_completion(self):
        self.entry["weekly"] = False
        self.finish()
        self.new_run()
        self.assertEqual(self.branch(), ["点击开始执行"])
        self.entry["weekly"] = True
        self.new_run()
        self.assertEqual(self.branch(), ["点击执行多次"])

    def test_accounts_are_isolated_and_unknown_login_cannot_save(self):
        for suffix, expected in (("0012", "点击开始执行"), ("0013", "点击开始执行"),
                                 ("0012", "点击执行多次")):
            self.state.begin_login(self.context, 10, True)
            self.state.set_login_suffix(self.context, suffix)
            self.state.set_login_server(self.context, "千羽森林")
            self.context.tasker.get_task_detail.return_value.status.succeeded = True
            self.new_run()
            self.assertEqual(self.branch(), [expected])
            if expected == "点击开始执行":
                self.invoke("complete")
        self.state.begin_login(self.context, 10, True)
        with patch.object(self.state.StateStore, "mark_secret_task_completed") as save:
            self.new_run()
            self.assertEqual(self.branch(), ["点击开始执行"])
            self.invoke("complete")
        save.assert_not_called()

    def test_stop_wrong_task_and_new_week_do_not_save(self):
        self.new_run()
        with patch.object(self.state.StateStore, "mark_secret_task_completed") as save:
            self.assertFalse(self.invoke("complete", task_id=21))
            self.context.tasker.stopping = True
            self.assertFalse(self.invoke("complete"))
            self.context.tasker.stopping = False
            self.now = datetime(2026, 11, 23, 5, tzinfo=self.state.BEIJING)
            self.assertTrue(self.invoke("complete"))
        save.assert_not_called()

    def test_settlement_record_is_after_confirmed_result_and_before_batch(self):
        pipeline = json.loads((ROOT / "assets/resource/pipeline/union.json").read_text(encoding="utf-8"))
        self.assertEqual(pipeline["机密任务本次执行进度"]["next"], ["机密任务比拼完成记录"])
        self.assertEqual(pipeline["机密任务比拼完成记录"]["action"]["param"]["custom_action_param"],
                         {"operation": "complete"})
        self.assertIn("点击执行多次", pipeline["机密任务比拼完成记录"]["next"])
        self.assertEqual(union_secret_task.BURST_START_OVERRIDE["等待机密任务比拼中"]["next"], [])

    def test_outfit_return_stops_inside_burst_but_normally_continues_to_donation(self):
        pipeline = json.loads((ROOT / "assets/resource/pipeline/union.json").read_text(encoding="utf-8"))
        self.assertEqual(pipeline["返回联盟页面"]["next"], ["点击联盟捐献"])
        self.assertEqual(union_secret_task.BURST_START_OVERRIDE["返回联盟页面"]["next"], [])
        self.assertNotIn("next", pipeline["点击竞技场-确定影召"])


if __name__ == "__main__":
    unittest.main()
