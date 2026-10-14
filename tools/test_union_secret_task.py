"""Offline tests for secret-task remaining-count routing and pipeline wiring."""

import importlib.util
import json
import sys
import unittest
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
            pipeline["点击竞技场-确定影召"]["next"], ["等待机密任务比拼中"]
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
                for execution_node in ("点击开始执行", "点击执行多次"):
                    if execution_node in next_nodes:
                        self.assertLess(
                            completion_index,
                            next_nodes.index(execution_node),
                        )


if __name__ == "__main__":
    unittest.main()
