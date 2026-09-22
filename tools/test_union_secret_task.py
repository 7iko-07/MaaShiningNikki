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

    def test_action_clicks_count_specific_target(self):
        for remaining, expected_click in ((1, (165, 1181)), (2, (296, 1182))):
            with self.subTest(remaining=remaining):
                context, controller, argv = self._action_fixture(remaining)
                action = union_secret_task.UnionSecretTaskExecuteAction()

                self.assertTrue(action.run(context, argv))
                controller.post_click.assert_called_once_with(*expected_click)
                context.override_next.assert_not_called()

    def test_action_routes_zero_remaining_without_click(self):
        context, controller, argv = self._action_fixture(0)
        action = union_secret_task.UnionSecretTaskExecuteAction()

        self.assertTrue(action.run(context, argv))
        controller.post_click.assert_not_called()
        context.override_next.assert_called_once_with(
            "点击执行多次",
            ["点击返回联盟"],
        )

    def test_action_does_not_click_without_execution_text(self):
        context, controller, argv = self._action_fixture(2, button_found=False)
        action = union_secret_task.UnionSecretTaskExecuteAction()

        self.assertFalse(action.run(context, argv))
        controller.post_click.assert_not_called()

    def test_action_cancels_purchase_popup_and_fails(self):
        context, controller, argv = self._action_fixture(1, purchase_popup=True)
        action = union_secret_task.UnionSecretTaskExecuteAction()

        self.assertFalse(action.run(context, argv))
        context.run_task.assert_called_once_with("取消购买机密任务次数")

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
