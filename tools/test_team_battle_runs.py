"""Offline checks for team-battle count selection and independent task options."""

import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from maa.agent.agent_server import AgentServer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location(
    "team_battle_runs_under_test", ROOT / "agent/custom/team_battle_runs.py"
)
module = importlib.util.module_from_spec(spec)
with patch.object(AgentServer, "register_custom_action", return_value=True):
    spec.loader.exec_module(module)


class TeamBattleRunsTests(unittest.TestCase):
    def setUp(self):
        self.action = module.TeamBattleSetRunsAction()
        self.context = MagicMock()
        self.context.tasker.stopping = False
        self.controller = self.context.tasker.controller
        self.controller.post_screencap.return_value.wait.return_value.succeeded = True
        self.context.override_next.return_value = True
        self.current = 3
        self.remaining = 3
        self.stamina = 120
        self.cost = 5
        self.missing = None
        self.context.run_recognition_direct.side_effect = self.recognize
        self.context.run_action_direct.side_effect = self.click
        self.sleep = patch.object(module.time, "sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def recognize(self, kind, params, image):
        texts = {
            tuple(module.TeamBattleSetRunsAction.FIELDS["选中次数"][0]): str(self.current),
            tuple(module.TeamBattleSetRunsAction.FIELDS["剩余次数"][0]): f"剩余次数：{self.remaining}/9",
            tuple(module.TeamBattleSetRunsAction.FIELDS["体力"][0]): f"{self.stamina}/120",
            tuple(module.TeamBattleSetRunsAction.FIELDS["每次体力消耗"][0]): f"每次消耗 {self.cost}",
        }
        if tuple(params.roi) == self.missing:
            return SimpleNamespace(hit=False)
        return SimpleNamespace(hit=True, best_result=SimpleNamespace(text=texts[tuple(params.roi)]))

    def click(self, kind, params):
        self.current += 1 if params.target[0] == 652 else -1
        return SimpleNamespace(success=True)

    def run_action(self, runs, task_name="时光钟表铺"):
        return self.action.run(self.context, SimpleNamespace(
            node_name=f"设置{task_name}刷取次数",
            custom_action_param=json.dumps({"task_name": task_name, "runs": runs}),
        ))

    def test_zero_keeps_existing_flow_without_reading_or_adjusting(self):
        self.assertTrue(self.run_action(0))
        self.context.run_recognition_direct.assert_not_called()
        self.context.run_action_direct.assert_not_called()
        self.controller.post_screencap.assert_not_called()
        self.context.override_next.assert_called_once_with(
            "设置时光钟表铺刷取次数", ["点击时光钟表铺快速组队"]
        )

    def test_decrease_and_increase_are_verified(self):
        self.assertTrue(self.run_action(1))
        self.assertEqual(self.current, 1)
        self.assertEqual(self.context.run_action_direct.call_count, 2)
        self.context.run_action_direct.reset_mock()
        self.assertTrue(self.run_action(3))
        self.assertEqual(self.current, 3)
        self.assertEqual(self.context.run_action_direct.call_count, 2)

    def test_remaining_and_stamina_cap_requested_count(self):
        self.assertTrue(self.run_action(99))
        self.assertEqual(self.current, 3)
        self.context.run_action_direct.assert_not_called()
        self.stamina = 9
        self.assertTrue(self.run_action(99))
        self.assertEqual(self.current, 1)

    def test_zero_capacity_returns_without_quick_team_and_reentry_resets_next(self):
        for capacity in ("remaining", "stamina"):
            with self.subTest(capacity=capacity):
                self.remaining, self.stamina = 3, 120
                setattr(self, capacity, 0)
                self.assertTrue(self.run_action(2, "印象旅航"))
                self.context.override_next.assert_called_with(
                    "设置印象旅航刷取次数", ["点击印象旅航返回"]
                )
                self.context.run_action_direct.assert_not_called()
                self.assertTrue(self.run_action(0, "印象旅航"))
                self.context.override_next.assert_called_with(
                    "设置印象旅航刷取次数", ["点击印象旅航快速组队"]
                )

    def test_invalid_counts_fail_before_any_action(self):
        for runs in (-1, 100, 1.5, True, "2", None):
            with self.subTest(runs=runs):
                self.assertFalse(self.run_action(runs))
        self.context.run_recognition_direct.assert_not_called()
        self.context.run_action_direct.assert_not_called()

    def test_zero_selected_with_no_remaining_runs_completes_normally(self):
        self.current = self.remaining = 0
        self.assertTrue(self.run_action(1))
        self.context.run_action_direct.assert_not_called()
        self.context.override_next.assert_called_with(
            "设置时光钟表铺刷取次数", ["点击时光钟表铺返回"]
        )

    def test_each_missing_number_retries_and_never_clicks(self):
        for roi, _ in self.action.FIELDS.values():
            with self.subTest(roi=roi):
                self.missing = tuple(roi)
                self.controller.post_screencap.reset_mock()
                self.assertFalse(self.run_action(1))
                self.assertEqual(self.controller.post_screencap.call_count, 3)
                self.context.run_action_direct.assert_not_called()

    def test_failed_or_ineffective_adjustment_stops_without_quick_team(self):
        for result in (None, SimpleNamespace(success=False), SimpleNamespace(success=True)):
            with self.subTest(result=result):
                self.context.run_action_direct.reset_mock()
                self.context.run_action_direct.side_effect = None
                self.context.run_action_direct.return_value = result
                self.assertFalse(self.run_action(1))
                self.context.run_action_direct.assert_called_once()

    def test_unexpected_count_change_stops(self):
        def jump(kind, params):
            self.current = 1
            return SimpleNamespace(success=True)
        self.context.run_action_direct.side_effect = jump
        self.assertFalse(self.run_action(2))
        self.context.run_action_direct.assert_called_once()

    def test_screenshot_failure_and_cancel_never_click(self):
        self.controller.post_screencap.return_value.wait.return_value.succeeded = False
        self.assertFalse(self.run_action(1))
        self.context.run_recognition_direct.assert_not_called()
        self.context.run_action_direct.assert_not_called()
        self.context.tasker.stopping = True
        self.assertFalse(self.run_action(1))
        self.context.run_action_direct.assert_not_called()

    def test_deadline_and_failed_next_override_stop(self):
        with patch.object(module.time, "monotonic", side_effect=[0, 61]):
            self.assertFalse(self.run_action(1))
        self.context.run_action_direct.assert_not_called()
        self.context.override_next.return_value = False
        self.assertFalse(self.run_action(0))


class TeamBattleOptionsTests(unittest.TestCase):
    def test_each_option_only_overrides_its_own_ready_node(self):
        interface = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
        tasks = {task["name"]: task for task in interface["task"]}
        self.assertEqual(tasks["结伴"]["option"], ["结伴子任务"])
        companion = interface["option"]["结伴子任务"]
        selections = {case["name"]: case for case in companion["cases"]}
        self.assertEqual(companion["default_case"], list(selections))
        for name, filename, stage in (
            ("时光钟表铺", "time_clock_shop.json", "点击04"),
            ("不落的帷幕", "unending_curtain.json", "点击第八幕"),
            ("印象旅航", "impressions_of_travel.json", "点击虚渊"),
        ):
            with self.subTest(task=name):
                self.assertNotIn(name, tasks)
                self.assertEqual(
                    selections[name]["pipeline_override"],
                    {f"结伴选择{name}": {"enabled": True}},
                )
                option_name = f"{name}刷取次数"
                option = interface["option"][option_name]
                field = option["inputs"][0]
                self.assertEqual(field["default"], "0")
                for number in range(100):
                    self.assertIsNotNone(re.fullmatch(field["verify"], str(number)))
                for invalid in ("-1", "100", "1.5", "", "01", "abc"):
                    self.assertIsNone(re.fullmatch(field["verify"], invalid))
                node = f"设置{name}刷取次数"
                self.assertEqual(list(option["pipeline_override"]), [node])
                params = option["pipeline_override"][node]["action"]["param"]["custom_action_param"]
                self.assertEqual(params, {"task_name": name, "runs": "{runs}"})
                if name == "印象旅航":
                    self.assertIn(option_name, selections[name]["option"])
                else:
                    self.assertIn(f"{name}模式", selections[name]["option"])
                    cases = interface["option"][f"{name}模式"]["cases"]
                    self.assertIn(option_name, cases[0]["option"])
                    self.assertNotIn(option_name, cases[1]["option"])
                pipeline = json.loads((ROOT / "assets/resource/pipeline" / filename).read_text(encoding="utf-8"))
                self.assertEqual(pipeline[stage]["next"], [node])
                self.assertEqual(pipeline[node]["recognition"]["param"]["expected"], ["快速组队"])
                self.assertEqual(pipeline[node]["next"], [f"点击{name}快速组队"])


if __name__ == "__main__":
    unittest.main()
