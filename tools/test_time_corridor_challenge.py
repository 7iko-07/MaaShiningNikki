"""Offline checks for time-corridor remaining counts and OCR button clicks."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from maa.agent.agent_server import AgentServer
from maa.define import Rect


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location(
    "time_corridor_challenge_under_test", ROOT / "agent/custom/make_target_clothes.py"
)
module = importlib.util.module_from_spec(spec)
with patch.object(AgentServer, "register_custom_action", return_value=True):
    spec.loader.exec_module(module)


class TimeCorridorChallengeTests(unittest.TestCase):
    def setUp(self):
        self.action = module.MakeTargetClothesChallengeAction()
        self.context = MagicMock()
        self.controller = self.context.tasker.controller
        self.roi = [60, 809, 130, 171]

    def click(self, retry=3):
        return self.action._click_challenge_button(
            self.context, self.controller, self.roi, retry, 0
        )

    def test_remaining_count_is_taken_after_label_before_slash(self):
        cases = {
            "今日剩余挑战次数：0/15": 0,
            "今日剩余挑战次数: 3 / 15": 3,
            "今日 剩余 挑战 次数 １２／１５": 12,
            "已拥有：17720\n今日剩余挑战次数：0/15": 0,
            "已拥有：17720": None,
            "0/15": None,
            "今日剩余挑战次数：3": None,
            "": None,
        }
        roi = [323, 749, 351, 226]
        for text, expected in cases.items():
            with self.subTest(text=text), patch.object(module, "read_ocr_text", return_value=text) as read:
                self.assertEqual(self.action._read_challenge_count(self.context, None, roi), expected)
                self.assertEqual(read.call_args.args[3], roi)
                self.assertEqual(read.call_args.args[4], [self.action._CHALLENGE_COUNT_PATTERN.pattern])

    def test_wrong_count_retries_without_clicking_and_zero_completes(self):
        for text, expected_success, attempts in (("已拥有：17720", False, 3),
                                                  ("今日剩余挑战次数：0/15", True, 1)):
            with self.subTest(text=text):
                params = {
                    "challenge_count_roi": [323, 749, 351, 226],
                    "stamina_roi": [0, 0, 218, 37],
                    "once_button": self.roi,
                    "multi_button": [218, 807, 121, 172],
                    "load_delay": 0,
                    "retry_delay": 0,
                }
                with patch.object(module, "read_ocr_text", return_value=text) as read, \
                     patch.object(module, "read_ocr_number", return_value=100), \
                     patch.object(self.action, "_click_challenge_button") as click:
                    self.assertEqual(self.action.run(
                        self.context, SimpleNamespace(custom_action_param=json.dumps(params))
                    ), expected_success)
                    self.assertEqual(read.call_count, attempts)
                    click.assert_not_called()

    def test_click_uses_ocr_box_and_refreshes_image(self):
        box = Rect(75, 920, 70, 30)
        self.context.run_recognition_direct.return_value = SimpleNamespace(hit=True, box=box)
        self.context.run_action_direct.return_value = SimpleNamespace(success=True)
        self.assertTrue(self.click())
        self.controller.post_screencap.assert_called_once()
        reco_args = self.context.run_recognition_direct.call_args.args
        self.assertEqual(reco_args[1].roi, tuple(self.roi))
        self.assertEqual(reco_args[1].expected, ["挑战"])
        self.assertIs(reco_args[2], self.controller.cached_image)
        self.assertTrue(self.context.run_action_direct.call_args.args[1].target)
        self.assertEqual(self.context.run_action_direct.call_args.kwargs["box"], box)

    def test_missing_or_invalid_detection_never_clicks(self):
        for result in (None, SimpleNamespace(hit=False, box=Rect(75, 920, 70, 30)),
                       SimpleNamespace(hit=True, box=None),
                       SimpleNamespace(hit=True, box=Rect(75, 920, 0, 30))):
            with self.subTest(result=result):
                self.context.reset_mock()
                self.context.run_recognition_direct.return_value = result
                self.assertFalse(self.click())
                self.assertEqual(self.controller.post_screencap.call_count, 3)
                self.context.run_action_direct.assert_not_called()

    def test_detection_can_succeed_on_retry(self):
        self.context.run_recognition_direct.side_effect = [
            SimpleNamespace(hit=False, box=None),
            SimpleNamespace(hit=True, box=Rect(75, 920, 70, 30)),
        ]
        self.context.run_action_direct.return_value = SimpleNamespace(success=True)
        self.assertTrue(self.click())
        self.assertEqual(self.controller.post_screencap.call_count, 2)
        self.context.run_action_direct.assert_called_once()

    def test_failed_click_is_not_repeated(self):
        self.context.run_recognition_direct.return_value = SimpleNamespace(
            hit=True, box=Rect(75, 920, 70, 30)
        )
        for result in (None, SimpleNamespace(success=False)):
            with self.subTest(result=result):
                self.context.reset_mock()
                self.context.run_action_direct.return_value = result
                self.assertFalse(self.click())
                self.context.run_action_direct.assert_called_once()
                self.context.run_recognition_direct.assert_called_once()

    def test_run_uses_both_rois_and_stops_on_missing_button(self):
        pipeline = json.loads(
            (ROOT / "assets/resource/pipeline/Make_the_target_clothes.json").read_text(encoding="utf-8")
        )
        params = pipeline["计算并挑战时空回廊制衣材料"]["action"]["param"]["custom_action_param"]
        params = dict(params, load_delay=0, click_delay=0, popup_close_delay=0)
        argv = SimpleNamespace(custom_action_param=json.dumps(params))
        for button_results, expected_success in (([True, True], True), ([False], False)):
            with self.subTest(button_results=button_results):
                with patch.object(module, "read_ocr_text", return_value="今日剩余挑战次数：11/15"), \
                     patch.object(module, "read_ocr_number", return_value=100), \
                     patch.object(module.time, "sleep"), \
                     patch.object(self.action, "_click_roi"), \
                     patch.object(self.action, "_click_challenge_button", side_effect=button_results) as click:
                    self.assertEqual(self.action.run(self.context, argv), expected_success)
                    self.assertEqual(click.call_args_list[0].args[2], [218, 807, 121, 172])
                    if expected_success:
                        self.assertEqual(click.call_args_list[1].args[2], self.roi)
                    else:
                        click.assert_called_once()


if __name__ == "__main__":
    unittest.main()
