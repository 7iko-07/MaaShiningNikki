"""心阶快速战斗：颜色确认、计算等待、乱序弹窗和累计期限。"""

import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from maa.define import Rect
from tools.test_workflow_guards import ROOT, stair


TITLE = "导航在心阶页面"
LOADING = "在加载页面"
OPTION = "心阶快速战斗选项"
RED = "心阶快速战斗已开启"
CALCULATING = "心阶搭配之力计算提示"
POPUPS = ("心阶挑战奖励弹窗", "心阶战斗评分弹窗", "心阶部件制作提示")
CHALLENGES = tuple(stair.MemoryStairwayChallengeAction.NEXT_NODES)


class ChallengeTests(unittest.TestCase):
    def simulate(self, frames=None, results=None, *, node=CHALLENGES[0], capture_success=True,
                 fail_node=None, click_success=True, slow_node=None, cancel_node=None,
                 settling_node=None, box=None, click_delay=0):
        frames = frames or [{OPTION, RED, node}]
        results = results if results is not None else [{TITLE, POPUPS[0]}, {TITLE}, {TITLE}]
        context = Mock()
        context.tasker.stopping = False
        context.override_next.return_value = True
        controller = context.tasker.controller
        now, prep_index, result_index = [0.0], [0], [0]
        started, clicks, captures, settling = [], [], [], [False]

        def capture():
            sequence, index = (results, result_index) if started else (frames, prep_index)
            frame = sequence[min(index[0], len(sequence) - 1)]
            index[0] += 1
            controller.cached_image = frame
            captures.append((now[0], frame))
            return SimpleNamespace(wait=lambda: SimpleNamespace(status=SimpleNamespace(succeeded=capture_success)))

        def recognize(name, image):
            if name == slow_node:
                now[0] += 31
            if name == cancel_node:
                context.tasker.stopping = True
            if name == settling_node:
                settling[0] = True
            if name == fail_node:
                return None
            return SimpleNamespace(hit=name in image, box=box if box is not None else Rect(480, 970, 160, 40))

        def click(kind, param):
            self.assertEqual(kind, "Click")
            target = tuple(param.target)
            clicks.append((now[0], target))
            if target not in ((96, 990, 1, 1), (613, 48, 1, 1)):
                started.append(now[0])
                now[0] += click_delay
            return SimpleNamespace(success=click_success)

        controller.post_screencap.side_effect = capture
        context.run_recognition.side_effect = recognize
        context.run_action_direct.side_effect = click
        argv = SimpleNamespace(node_name=node)
        with patch.object(stair.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(stair.time, "sleep", side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)), \
             patch.object(stair, "in_settlement", side_effect=lambda: settling[0]):
            success = stair.MemoryStairwayChallengeAction().run(context, argv)
        return success, clicks, now[0], started, context, captures

    def test_checked_option_is_not_toggled_and_challenge_is_clicked_only_once(self):
        for node in CHALLENGES:
            with self.subTest(node=node):
                success, clicks, _, started, context, _ = self.simulate(node=node)
                self.assertTrue(success)
                self.assertEqual(len(started), 1)
                self.assertNotIn((96, 990, 1, 1), [target for _, target in clicks])
                context.override_next.assert_called_once_with(
                    node, [stair.MemoryStairwayChallengeAction.NEXT_NODES[node]])

    def test_unchecked_option_is_clicked_once_and_red_is_confirmed_on_a_new_frame(self):
        for node in CHALLENGES:
            ready = {OPTION, node}
            success, clicks, _, started, _, _ = self.simulate(
                [ready, ready, ready | {RED}], node=node)
            self.assertTrue(success)
            self.assertEqual(sum(target == (96, 990, 1, 1) for _, target in clicks), 1)
            self.assertEqual(len(started), 1)
            self.assertGreaterEqual(started[0], 1)

    def test_calculation_reappearing_resets_two_frame_confirmation(self):
        ready = {OPTION, RED, CHALLENGES[0]}
        success, _, _, started, _, _ = self.simulate([ready, ready | {CALCULATING}, ready, ready])
        self.assertTrue(success)
        self.assertEqual(started, [1.5])

    def test_calculation_and_failed_toggle_each_stop_at_thirty_seconds(self):
        for frame, expected_clicks in (({OPTION, RED, CHALLENGES[0], CALCULATING}, 0),
                                       ({OPTION, CHALLENGES[0]}, 1)):
            success, clicks, elapsed, started, _, _ = self.simulate([frame])
            self.assertFalse(success)
            self.assertEqual(len(clicks), expected_clicks)
            self.assertEqual(started, [])
            self.assertEqual(elapsed, 30)

    def test_encounters_do_not_wait_for_true_self_calculation_hint(self):
        for node in CHALLENGES[1:]:
            success, _, _, started, context, _ = self.simulate([{OPTION, RED, node, CALCULATING}], node=node)
            self.assertTrue(success)
            self.assertEqual(started, [0])
            self.assertNotIn(CALCULATING, [call.args[0] for call in context.run_recognition.call_args_list])

    def test_unknown_loading_or_missing_button_pages_never_toggle_or_challenge(self):
        ready = {OPTION, RED, CHALLENGES[0]}
        for frame in (set(), ready | {LOADING}, ready - {OPTION}, ready - {CHALLENGES[0]}):
            success, clicks, elapsed, _, _, _ = self.simulate([frame])
            self.assertFalse(success)
            self.assertEqual(clicks, [])
            self.assertEqual(elapsed, 30)

    def test_recognition_capture_click_and_invalid_box_failures_stop(self):
        for options in ({"capture_success": False}, {"click_success": False},
                        *({"fail_node": node} for node in (OPTION, RED, CALCULATING, *POPUPS)),
                        {"box": Rect(480, 970, 0, 40)}):
            with self.subTest(options=options):
                success, _, _, _, _, _ = self.simulate(**options)
                self.assertFalse(success)

    def test_slow_recognition_and_cancellation_never_start_challenge(self):
        for key in ("slow_node", "cancel_node"):
            for node in (LOADING, OPTION, RED, CALCULATING):
                success, clicks, _, started, _, _ = self.simulate(**{key: node})
                self.assertFalse(success)
                self.assertEqual(started, [])
                self.assertEqual(clicks, [])

    def test_settlement_before_start_skips_successors_without_clicking(self):
        success, clicks, _, started, context, _ = self.simulate(settling_node=RED)
        self.assertTrue(success)
        self.assertEqual(clicks, [])
        self.assertEqual(started, [])
        context.override_next.assert_called_with(CHALLENGES[0], [])

    def test_all_popup_orders_are_handled_and_each_frame_checks_all_three(self):
        for order in itertools.permutations(POPUPS):
            frames = [{TITLE, popup} for popup in order for _ in range(2)] + [{TITLE}, {TITLE}]
            success, clicks, _, started, context, _ = self.simulate(results=frames)
            self.assertTrue(success)
            self.assertEqual(len(started), 1)
            blank_times = [when for when, target in clicks if target == (613, 48, 1, 1)]
            self.assertEqual(len(blank_times), 3)
            self.assertTrue(all(b - a >= 1 for a, b in zip(blank_times, blank_times[1:])))
            calls = [call.args[0] for call in context.run_recognition.call_args_list]
            self.assertEqual(len({calls.count(popup) for popup in POPUPS}), 1)

    def test_one_optional_popup_and_delayed_first_result_are_supported(self):
        for popup in POPUPS:
            frames = [{TITLE}] * 4 + [{TITLE, popup}, {TITLE}, {TITLE}]
            success, clicks, _, _, _, _ = self.simulate(results=frames)
            self.assertTrue(success)
            self.assertEqual(sum(target == (613, 48, 1, 1) for _, target in clicks), 1)

    def test_late_popup_and_loading_reset_clear_confirmation(self):
        for middle in ({TITLE, POPUPS[1]}, {TITLE, LOADING}):
            frames = [{TITLE, POPUPS[0]}, {TITLE}, middle, {TITLE}, {TITLE}]
            success, _, _, _, _, captures = self.simulate(results=frames)
            self.assertTrue(success)
            self.assertEqual([frame for _, frame in captures[-5:]], frames)

    def test_no_result_or_persistent_popups_stop_thirty_seconds_after_click(self):
        for frames in ([{TITLE}], [{TITLE, POPUPS[0]}],
                       [{TITLE, popup} for _ in range(30) for popup in POPUPS]):
            success, _, elapsed, started, _, _ = self.simulate(results=frames, click_delay=2)
            self.assertFalse(success)
            self.assertEqual(len(started), 1)
            self.assertEqual(elapsed - started[0], 30)

    def test_popup_capture_recognition_click_failure_and_cancel_stop(self):
        for options in ({"fail_node": POPUPS[0]}, {"cancel_node": POPUPS[1]}, {"slow_node": POPUPS[2]}):
            success, clicks, _, started, _, _ = self.simulate(**options)
            self.assertFalse(success)
            self.assertEqual(len(started), 1)
            self.assertEqual(len(clicks), 1)


def native_challenges():
    from maa.custom_recognition import CustomRecognition
    from maa.resource import Resource
    from maa.tasker import Tasker
    from tools.test_make_target_clothes_readiness import ScreenController

    source = json.loads((ROOT / "assets/resource/pipeline/memory_stairway.json").read_text(encoding="utf-8"))
    controller = ScreenController()
    state = {"started": False, "cleared": False}
    transitions = []

    class Controller(ScreenController):
        def click(self, x, y):
            if (x, y) == (560, 990):
                assert not state["started"]
                state["started"] = True
            elif (x, y) == (613, 48):
                assert state["started"] and not state["cleared"]
                state["cleared"] = True
            else:
                raise AssertionError((x, y))
            return super().click(x, y)

    controller = Controller()

    class Recognition(CustomRecognition):
        def analyze(self, context, argv):
            name = argv.node_name
            hit = (name in (OPTION, RED, *CHALLENGES) and not state["started"]
                   or name == TITLE or name == POPUPS[1] and state["started"] and not state["cleared"])
            return self.AnalyzeResult(box=(480, 970, 160, 40), detail={}) if hit else None

    class EnterEncounter(stair.CustomAction):
        def run(self, context, argv):
            assert state["cleared"]
            transitions.append(argv.node_name)
            state.update(started=False, cleared=False)
            return True

    nodes = {name: source[name] for name in CHALLENGES}
    for name in (TITLE, LOADING, OPTION, RED, CALCULATING, *POPUPS, *CHALLENGES):
        nodes.setdefault(name, {})["recognition"] = {
            "type": "Custom", "param": {"custom_recognition": "screen"}}
    for index in (1, 2):
        name = f"点击境遇{index}"
        nodes[name] = {"recognition": "DirectHit", "action": {"type": "Custom", "param": {
            "custom_action": "enter"}}, "next": [CHALLENGES[index]], "post_delay": 0}
    nodes["stop"] = {"action": "DoNothing", "post_delay": 0}
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory)
        (bundle / "pipeline").mkdir()
        (bundle / "pipeline/test.json").write_text(json.dumps(nodes, ensure_ascii=False), encoding="utf-8")
        resource = Resource()
        assert resource.register_custom_recognition("screen", Recognition())
        assert resource.register_custom_action("enter", EnterEncounter())
        assert resource.register_custom_action("memory_stairway_challenge", stair.MemoryStairwayChallengeAction())
        assert resource.post_bundle(bundle).wait().status.succeeded
        assert controller.post_connection().wait().status.succeeded
        tasker = Tasker()
        assert tasker.bind(resource, controller)
        with patch.object(stair, "in_settlement", return_value=False):
            assert tasker.post_task(CHALLENGES[0]).wait().status.succeeded
        assert transitions == ["点击境遇1", "点击境遇2"]
        assert [(x, y) for x, y, _ in controller.clicks] == [(560, 990), (613, 48)] * 3
        del tasker
        del resource


class NativeChallengeTests(unittest.TestCase):
    def test_three_stages_in_real_framework_and_python_callbacks(self):
        result = subprocess.run([sys.executable, "-X", "utf8", "-c",
            "from maa.library import Library; Library.version(); "
            "from tools.test_memory_stairway_challenge import native_challenges; native_challenges()"],
            cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", timeout=25)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Exception ignored on calling ctypes callback", result.stderr)


if __name__ == "__main__":
    unittest.main()
