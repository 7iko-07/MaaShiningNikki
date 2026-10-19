"""心阶下一步：目标页确认、实时按钮框及累计期限回归。"""

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


BUTTON = "心阶搭配下一步按钮"
LOADING = "在加载页面"


class NextStepTests(unittest.TestCase):
    def simulate(self, frames, suffix="", *, capture_success=True, click_success=True,
                 failed_node=None, slow_node=None, cancel_node=None):
        context = Mock()
        context.tasker.stopping = False
        controller = context.tasker.controller
        ready = f"点击心阶-装备印象{suffix}"
        now, captures, clicks = [0.0], [], []

        def capture():
            frame = frames[min(len(captures), len(frames) - 1)]
            captures.append(frame)
            controller.cached_image = frame
            return SimpleNamespace(wait=lambda: SimpleNamespace(status=SimpleNamespace(succeeded=capture_success)))

        def recognize(node, image):
            if node == slow_node:
                now[0] += 12
            if node == cancel_node:
                context.tasker.stopping = True
            if node == failed_node:
                return None
            return SimpleNamespace(hit=node in image, box=image.get(node))

        def click(kind, param):
            self.assertEqual(kind, "Click")
            clicks.append((now[0], param.target))
            return SimpleNamespace(success=click_success)

        controller.post_screencap.side_effect = capture
        context.run_recognition.side_effect = recognize
        context.run_action_direct.side_effect = click
        argv = SimpleNamespace(custom_action_param=json.dumps({"ready_node": ready}))
        with patch.object(stair.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(stair.time, "sleep", side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)):
            result = stair.MemoryStairwayNextStepAction().run(context, argv)
        return result, clicks, now[0], context, captures

    def test_three_outfits_wait_for_target_and_use_refreshed_box_centers(self):
        for suffix in ("", "2", "3"):
            ready = f"点击心阶-装备印象{suffix}"
            frames = [{BUTTON: Rect(100, 1180, 80, 40)}, {},
                      {BUTTON: [120, 1184, 100, 40]}, {ready: Rect(574, 395, 96, 24)}]
            success, clicks, elapsed, _, _ = self.simulate(frames, suffix)
            self.assertTrue(success)
            self.assertEqual(clicks, [(0, (140, 1200, 1, 1)), (1, (170, 1204, 1, 1))])
            self.assertEqual(elapsed, 1.5)

    def test_ready_takes_priority_over_next_button_visible_behind_panel(self):
        success, clicks, elapsed, _, _ = self.simulate([
            {BUTTON: Rect(100, 1180, 80, 40), "点击心阶-装备印象": Rect(574, 395, 96, 24)}])
        self.assertTrue(success)
        self.assertEqual(clicks, [])
        self.assertEqual(elapsed, 0)

    def test_unknown_or_loading_page_never_clicks_and_times_out(self):
        for frame in ({}, {LOADING: True, BUTTON: Rect(100, 1180, 80, 40)}):
            success, clicks, elapsed, _, captures = self.simulate([frame])
            self.assertFalse(success)
            self.assertEqual(clicks, [])
            self.assertEqual(elapsed, 10)
            self.assertEqual(len(captures), 20)

    def test_persistent_next_button_retries_without_resetting_deadline(self):
        success, clicks, elapsed, _, _ = self.simulate([{BUTTON: Rect(100, 1180, 80, 40)}])
        self.assertFalse(success)
        self.assertEqual(elapsed, 10)
        self.assertEqual([value[0] for value in clicks], list(range(10)))

    def test_slow_recognition_cannot_click_or_report_success_after_deadline(self):
        for node in (BUTTON, "点击心阶-装备印象"):
            frames = [{BUTTON: Rect(100, 1180, 80, 40)},
                      {"点击心阶-装备印象": Rect(574, 395, 96, 24)}]
            success, clicks, _, _, _ = self.simulate(frames, slow_node=node)
            self.assertFalse(success)
            self.assertEqual(clicks, [])

    def test_capture_recognition_and_click_failure_propagate(self):
        for options in ({"capture_success": False}, {"click_success": False},
                        {"failed_node": BUTTON}, {"failed_node": LOADING},
                        {"failed_node": "点击心阶-装备印象"}):
            with self.subTest(options=options):
                success, clicks, _, _, _ = self.simulate([{BUTTON: Rect(100, 1180, 80, 40)}], **options)
                self.assertFalse(success)
                if options.get("click_success") is not False:
                    self.assertEqual(clicks, [])

    def test_cancellation_and_invalid_boxes_never_click(self):
        for node in (BUTTON, LOADING, "点击心阶-装备印象"):
            success, clicks, _, _, _ = self.simulate(
                [{BUTTON: Rect(100, 1180, 80, 40)}], cancel_node=node)
            self.assertFalse(success)
            self.assertEqual(clicks, [])
        for box in (None, Rect(100, 1180, 0, 40), [100, 1180, 80, -1]):
            success, clicks, _, _, _ = self.simulate([{BUTTON: box}])
            self.assertFalse(success)
            self.assertEqual(clicks, [])


def native_next_step():
    from maa.custom_recognition import CustomRecognition
    from maa.resource import Resource
    from maa.tasker import Tasker
    from tools.test_make_target_clothes_readiness import ScreenController

    source = json.loads((ROOT / "assets/resource/pipeline/memory_stairway.json").read_text(encoding="utf-8"))
    controller = ScreenController()

    class Recognition(CustomRecognition):
        def analyze(self, context, argv):
            hit = argv.node_name == BUTTON or (
                argv.node_name.startswith("点击心阶-装备印象") and len(controller.clicks) >= 1)
            return self.AnalyzeResult(box=(100, 1180, 80, 40), detail={}) if hit else None

    nodes = {}
    for suffix in ("", "2", "3"):
        name, ready = f"点击心阶-下一步{suffix}", f"点击心阶-装备印象{suffix}"
        nodes[name] = source[name]
        nodes[ready] = {"recognition": {"type": "Custom", "param": {"custom_recognition": "snapshot"}},
                        "action": "DoNothing", "post_delay": 0}
    for name in (BUTTON, LOADING):
        nodes[name] = {"recognition": {"type": "Custom", "param": {"custom_recognition": "snapshot"}}}
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory)
        (bundle / "pipeline").mkdir()
        (bundle / "pipeline/test.json").write_text(json.dumps(nodes, ensure_ascii=False), encoding="utf-8")
        resource = Resource()
        assert resource.register_custom_recognition("snapshot", Recognition())
        assert resource.register_custom_action("memory_stairway_next_step", stair.MemoryStairwayNextStepAction())
        assert resource.post_bundle(bundle).wait().status.succeeded
        assert controller.post_connection().wait().status.succeeded
        tasker = Tasker()
        assert tasker.bind(resource, controller)
        for suffix in ("", "2", "3"):
            controller.clicks.clear()
            assert tasker.post_task(f"点击心阶-下一步{suffix}").wait().status.succeeded
            assert len(controller.clicks) == 1 and controller.clicks[0][:2] == (140, 1200)
        del tasker
        del resource


class NativeNextStepTests(unittest.TestCase):
    def test_real_sdk_callback_and_three_pipeline_branches(self):
        result = subprocess.run([sys.executable, "-X", "utf8", "-c",
            "from maa.library import Library; Library.version(); "
            "from tools.test_memory_stairway_next_step import native_next_step; native_next_step()"],
            cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Exception ignored on calling ctypes callback", result.stderr)


if __name__ == "__main__":
    unittest.main()
