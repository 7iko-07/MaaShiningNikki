"""独立剧情入口复用原 Pipeline 的回归；原生测试模拟画面，不替代实机验证。"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from maa.agent.agent_server import AgentServer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location(
    "story_auto_click_under_test", ROOT / "agent/custom/story_auto_click.py")
story = importlib.util.module_from_spec(spec)
with patch.object(AgentServer, "register_custom_action", return_value=True), \
     patch.object(AgentServer, "register_custom_recognition", return_value=True), \
     patch.object(AgentServer, "add_controller_sink"):
    spec.loader.exec_module(story)

ENTRY = "剧情自动点击任务链入口"


class StoryEntryTests(unittest.TestCase):
    def run_entry(self, params=None, success=True):
        context = MagicMock()
        context.tasker.stopping = False
        context.tasker.controller.uuid = "story-test"
        context.tasker.controller._handle = 101
        context.clone.return_value.run_task.return_value = SimpleNamespace(
            status=SimpleNamespace(succeeded=success, failed=not success), nodes=[])
        result = story.StoryAutoClickAction().run(context, SimpleNamespace(
            custom_action_param=json.dumps(params or {})))
        return result, context

    def test_default_keeps_original_dispatch_and_jumpback_without_fullscreen(self):
        result, context = self.run_entry()
        self.assertTrue(result)
        call = context.clone.return_value.run_task.call_args
        self.assertEqual(call.args, ("剧情自动点击调度",))
        candidates = call.kwargs["pipeline_override"]["剧情自动点击调度"]["next"]
        self.assertIn("[JumpBack]剧情自动点击推进对话", candidates)
        self.assertFalse(any("点击剧情叉号" in node for node in candidates))
        self.assertFalse(any("Anchor" in node for node in candidates))
        context.run_recognition.assert_not_called()
        context.run_action_direct.assert_not_called()

    def test_enabled_mode_enters_original_fullscreen_chain_and_anchor(self):
        result, context = self.run_entry({"search_chapters": True})
        self.assertTrue(result)
        call = context.clone.return_value.run_task.call_args
        self.assertEqual(call.args, ("剧情自动点击查找章节",))
        self.assertEqual(call.kwargs["pipeline_override"]["剧情自动点击调度"]["next"][-1],
                         "[Anchor]剧情自动点击查找章节")

    def test_original_subtask_failure_is_not_reported_as_completion(self):
        result, _ = self.run_entry(success=False)
        self.assertFalse(result)

    def test_missing_subtask_result_is_failure(self):
        context = MagicMock()
        context.clone.return_value.run_task.return_value = None
        self.assertFalse(story.StoryAutoClickAction().run(
            context, SimpleNamespace(custom_action_param="{}")))

    def test_string_boolean_cannot_enable_fullscreen(self):
        result, context = self.run_entry({"search_chapters": "false"})
        self.assertFalse(result)
        context.clone.assert_not_called()

    def test_special_interaction_stops_without_automatic_choice(self):
        context = MagicMock()
        self.assertTrue(story.StoryAutoClickManualStopAction().run(context, None))
        context.run_task.assert_not_called()

    def test_recognition_timeout_is_normal_but_failed_click_is_not(self):
        last = SimpleNamespace(name=story.DISPATCH_NODE, completed=True,
                               action=SimpleNamespace(success=True))
        placeholder = SimpleNamespace(name="", completed=False, action=None, recognition=None)
        task = SimpleNamespace(status=SimpleNamespace(succeeded=False, failed=True),
                               nodes=[last, placeholder])
        self.assertTrue(story.is_manual_end(task))
        task.nodes = [placeholder]
        self.assertFalse(story.is_manual_end(task))
        task.nodes = [last]
        last.action.success = False
        self.assertFalse(story.is_manual_end(task))
        last.name = "剧情自动点击推进对话"
        last.action.success = True
        self.assertFalse(story.is_manual_end(task))

    def test_dialogue_jumpback_timeout_is_normal_only_after_successful_actions(self):
        dispatch = SimpleNamespace(name=story.DISPATCH_NODE, completed=True,
                                   action=SimpleNamespace(success=True))
        dialogue = SimpleNamespace(name="剧情自动点击推进对话", completed=True,
                                   action=SimpleNamespace(success=True))
        placeholder = SimpleNamespace(name="", completed=False, action=None, recognition=None)
        task = SimpleNamespace(status=SimpleNamespace(succeeded=False, failed=True),
                               nodes=[dispatch, dialogue, dialogue, placeholder])
        self.assertTrue(story.is_manual_end(task))
        dialogue.action.success = False
        self.assertFalse(story.is_manual_end(task))
        dialogue.action.success = True
        dialogue.completed = False
        self.assertFalse(story.is_manual_end(task))
        dialogue.completed = True
        dispatch.action.success = False
        self.assertFalse(story.is_manual_end(task))
        dispatch.action.success = True
        task.nodes = [dialogue, placeholder]
        self.assertFalse(story.is_manual_end(task))
        task.nodes = [dispatch, SimpleNamespace(name="意外节点", completed=True,
                                              action=SimpleNamespace(success=True)), placeholder]
        self.assertFalse(story.is_manual_end(task))

    def test_controller_failure_is_scoped_and_not_converted_to_manual_end(self):
        detail = SimpleNamespace(uuid="story-test", action="screencap")
        controller = SimpleNamespace(_handle=101)
        with story.controller_failures.watch(101) as failures:
            story.controller_failures.on_controller_action(controller, story.NotificationType.Failed, detail)
            self.assertEqual(failures, {"screencap"})
        self.assertFalse(story.controller_failures._active)
        with story.controller_failures.watch(202) as failures:
            # 即使 uuid 相同，另一个实时视图控制器的错误也不能污染当前任务。
            story.controller_failures.on_controller_action(controller, story.NotificationType.Failed, detail)
            self.assertFalse(failures)
        self.assertFalse(story.controller_failures._active)


class NativeStoryTests(unittest.TestCase):
    def test_original_pipeline_retries_transient_misses_and_preserves_other_tasks(self):
        result = subprocess.run([
            sys.executable, "-X", "utf8", "-c",
            "from maa.library import Library; Library.version(); "
            "from tools.test_story_auto_click import native_story_reuse; native_story_reuse()",
        ], cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Exception ignored on calling ctypes callback", result.stderr)


def native_story_reuse(mode_filter=None):
    import numpy as np
    from maa.custom_recognition import CustomRecognition
    from maa.resource import Resource
    from maa.tasker import Tasker
    from tools.test_make_target_clothes_readiness import ScreenController

    click_spec = importlib.util.spec_from_file_location(
        "story_original_click_region", ROOT / "agent/custom/click_region.py")
    click_module = importlib.util.module_from_spec(click_spec)
    with patch.object(AgentServer, "register_custom_action", return_value=True):
        click_spec.loader.exec_module(click_module)

    pipeline = json.loads((ROOT / "assets/resource/pipeline/story_auto_click.json").read_text(encoding="utf-8"))
    interface = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
    option = interface["option"]["全屏点击查找剧情章节"]
    assert option["default_case"] == "No"
    assert option["__mfa_confirm_on_enable"] == "非活动场景请勿开启全屏点击，否则可能误触并消耗资源！"
    assert "剧情自动点击未识别结束" not in pipeline
    prompt = pipeline[ENTRY]["focus"]["Node.Action.Succeeded"]
    assert prompt == {"content": "[size:14][color:#A16207][b]剧情点击结束，需要手动操作。[/b][/color][/size]",
                      "display": "log"}

    original_names = (
        "剧情自动点击查找章节", "剧情自动点击调度", "剧情自动点击推进对话", "剧情自动点击结束识别")
    nodes = pipeline
    nodes["stop"] = {"action": "StopTask"}

    class Controller(ScreenController):
        mode = "unknown"
        misses = 0
        failed_dialogue_click = False

        def screencap(self):
            self.frames += 1
            if self.mode == "capture_fail" and self.frames > 1:
                return np.empty((0, 0, 3), dtype=np.uint8)
            return np.full_like(self.image, 255 if len(self.clicks) % 2 == 0 else 0)

        def click(self, x, y):
            if self.mode == "action_fail":
                return False
            if self.mode == "dialogue_click_fail" and not self.failed_dialogue_click:
                self.failed_dialogue_click = True
                return False
            return super().click(x, y)

    controller = Controller()

    class Recognition(CustomRecognition):
        def analyze(self, context, argv):
            clicks = len(controller.clicks)
            mode = controller.mode
            hit = False
            if argv.node_name == "剧情自动点击结束识别":
                target = 2 if mode in ("search", "transient_after") else 1
                hit = mode not in ("unknown", "special", "capture_fail", "action_fail", "dialogue_click_fail",
                                   "dialogue_timeout") and clicks >= target
            elif argv.node_name == "剧情自动点击推进对话":
                hit = mode in ("dialogue", "transient_before", "transient_after", "dialogue_click_fail", "action_fail") or (
                    mode == "dialogue_timeout" and clicks < 2) or (
                    mode == "search" and clicks == 1)
                if ((mode == "transient_before" and clicks == 0)
                        or (mode == "transient_after" and clicks == 1)) and controller.misses == 0:
                    controller.misses += 1
                    hit = False
            elif argv.node_name == story.SPECIAL_NODE:
                hit = mode == "special"
            return self.AnalyzeResult(box=(538, 0, 182, 277), detail={}) if hit else None

    for name in ("剧情自动点击结束识别", "剧情自动点击推进对话", story.SPECIAL_NODE):
        nodes[name]["recognition"] = {
            "type": "Custom", "param": {"custom_recognition": "story_snapshot"}}

    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory)
        (bundle / "pipeline").mkdir()
        (bundle / "pipeline/story.json").write_text(json.dumps(nodes, ensure_ascii=False), encoding="utf-8")
        # 覆盖升级可能残留旧活动文件；原节点名共存时也必须正常加载和运行。
        legacy_nodes = {name: {} for name in (
            "点击活动剧情", "在活动剧情页面", "点击聊天框", "识别剧情过完了并停止任务")}
        (bundle / "pipeline/legacy_activity.json").write_text(
            json.dumps(legacy_nodes, ensure_ascii=False), encoding="utf-8")
        resource = Resource()
        assert resource.register_custom_recognition("story_snapshot", Recognition())
        assert resource.register_custom_recognition("story_auto_click_controller_failed", story.StoryAutoClickControllerFailedRecognition())
        assert resource.register_custom_action("story_auto_click", story.StoryAutoClickAction())
        assert resource.register_custom_action("story_auto_click_manual_stop", story.StoryAutoClickManualStopAction())
        assert resource.register_custom_action("story_auto_click_controller_guard", story.StoryAutoClickControllerGuardAction())
        assert resource.register_custom_action("click_region", click_module.ClickRegionAction())
        assert resource.post_bundle(bundle).wait().status.succeeded
        assert controller.add_sink(story.controller_failures) is not None
        assert controller.post_connection().wait().status.succeeded
        tasker = Tasker()
        assert tasker.bind(resource, controller)
        baseline = {name: resource.get_node_data(name) for name in original_names}
        for mode, enabled, expected, success in (
            ("dialogue", False, [(0, 301)], True),
            ("transient_before", False, [(0, 301)], True),
            ("transient_after", False, [(0, 301), (0, 301)], True),
            ("search", True, [(0, 0), (0, 301)], True),
            ("dialogue_timeout", False, [(0, 301), (0, 301)], True),
            ("unknown", False, [], True),
            ("special", False, [], True),
            ("capture_fail", False, [], False),
            ("action_fail", False, [], False),
            ("dialogue_click_fail", False, [(70, 301)], False),
        ):
            if mode_filter is not None and mode != mode_filter:
                continue
            print(f"native story: {mode}", flush=True)
            controller.mode = mode
            controller.clicks.clear()
            controller.misses = 0
            controller.frames = 0
            controller.failed_dialogue_click = False
            case = next(case for case in option["cases"] if case["name"] == ("Yes" if enabled else "No"))
            overrides = dict(case["pipeline_override"])
            if mode == "action_fail":
                overrides["剧情自动点击推进对话"] = {"action": {"type": "Click", "param": {}}}
            status = tasker.post_task(ENTRY, overrides).wait().status
            assert status.succeeded == success, (mode, status)
            assert [(x, y) for x, y, _ in controller.clicks] == expected, (mode, controller.clicks)
            if mode.startswith("transient"):
                assert controller.misses == 1
            assert {name: resource.get_node_data(name) for name in original_names} == baseline
            assert not story.controller_failures._active
        del tasker
        del resource


if __name__ == "__main__":
    unittest.main()
