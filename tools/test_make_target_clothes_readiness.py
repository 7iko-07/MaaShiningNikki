"""Run clothing readiness gates in MaaFramework with simulated OCR/screens."""

import copy
import json
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from maa.controller import CustomController
from maa.custom_action import CustomAction
from maa.custom_recognition import CustomRecognition
from maa.resource import Resource
from maa.tasker import Tasker


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = json.loads((ROOT / "assets/resource/pipeline/Make_the_target_clothes.json").read_text(encoding="utf-8"))


class ScreenController(CustomController):
    def __init__(self, animated=False):
        super().__init__()
        self.frames = 0
        self.animated = animated
        self.clicks = []
        self.image = np.random.default_rng(0).integers(0, 256, (1280, 720, 3), dtype=np.uint8)

    def connect(self):
        return True

    def request_uuid(self):
        return "clothing-readiness-test"

    def get_features(self):
        # Route native Click through click(), not touch_down()/touch_up().
        return 0

    def screencap(self):
        self.frames += 1
        return 255 - self.image if self.animated and self.frames % 2 else self.image

    def click(self, x, y):
        self.clicks.append((x, y, self.frames))
        return True


class LoadingRecognition(CustomRecognition):
    def __init__(self, controller, frames):
        super().__init__()
        self.controller = controller
        self.frames = frames

    def analyze(self, context, argv):
        if self.controller.frames in self.frames:
            return self.AnalyzeResult(box=(310, 592, 101, 98), detail={})
        return None


class ScreenRecognition(CustomRecognition):
    def __init__(self, controller, texts, ready_frame, back_visible=True):
        super().__init__()
        self.controller = controller
        self.texts = texts
        self.ready_frame = ready_frame
        self.back_visible = back_visible

    def analyze(self, context, argv):
        param = json.loads(argv.custom_recognition_param)
        # A return button can be visible even on the old, not-ready page.
        hit = param.get("return_button", False) and self.back_visible
        if self.controller.frames >= self.ready_frame and "roi" in param:
            texts = self.texts() if callable(self.texts) else self.texts
            text = texts.get(tuple(param["roi"]), "")
            hit = any(re.search(pattern, text) for pattern in param["expected"])
        return self.AnalyzeResult(box=(20, 20, 30, 30), detail={}) if hit else None


class RecordAction(CustomAction):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.calls = []

    def run(self, context, argv):
        self.calls.append((argv.node_name, self.controller.frames))
        return True


def simulated_recognition(recognition):
    kind = recognition if isinstance(recognition, str) else recognition["type"]
    if kind in ("And", "Or"):
        result = copy.deepcopy(recognition)
        key = "all_of" if kind == "And" else "any_of"
        for child in result["param"][key]:
            if isinstance(child, dict):
                child["recognition"] = simulated_recognition(child["recognition"])
        return result
    if kind == "DirectHit":
        return recognition
    params = recognition["param"] if kind == "OCR" else {"return_button": True}
    return {"type": "Custom", "param": {
        "custom_recognition": "screen", "custom_recognition_param": params,
    }}


class ClothingReadinessTests(unittest.TestCase):
    def run_return(self, entry, stuck=False, back_visible=True, already_ready=False):
        from tools.test_time_corridor_challenge import module
        pipeline = copy.deepcopy(PIPELINE)
        params = pipeline[entry]["action"]["param"]["custom_action_param"]
        params["timeout"] = 2500 if not stuck and back_visible else 1000
        confirm = pipeline[entry]["next"][0]
        challenge = {
            (323, 749, 351, 226): "今日剩余挑战次数：0/15",
            (0, 0, 218, 37): "193/140",
            (82, 0, 145, 40): "193/140",
            (55, 755, 205, 52): "每次消耗：8",
            (298, 755, 330, 52): "剩余心灵迷宫次数：0/30",
            (68, 817, 108, 40): "挑战一次",
        }
        ready = {(121, 1052, 599, 130): "全部材料"}
        phases = [challenge]
        if "时空回廊" in entry:
            phases.append({(0, 145, 195, 65): "套装制作榜", (500, 1204, 220, 76): "当前积分：8000"})
        if "心灵迷宫" in entry:
            # The real failure stopped on a map absent from the old source list.
            phases.append({})
        phases += [{**ready, (80, 335, 220, 75): "前往收集材料"}, ready]
        if already_ready:
            phases = [ready]
        controller = ScreenController()
        texts = lambda: phases[0 if stuck else min(len(controller.clicks), len(phases) - 1)]
        recognition = ScreenRecognition(controller, texts, ready_frame=1, back_visible=back_visible)
        record = RecordAction(controller)
        for name, node in pipeline.items():
            node["recognition"] = simulated_recognition(node["recognition"])
            node.pop("post_wait_freezes", None)
            if name != entry:
                node["next"] = []
                node["action"] = "DoNothing"
        pipeline["在加载页面"] = {"recognition": {"type": "Custom", "param": {
            "custom_recognition": "loading",
        }}, "action": "DoNothing"}
        pipeline[confirm].update(action={"type": "Custom", "param": {"custom_action": "record"}}, next=[])
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "pipeline").mkdir()
            (bundle / "pipeline/test.json").write_text(json.dumps(pipeline, ensure_ascii=False), encoding="utf-8")
            resource = Resource()
            self.assertTrue(resource.register_custom_recognition("screen", recognition))
            self.assertTrue(resource.register_custom_recognition("loading", LoadingRecognition(controller, [])))
            self.assertTrue(resource.register_custom_action("make_target_clothes_return", module.MakeTargetClothesReturnAction()))
            self.assertTrue(resource.register_custom_action("record", record))
            self.assertTrue(resource.post_bundle(bundle).wait().status.succeeded)
            self.assertTrue(controller.post_connection().wait().status.succeeded)
            tasker = Tasker()
            self.assertTrue(tasker.bind(resource, controller))
            succeeded = tasker.post_task(entry).wait().status.succeeded
            del tasker
            del resource
        return succeeded, controller.clicks, record.calls

    def test_return_walks_layers_and_preserves_each_branch(self):
        for entry in PIPELINE:
            if not entry.startswith("返回制衣页面-"):
                continue
            with self.subTest(entry=entry):
                success, clicks, calls = self.run_return(entry)
                self.assertTrue(success)
                self.assertEqual(len(clicks), 2 if "主线" in entry else 3)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][0], PIPELINE[entry]["next"][0])
                self.assertLess(clicks[-1][2], calls[0][1])
                # The material popup must be closed via its known blank area,
                # not by following the visible background return arrow.
                self.assertTrue(304 <= clicks[-1][0] < 375)
                self.assertTrue(144 <= clicks[-1][1] < 180)

    def test_return_stuck_page_retries_and_still_times_out(self):
        success, clicks, calls = self.run_return("返回制衣页面-主线", stuck=True)
        self.assertFalse(success)
        self.assertGreater(len(clicks), 1)
        self.assertEqual(calls, [])

    def test_return_missing_button_never_clicks(self):
        success, clicks, calls = self.run_return("返回制衣页面-主线", back_visible=False)
        self.assertFalse(success)
        self.assertEqual(clicks, [])
        self.assertEqual(calls, [])

    def test_return_already_on_clothes_list_does_not_click(self):
        success, clicks, calls = self.run_return("返回制衣页面-时空回廊", already_ready=True)
        self.assertTrue(success)
        self.assertEqual(clicks, [])
        self.assertEqual(len(calls), 1)

    def test_navigation_fallback_reuses_native_recognition_and_preserves_post_delay(self):
        from tools.test_page_navigation import navigation
        evidence = {}

        class Button(CustomRecognition):
            def analyze(self, context, argv):
                evidence["recognitions"] = evidence.get("recognitions", 0) + 1
                return self.AnalyzeResult(box=(24, 51, 36, 37), detail={"button": True})

        class Click(CustomAction):
            def run(self, context, argv):
                evidence["action_node"] = context.get_node_data("点击返回")
                evidence["box"] = (argv.box.x, argv.box.y, argv.box.w, argv.box.h)
                evidence["action_time"] = time.monotonic()
                return True

        class Fallback(CustomAction):
            def run(self, context, argv):
                succeeded = navigation.PageNavigateAction()._run_fallback_step(context, ["点击返回"], [])
                evidence["returned_time"] = time.monotonic()
                evidence["original_node"] = context.get_node_data("点击返回")
                return succeeded

        controller = ScreenController()
        resource = Resource()
        resource.register_custom_recognition("button", Button())
        resource.register_custom_action("click", Click())
        resource.register_custom_action("fallback", Fallback())
        pipeline = {
            "entry": {"recognition": "DirectHit", "action": {"type": "Custom", "param": {
                "custom_action": "fallback",
            }}, "pre_delay": 0, "post_delay": 0},
            "点击返回": {"recognition": {"type": "Custom", "param": {
                "custom_recognition": "button",
            }}, "action": {"type": "Custom", "param": {"custom_action": "click"}},
                "pre_delay": 500, "post_delay": 120},
        }
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "pipeline").mkdir()
            (bundle / "pipeline/test.json").write_text(json.dumps(pipeline, ensure_ascii=False), encoding="utf-8")
            self.assertTrue(resource.post_bundle(bundle).wait().status.succeeded)
            self.assertTrue(controller.post_connection().wait().status.succeeded)
            tasker = Tasker()
            self.assertTrue(tasker.bind(resource, controller))
            self.assertTrue(tasker.post_task("entry").wait().status.succeeded)
            del tasker
            del resource
        self.assertEqual(evidence["recognitions"], 1)
        self.assertEqual(evidence["box"], (24, 51, 36, 37))
        self.assertEqual(evidence["action_node"]["pre_delay"], 0)
        self.assertEqual(evidence["action_node"]["post_delay"], 120)
        self.assertEqual(evidence["original_node"]["pre_delay"], 500)
        self.assertGreaterEqual(evidence["returned_time"] - evidence["action_time"], .1)

    def run_gate(self, entry, texts, ready_frame=1, loading_frames=(), animated=False, wait_timeout=800):
        pipeline = copy.deepcopy(PIPELINE)
        gate = pipeline[entry]["next"][0]
        successors = pipeline[gate].get("next", [])
        gate_action = pipeline[gate].get("action")
        custom_wait = isinstance(gate_action, dict) and gate_action.get("type") == "Custom"
        if custom_wait:
            gate_action["param"]["custom_action_param"]["timeout"] = wait_timeout
        # Keep the real gate graph and native freeze wait. Replace only IO/business actions.
        for name, node in pipeline.items():
            node["recognition"] = simulated_recognition(node["recognition"])
            if name not in (entry, gate):
                node["action"] = {"type": "Custom", "param": {"custom_action": "record"}}
                node["next"] = []
                node["pre_delay"] = node["post_delay"] = 0
                node.pop("post_wait_freezes", None)
        pipeline[entry].update(recognition="DirectHit", action="DoNothing", pre_delay=0,
                               post_delay=0, timeout=400, rate_limit=10)
        pipeline["在加载页面"] = {
            "recognition": {"type": "Custom", "param": {"custom_recognition": "loading"}},
            "action": "DoNothing",
        }
        controller = ScreenController(animated=animated)
        recognition = ScreenRecognition(controller, texts, ready_frame)
        loading = LoadingRecognition(controller, loading_frames)
        action = RecordAction(controller)
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "pipeline").mkdir()
            (bundle / "pipeline/test.json").write_text(json.dumps(pipeline, ensure_ascii=False), encoding="utf-8")
            resource = Resource()
            self.assertTrue(resource.register_custom_recognition("screen", recognition))
            self.assertTrue(resource.register_custom_action("record", action))
            self.assertTrue(resource.register_custom_recognition("loading", loading))
            if custom_wait:
                # Library.version() has already selected client mode in this process.
                from tools.test_time_corridor_challenge import module
                waiter = module.MakeTargetClothesWaitReadyAction()
                self.assertTrue(resource.register_custom_action("make_target_clothes_wait_ready", waiter))
            self.assertTrue(resource.post_bundle(bundle).wait().status.succeeded)
            self.assertTrue(controller.post_connection().wait().status.succeeded)
            tasker = Tasker()
            self.assertTrue(tasker.bind(resource, controller))
            status = tasker.post_task(entry).wait().status
            self.assertTrue(all(name in successors for name, _ in action.calls))
            del tasker
            del resource
        return status.succeeded, action.calls

    def test_old_material_page_return_button_cannot_bypass_readiness(self):
        success, calls = self.run_gate("点击全部材料-时空回廊", {}, ready_frame=10000)
        self.assertFalse(success)
        self.assertEqual(calls, [])

    def test_other_material_source_allows_return_only_after_page_arrives(self):
        success, calls = self.run_gate("点击全部材料-时空回廊", {
            (181, 471, 97, 511): "心灵迷宫",
        }, ready_frame=4)
        self.assertTrue(success)
        self.assertEqual([name for name, _ in calls], ["点击返回材料页-时空回廊"])
        self.assertGreaterEqual(calls[0][1], 4)

    def test_completed_materials_on_last_item_keep_completion_branch(self):
        success, calls = self.run_gate("点击末项全部材料-主线", {
            (234, 662, 256, 100): "前往制作",
        })
        self.assertTrue(success)
        self.assertEqual([name for name, _ in calls], ["材料已收集完毕-主线末项"])

    def test_challenge_waits_for_numbers_and_button(self):
        entry = "点击前往获取时空回廊制衣材料"
        texts = {(323, 749, 351, 226): "今日剩余挑战次数：5/15", (0, 0, 218, 37): "234/140"}
        success, calls = self.run_gate(entry, texts)
        self.assertFalse(success)
        self.assertEqual(calls, [])
        texts[(60, 809, 130, 171)] = "挑战一次"
        success, calls = self.run_gate(entry, texts, ready_frame=4)
        self.assertTrue(success)
        self.assertEqual([name for name, _ in calls], ["计算并挑战时空回廊制衣材料"])
        self.assertGreaterEqual(calls[0][1], 4)

    def test_zero_count_and_zero_stamina_do_not_require_enabled_buttons(self):
        cases = [
            ("时空回廊", {(323, 749, 351, 226): "今日剩余挑战次数：0/15", (0, 0, 218, 37): "234/140"}),
            ("心灵迷宫", {(82, 0, 145, 40): "100/140", (55, 755, 205, 52): "每次消耗：8", (298, 755, 330, 52): "剩余心灵迷宫次数：0/30"}),
            ("主线", {(82, 0, 145, 40): "0/140", (55, 755, 205, 52): "每次消耗：4"}),
        ]
        for branch, texts in cases:
            with self.subTest(branch=branch):
                success, calls = self.run_gate("点击前往获取" + branch + "制衣材料", texts)
                self.assertTrue(success)
                self.assertEqual(len(calls), 1)

    def test_batch_challenge_requires_cost_and_heart_maze_count(self):
        for branch in ("心灵迷宫", "主线"):
            with self.subTest(branch=branch):
                texts = {(82, 0, 145, 40): "100/140", (210, 806, 135, 67): "挑战多次"}
                entry = "点击前往获取" + branch + "制衣材料"
                success, calls = self.run_gate(entry, texts)
                self.assertFalse(success)
                self.assertEqual(calls, [])
                texts[(55, 755, 205, 52)] = "每次消耗：8"
                if branch == "心灵迷宫":
                    success, calls = self.run_gate(entry, texts)
                    self.assertFalse(success)
                    self.assertEqual(calls, [])
                    texts[(298, 755, 330, 52)] = "剩余心灵迷宫次数：10/30"
                success, calls = self.run_gate(entry, texts)
                self.assertTrue(success)
                self.assertEqual(len(calls), 1)

    def test_sufficient_quantity_can_finish_without_challenge_numbers(self):
        for branch in ("心灵迷宫", "主线"):
            with self.subTest(branch=branch):
                success, calls = self.run_gate("点击前往获取" + branch + "制衣材料", {
                    (379, 355, 265, 63): "数量已足够",
                })
                self.assertTrue(success)
                self.assertEqual(len(calls), 1)

    def test_option_overrides_cannot_restore_initial_three_second_sleep(self):
        interface = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
        for branch in ("心灵迷宫", "主线"):
            name = "批量挑战" + branch + "制衣材料"
            self.assertEqual(PIPELINE[name]["action"]["param"]["custom_action_param"]["load_delay"], 0)
            option = interface["option"]["制作目标服装" + branch + "挑战次数"]
            self.assertEqual(option["pipeline_override"][name]["action"]["param"]["custom_action_param"]["load_delay"], 0)

    def test_challenge_result_waits_for_completion_text(self):
        success, _ = self.run_gate("等待制衣挑战结果", {
            (96, 568, 553, 597): "过 关 完 成",
        }, ready_frame=4)
        self.assertTrue(success)

    def test_challenge_result_missing_text_times_out(self):
        success, _ = self.run_gate("等待制衣挑战结果", {
            (96, 568, 553, 597): "第7次挑战",
        })
        self.assertFalse(success)

    def test_loading_blocks_visible_ready_content_until_it_disappears(self):
        for frames in (range(1, 9), range(3, 9)):
            with self.subTest(loading_frames=frames):
                success, calls = self.run_gate("点击前往获取主线制衣材料", {
                    (379, 355, 265, 63): "数量已足够",
                }, loading_frames=frames, wait_timeout=1600)
                self.assertTrue(success)
                self.assertEqual(len(calls), 1)
                self.assertGreater(calls[0][1], 8)

    def test_no_loading_symbol_is_required(self):
        success, calls = self.run_gate("点击前往获取主线制衣材料", {
            (379, 355, 265, 63): "数量已足够",
        }, loading_frames=())
        self.assertTrue(success)
        self.assertEqual(len(calls), 1)

    def test_persistent_loading_and_animation_share_bounded_wait(self):
        for kwargs in ({"loading_frames": range(100000)}, {"animated": True}):
            with self.subTest(kwargs=kwargs):
                started = time.monotonic()
                success, calls = self.run_gate("点击前往获取主线制衣材料", {
                    (379, 355, 265, 63): "数量已足够",
                }, **kwargs)
                self.assertFalse(success)
                self.assertEqual(calls, [])
                self.assertLess(time.monotonic() - started, 3)

    def test_get_materials_click_preserves_offset_and_has_no_fixed_wait(self):
        for branch in ("时空回廊", "心灵迷宫", "主线"):
            entry = PIPELINE["点击前往获取" + branch + "制衣材料"]
            self.assertEqual(entry["action"], {
                "type": "Click", "param": {
                    "target": True, "target_offset": [339, -16, -23, -6],
                },
            })
            self.assertEqual(entry["post_delay"], 0)


class NativeReadinessProcessTests(unittest.TestCase):
    def test_readiness_in_separate_client_process(self):
        # Other offline tests import maa.agent and select server mode globally.
        # Keep the native Tasker/controller in its own client process.
        code = (
            "from maa.library import Library; Library.version(); "
            "import unittest, sys; "
            "from tools.test_make_target_clothes_readiness import ClothingReadinessTests; "
            "suite = unittest.defaultTestLoader.loadTestsFromTestCase(ClothingReadinessTests); "
            "result = unittest.TextTestRunner(verbosity=2).run(suite); "
            "sys.exit(not result.wasSuccessful())"
        )
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=ROOT,
                                capture_output=True, encoding="utf-8", errors="replace", timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def load_tests(loader, tests, pattern):
    return loader.loadTestsFromTestCase(NativeReadinessProcessTests)


if __name__ == "__main__":
    unittest.main()
