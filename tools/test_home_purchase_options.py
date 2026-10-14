"""Home purchase quantities, zero-count routing, and native override checks."""

import copy
import itertools
import json
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = json.loads((ROOT / "assets/resource/pipeline/back_home.json").read_text(encoding="utf-8"))
INTERFACE = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
SHARED_NODES = json.loads((ROOT / "assets/resource/pipeline/tools.json").read_text(encoding="utf-8"))
# Option, routing node, selection node, purchase node, maximum, default.
PRODUCTS = (
    ("采购番茄炒蛋次数", "采购番茄炒蛋分流", "点击番茄炒蛋", "点击购买番茄炒蛋", 3, 3),
    ("采购电影票次数", "采购电影票分流", "点击电影票", "点击购买电影票", 2, 2),
    ("采购调色板次数", "采购调色板分流", "点击缤纷调色板", "点击购买缤纷调色板", 2, 2),
    ("采购晴雨伞次数", "采购晴雨伞分流", "点击晴雨伞", "点击购买晴雨伞", 2, 1),
)
EXIT = "点击空白-日常采购"
COMBINATIONS = tuple(itertools.product(*(range(product[4] + 1) for product in PRODUCTS)))


def selected_overrides(quantities):
    overrides = {}
    for product, quantity in zip(PRODUCTS, quantities):
        option = INTERFACE["option"][product[0]]
        case = next(case for case in option["cases"] if case["name"] == str(quantity))
        for name, fields in case["pipeline_override"].items():
            # These cases only replace top-level next/repeat fields.
            overrides.setdefault(name, {}).update(copy.deepcopy(fields))
    return overrides


def configured_pipeline(quantities):
    pipeline = copy.deepcopy(PIPELINE)
    for name, fields in selected_overrides(quantities).items():
        pipeline[name].update(fields)
    return pipeline


def ready_path(get_node):
    """Follow the first ready normal edge, with no popup or scrolling needed."""
    current = "采购番茄炒蛋分流"
    path = []
    while current != EXIT:
        if current in path:
            raise AssertionError(f"Purchase route loops at {current}: {path}")
        path.append(current)
        successors = get_node(current).get("next", [])
        normal_edges = [edge["name"] if isinstance(edge, dict) else edge for edge in successors
                        if not (edge.get("jump_back", False) if isinstance(edge, dict)
                                else edge.startswith("[JumpBack]"))]
        if not normal_edges:
            raise AssertionError(f"Purchase route ends before exit: {path}")
        current = normal_edges[0]
    return path


class HomePurchaseOptionTests(unittest.TestCase):
    def test_choices_defaults_and_task_registration(self):
        task = next(task for task in INTERFACE["task"] if task["name"] == "回家")
        for option_name, _, _, _, maximum, default in PRODUCTS:
            option = INTERFACE["option"][option_name]
            self.assertIn(option_name, task["option"])
            self.assertEqual(option["type"], "select")
            self.assertEqual([case["name"] for case in option["cases"]],
                             [str(quantity) for quantity in range(maximum + 1)])
            self.assertEqual(option["default_case"], str(default))

    def test_cases_only_change_their_own_route_and_quantity(self):
        for option_name, gate, _, purchase, _, _ in PRODUCTS:
            for case in INTERFACE["option"][option_name]["cases"]:
                with self.subTest(option=option_name, case=case["name"]):
                    for name, fields in case["pipeline_override"].items():
                        self.assertIn(name, (gate, purchase))
                        self.assertIn(name, PIPELINE)
                        self.assertLessEqual(set(fields), {"next", "repeat"})
                        for edge in fields.get("next", []):
                            self.assertIn(edge.removeprefix("[JumpBack]"), PIPELINE.keys() | SHARED_NODES.keys())
                    if case["name"] != "0":
                        self.assertGreater(case["pipeline_override"][purchase]["repeat"], 0)

    def test_all_108_quantity_combinations_skip_zero_and_preserve_order(self):
        for quantities in COMBINATIONS:
            with self.subTest(quantities=quantities):
                pipeline = configured_pipeline(quantities)
                path = ready_path(pipeline.__getitem__)
                expected = []
                for product, quantity in zip(PRODUCTS, quantities):
                    _, _, selection, purchase, _, _ = product
                    self.assertEqual(selection in path, quantity > 0)
                    self.assertEqual(purchase in path, quantity > 0)
                    if quantity:
                        self.assertEqual(pipeline[purchase].get("repeat", 1), quantity)
                        expected.append(purchase)
                self.assertEqual([name for name in path if name.startswith("点击购买")], expected)
                self.assertEqual("点击采购零食" in path, quantities[0] > 0)

    def test_study_setting_is_independent_of_purchase_quantities(self):
        for quantities in COMBINATIONS:
            for case in INTERFACE["option"]["是否做自习室好感任务"]["cases"]:
                with self.subTest(quantities=quantities, study=case["name"]):
                    pipeline = configured_pipeline(quantities)
                    for name, fields in case.get("pipeline_override", {}).items():
                        pipeline[name].update(fields)
                    ready_path(pipeline.__getitem__)
                    self.assertEqual(pipeline[EXIT]["next"],
                                     ["stop"] if case["name"] == "No" else ["点击暖心传音"])

    def test_each_product_is_selected_by_text_center_before_gold_purchase(self):
        for _, _, selection, purchase, _, _ in PRODUCTS:
            selected = PIPELINE[selection]
            self.assertEqual(selected["recognition"]["type"], "OCR")
            self.assertEqual(selected["action"]["param"], {"custom_action": "click_center"})
            self.assertEqual(selected["next"], [purchase])
            recognition = PIPELINE[purchase]["recognition"]
            self.assertEqual(recognition["type"], "TemplateMatch")
            self.assertEqual(recognition["param"]["template"], "home_gold_corn.png")
            self.assertEqual(recognition["param"]["roi"], [351, 936, 369, 145])

    def test_swipe_and_popups_return_to_their_dispatcher(self):
        self.assertEqual(PIPELINE["采购晴雨伞分流"]["next"], [
            "点击晴雨伞", "[JumpBack]点击确定活力不足", "[JumpBack]点击体力用尽确定",
            "[JumpBack]上划日常采购",
        ])
        for leaf in ("上划日常采购", "点击确定活力不足"):
            self.assertFalse(PIPELINE[leaf].get("next"))

    def test_native_overrides_and_recorded_pipeline_execution(self):
        # Keep the native client independent of AgentServer imports in other tests.
        code = ("from maa.library import Library; Library.version(); "
                "from tools.test_home_purchase_options import native_audit; native_audit()")
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def native_audit():
    import numpy as np
    from maa.controller import CustomController
    from maa.custom_action import CustomAction
    from maa.resource import Resource
    from maa.tasker import Tasker, LoggingLevelEnum

    Tasker.set_stdout_level(LoggingLevelEnum.Error)
    resource = Resource()
    assert resource.post_bundle(ROOT / "assets/resource").wait().succeeded
    for quantities in COMBINATIONS:
        assert resource.override_pipeline(selected_overrides(quantities))
        path = ready_path(resource.get_node_data)
        for product, quantity in zip(PRODUCTS, quantities):
            _, _, selection, purchase, _, _ = product
            assert (selection in path) == (quantity > 0), (quantities, path)
            assert (purchase in path) == (quantity > 0), (quantities, path)
            node = resource.get_node_data(purchase)
            if quantity:
                assert node["repeat"] == quantity, (quantities, node)
            assert node["recognition"]["type"] == "TemplateMatch"
            assert node["recognition"]["param"]["template"] == ["home_gold_corn.png"]

    class RecordingController(CustomController):
        def connect(self): return True
        def request_uuid(self): return "home-options-offline"
        def get_features(self): return 0
        def screencap(self): return np.zeros((1280, 720, 3), dtype=np.uint8)
        def forbidden(self, *args): raise AssertionError("Unexpected device action")
        click = swipe = touch_down = touch_move = touch_up = forbidden
        click_key = input_text = start_app = stop_app = key_down = key_up = forbidden

    executed = []

    class Record(CustomAction):
        def run(self, context, argv):
            executed.append(argv.node_name)
            return True

    assert resource.override_pipeline(PIPELINE)
    assert resource.register_custom_action("home_options_record", Record())
    fixtures = {}
    for name in PIPELINE:
        fixtures[name] = {
            "recognition": "DirectHit",
            "inverse": name in ("点击确定活力不足", "点击体力用尽确定"),
            "action": {"type": "Custom", "param": {"custom_action": "home_options_record"}},
            "pre_delay": 0, "post_delay": 0, "repeat_delay": 0,
        }
    fixtures[EXIT]["next"] = []  # End before self-study; no device is connected.
    fixtures["点击体力用尽确定"] = {"recognition": "DirectHit", "inverse": True}
    assert resource.override_pipeline(fixtures)
    controller = RecordingController()
    assert controller.post_connection().wait().succeeded
    tasker = Tasker()
    assert tasker.bind(resource, controller)
    samples = ((3, 2, 2, 1), (0, 0, 0, 0), (3, 2, 2, 2), (3, 0, 0, 0),
               (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2), (3, 2, 2, 1))
    for quantities in samples:
        executed.clear()
        assert resource.override_pipeline(selected_overrides(quantities))
        assert tasker.post_task("采购番茄炒蛋分流").wait().succeeded
        counts = Counter(executed)
        for product, quantity in zip(PRODUCTS, quantities):
            _, _, selection, purchase, _, _ = product
            assert counts[selection] == int(quantity > 0), (quantities, executed)
            assert counts[purchase] == quantity, (quantities, executed)
        assert executed[-1] == EXIT, executed
    print("PASS: 108 native option combinations and 8 recorded purchase flows")


if __name__ == "__main__":
    unittest.main()
