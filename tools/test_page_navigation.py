"""Offline regression tests for public navigation."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from maa.agent.agent_server import AgentServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location("navigation_under_test", ROOT / "agent/custom/page_navigation.py")
navigation = importlib.util.module_from_spec(spec)
with patch.object(AgentServer, "register_custom_action", return_value=True):
    spec.loader.exec_module(navigation)


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.action = navigation.PageNavigateAction()

    def test_all_five_targets_route_via_solo(self):
        targets = ("竞技场", "搭配评选赛", "时空回廊", "主线故事", "心阶")
        for source in targets:
            for target in targets:
                if source == target:
                    continue
                route = self.action._shortest_path(navigation.DEFAULT_ROUTES, source, target)
                self.assertEqual([edge["to"] for edge in route], ["独自", target])
        for target in targets:
            route = self.action._shortest_path(navigation.DEFAULT_ROUTES, "主页面", target)
            self.assertEqual([edge["to"] for edge in route], ["独自", target])

    def test_no_fixed_sleep_on_ready_screen(self):
        self.action._screencap = Mock(return_value=object())
        self.action._intermediate_reason = Mock(return_value="")
        with patch.object(navigation.time, "sleep") as sleep:
            self.assertTrue(self.action._wait_while_intermediate(None, [], 15, .25))
        sleep.assert_not_called()

    def test_loading_and_transient_hits_reset_confirmation(self):
        self.action._screencap = Mock(return_value=object())
        self.action._intermediate_reason = Mock(side_effect=["加载", "", "", "", ""])
        self.action._recognize = Mock(side_effect=[True, False, True, True])
        with patch.object(navigation.time, "sleep"):
            self.assertTrue(self.action._wait_for_page(None, ["target"], "目标", [], 15, .25))
        self.assertEqual(self.action._screencap.call_count, 5)

    def test_destination_timeout_never_clicks(self):
        context = Mock()
        self.action._screencap = Mock(return_value=object())
        self.action._intermediate_reason = Mock(return_value="")
        self.action._recognize = Mock(return_value=False)
        self.assertFalse(self.action._wait_for_page(context, ["target"], "目标", [], 0, .25))
        context.run_task.assert_not_called()

    def test_failed_arrival_stops_route(self):
        self.action._wait_while_intermediate = Mock(return_value=True)
        self.action._detect_current_page = Mock(return_value="独自")
        self.action._run_task = Mock(return_value=True)
        self.action._wait_for_page = Mock(return_value=False)
        self.action._run_fallback_step = Mock()
        argv = SimpleNamespace(custom_action_param=json.dumps({"target": "竞技场"}))
        self.assertFalse(self.action.run(Mock(), argv))
        self.assertEqual(self.action._run_task.call_count, 1)
        self.action._run_fallback_step.assert_not_called()

    def test_missing_or_failed_task_result_is_failure(self):
        for result in (None, SimpleNamespace(status=SimpleNamespace(succeeded=False))):
            self.assertFalse(self.action._run_task(SimpleNamespace(run_task=lambda _: result), "node"))
        result = SimpleNamespace(status=SimpleNamespace(succeeded=True))
        self.assertTrue(self.action._run_task(SimpleNamespace(run_task=lambda _: result), "node"))

    def test_route_references_and_menu_guards(self):
        nodes = {}
        for path in (ROOT / "assets/resource/pipeline").rglob("*.json"):
            nodes.update(json.loads(path.read_text(encoding="utf-8")))
        for route in navigation.DEFAULT_ROUTES:
            self.assertIn(route["from"], navigation.DEFAULT_PAGES)
            self.assertIn(route["to"], navigation.DEFAULT_PAGES)
            for task in route["tasks"]:
                self.assertIn(task, nodes)
        for page_nodes in navigation.DEFAULT_PAGES.values():
            for node in page_nodes:
                self.assertIn(node, nodes)
        for action, ready in (("导航点击打开底部菜单", "导航底部菜单已打开"),
                              ("导航点击打开侧边菜单", "导航侧边菜单已打开"),
                              ("导航点击独自", "导航在独自页面")):
            self.assertEqual(nodes[action]["next"], [ready])
            self.assertEqual(nodes[action]["post_delay"], 0)
        self.assertEqual(nodes["导航在独自页面"]["recognition"]["param"]["expected"], ["主线故事"])
        for name in ("导航点击心灵迷宫", "导航点击至美逐星", "导航点击旅程回顾"):
            self.assertNotIn(name, nodes)


class MainStoryReturnTests(unittest.TestCase):
    def simulate(self, destinations):
        now = [0.0]
        state = ["导航在主线故事页面"]
        pending = [None]
        clicks = []
        params = [{}]
        context = Mock()
        context.clone.return_value = context

        def screenshot(*args):
            if pending[0] and now[0] >= pending[0][0]:
                state[0] = pending[0][1]
                pending[0] = None
            return state[0]

        def override(nodes):
            params[0] = nodes["导航点击返回"]["action"]["param"]["custom_action_param"]
            return True

        def click(*args):
            clicks.append(state[0])
            pending[0] = (now[0] + .5, destinations[len(clicks) - 1])
            state[0] = "加载"
            return SimpleNamespace(success=True)

        def run_task(name):
            ok = navigation.NavigationClickAction().run(context, SimpleNamespace(
                node_name=name, custom_action_param=json.dumps(params[0])))
            return SimpleNamespace(success=ok)

        context.override_pipeline.side_effect = override
        context.run_task.side_effect = run_task
        context.run_action_direct.side_effect = click
        context.run_recognition.return_value = SimpleNamespace(
            hit=True, box=SimpleNamespace(x=0, y=36, w=97, h=66))
        with patch.object(navigation.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(navigation.time, "sleep", side_effect=lambda dt: now.__setitem__(0, now[0] + dt)), \
             patch.object(navigation.PageNavigateAction, "_screencap", side_effect=screenshot), \
             patch.object(navigation.PageNavigateAction, "_recognize", side_effect=lambda ctx, image, name: image == name), \
             patch.object(navigation.PageNavigateAction, "_intermediate_reason", side_effect=lambda ctx, image, nodes: image == "加载"):
            ok = navigation.MainStoryReturnToSoloAction().run(context, SimpleNamespace())
        return ok, clicks, now[0]

    def test_direct_return_clicks_only_once(self):
        ok, clicks, _ = self.simulate(["导航在独自页面"])
        self.assertTrue(ok)
        self.assertEqual(clicks, ["导航在主线故事页面"])

    def test_optional_investigation_returns_again(self):
        ok, clicks, _ = self.simulate(["导航在调查行动页面", "导航在独自页面"])
        self.assertTrue(ok)
        self.assertEqual(clicks, ["导航在主线故事页面", "导航在调查行动页面"])

    def test_unknown_page_stops_without_extra_clicks(self):
        for destinations, expected_time in ((["未知"], 15), (["导航在调查行动页面", "未知"], 15.5)):
            ok, clicks, elapsed = self.simulate(destinations)
            self.assertFalse(ok)
            self.assertEqual(len(clicks), len(destinations))
            self.assertEqual(elapsed, expected_time)


class NavigationClickTests(unittest.TestCase):
    def simulate(self, source=lambda t: True, target=lambda t: False,
                 loading=lambda t: False, button=lambda t: True):
        now = [0.0]
        clicks = []
        context = Mock()
        context.run_recognition.side_effect = lambda name, image: SimpleNamespace(
            hit=button(now[0]), box=SimpleNamespace(x=int(now[0] * 10), y=20, w=30, h=40))
        def click(kind, params):
            clicks.append((now[0], params.target))
            return SimpleNamespace(success=True)
        context.run_action_direct.side_effect = click
        def recognize(self, ctx, image, name):
            return target(now[0]) if name == "target" else source(now[0])
        argv = SimpleNamespace(node_name="button", custom_action_param=json.dumps({
            "source_nodes": ["source"], "target_nodes": ["target"]}))
        with patch.object(navigation.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(navigation.time, "sleep", side_effect=lambda dt: now.__setitem__(0, now[0] + dt)), \
             patch.object(navigation.PageNavigateAction, "_screencap", return_value=object()), \
             patch.object(navigation.PageNavigateAction, "_recognize", new=recognize), \
             patch.object(navigation.PageNavigateAction, "_intermediate_reason", side_effect=lambda *a: "加载" if loading(now[0]) else ""):
            ok = navigation.NavigationClickAction().run(context, argv)
        return ok, clicks, context

    def test_retries_after_1_5_seconds_using_fresh_boxes(self):
        ok, clicks, _ = self.simulate(target=lambda t: t >= 2)
        self.assertTrue(ok)
        self.assertEqual([t for t, _ in clicks], [0, 1.5])
        self.assertNotEqual(clicks[0][1], clicks[1][1])

    def test_three_click_limit_includes_initial_click(self):
        ok, clicks, _ = self.simulate()
        self.assertFalse(ok)
        self.assertEqual([t for t, _ in clicks], [0, 1.5, 3])

    def test_no_clicks_during_loading(self):
        ok, clicks, _ = self.simulate(loading=lambda t: t < 2, target=lambda t: t >= 2.5)
        self.assertTrue(ok)
        self.assertEqual([t for t, _ in clicks], [2])

    def test_wrong_page_or_missing_button_never_clicks(self):
        for kwargs in ({"source": lambda t: False}, {"button": lambda t: False}):
            ok, clicks, _ = self.simulate(**kwargs)
            self.assertFalse(ok)
            self.assertEqual(clicks, [])

    def test_target_arrival_latched_through_loading_and_recognition_miss(self):
        ok, clicks, _ = self.simulate(target=lambda t: t == .5, loading=lambda t: .5 <= t < 2)
        self.assertTrue(ok)
        self.assertEqual([t for t, _ in clicks], [0])

    def test_already_at_target_does_not_click(self):
        ok, clicks, _ = self.simulate(target=lambda t: True)
        self.assertTrue(ok)
        self.assertEqual(clicks, [])

    def test_pipeline_guard_registration_and_references(self):
        nodes = json.loads((ROOT / "assets/resource/pipeline/navigation.json").read_text(encoding="utf-8"))
        all_nodes = {}
        for path in (ROOT / "assets/resource/pipeline").rglob("*.json"):
            all_nodes.update(json.loads(path.read_text(encoding="utf-8")))
        guarded = []
        for name, node in nodes.items():
            action = node.get("action")
            if not isinstance(action, dict):
                continue
            params = action.get("param", {})
            if params.get("custom_action") != "navigation_click":
                continue
            guarded.append(name)
            for key in ("source_nodes", "target_nodes"):
                self.assertTrue(params["custom_action_param"][key])
                for ref in params["custom_action_param"][key]:
                    self.assertIn(ref, all_nodes)
        self.assertEqual(len(guarded), 26)


if __name__ == "__main__":
    unittest.main()
