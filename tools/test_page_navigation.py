"""Offline regression tests for public navigation."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from maa.agent.agent_server import AgentServer
from maa.define import Rect
from tools.validate_schema import load_jsonc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location("navigation_under_test", ROOT / "agent/custom/page_navigation.py")
navigation = importlib.util.module_from_spec(spec)
with patch.object(AgentServer, "register_custom_action", return_value=True):
    spec.loader.exec_module(navigation)


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.action = navigation.PageNavigateAction()

    def test_detection_keeps_target_ahead_of_parent_and_hubs(self):
        self.action._screencap = Mock(return_value=object())
        self.action._recognize = Mock(return_value=True)
        page = self.action._detect_current_page(
            None, navigation.DEFAULT_PAGES, 3, 1,
            ordered_pages=self.action._build_detection_order(
                navigation.DEFAULT_PAGES, navigation.DEFAULT_ROUTES, "美甲",
            ), target_page="美甲",
        )
        self.assertEqual(page, "美甲")
        self.assertEqual(self.action._recognize.call_count, 1)
        self.assertEqual(self.action._recognize.call_args.args[2], "导航在美甲页面")

    def test_custom_route_source_is_recognized_before_hubs(self):
        self.action._wait_while_intermediate = Mock(return_value=True)
        self.action._screencap = Mock(return_value=object())
        self.action._recognize = Mock(side_effect=lambda ctx, image, node: node == "自定义来源识别")
        self.action._run_task = Mock(return_value=True)
        self.action._wait_for_page = Mock(return_value=True)
        argv = SimpleNamespace(custom_action_param=json.dumps({
            "target": "自定义目标", "max_steps": 1,
            "pages": {"自定义目标": ["自定义目标识别"], "自定义来源": ["自定义来源识别"]},
            "routes": [{"from": "自定义来源", "to": "自定义目标", "tasks": ["自定义跳转"]}],
        }))
        self.assertTrue(self.action.run(Mock(), argv))
        self.assertEqual([call.args[2] for call in self.action._recognize.call_args_list], [
            "自定义目标识别", "自定义来源识别",
        ])
        self.assertEqual(self.action._run_task.call_args.args[1], "自定义跳转")
        self.action._wait_for_page.assert_called_once()

    def test_detection_keeps_reachable_pages_and_skips_unreachable_hubs(self):
        pages = {"其他页面": ["other"], "主页面": ["home"], "目标": ["target"],
                 "设计中心": ["unreachable"]}
        routes = [{"from": "其他页面", "to": "主页面"}, {"from": "主页面", "to": "目标"}]
        ordered_pages = self.action._build_detection_order(pages, routes, "目标")
        self.assertEqual(ordered_pages, ["目标", "主页面", "其他页面"])
        self.action._screencap = Mock(return_value=object())
        for hit, expected, checks in (("home", "主页面", ["target", "home"]),
                                      ("other", "其他页面", ["target", "home", "other"])):
            with self.subTest(hit=hit):
                self.action._recognize = Mock(side_effect=lambda ctx, image, node: node == hit)
                self.assertEqual(self.action._detect_current_page(
                    None, pages, 1, 1, ordered_pages=ordered_pages, target_page="目标",
                ), expected)
                self.assertEqual([call.args[2] for call in self.action._recognize.call_args_list], checks)

    def test_detection_retries_use_fresh_images_and_deduplicate_only_per_image(self):
        pages = {"目标": ["shared", "shared"], "其他页面": ["shared", "other"]}
        routes = [{"from": "其他页面", "to": "目标"}]
        self.action._screencap = Mock(side_effect=["first", "second"])
        self.action._recognize = Mock(side_effect=lambda ctx, image, node: image == "second" and node == "shared")
        with patch.object(navigation.time, "sleep") as sleep:
            self.assertEqual(self.action._detect_current_page(
                None, pages, 3, 1,
                ordered_pages=self.action._build_detection_order(pages, routes, "目标"),
                target_page="目标",
            ), "目标")
        self.assertEqual([(call.args[1], call.args[2]) for call in self.action._recognize.call_args_list], [
            ("first", "shared"), ("first", "other"), ("second", "shared"),
        ])
        sleep.assert_called_once_with(1)

    def test_unknown_page_retries_only_reachable_candidates_without_actions(self):
        context = Mock()
        self.action._screencap = Mock(side_effect=["first", "second", "third"])
        self.action._recognize = Mock(return_value=False)
        with patch.object(navigation.time, "sleep") as sleep:
            self.assertIsNone(self.action._detect_current_page(
                context, {"目标": ["target"], "其他页面": ["other"]}, 3, 1,
                ordered_pages=["目标"], target_page="目标",
            ))
        self.assertEqual(self.action._screencap.call_count, 3)
        self.assertEqual(self.action._recognize.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        context.run_task.assert_not_called()
        context.run_action_direct.assert_not_called()

    def test_default_targets_exclude_pages_without_a_route_to_target(self):
        for target, expected in (("主页面", {"主页面"}), ("任务", {"任务", "主页面"}),
                                 ("美甲", {"美甲", "主页面", "设计中心", "制衣", "情报屋", "美甲我的店铺"})):
            with self.subTest(target=target):
                order = self.action._build_detection_order(
                    navigation.DEFAULT_PAGES, navigation.DEFAULT_ROUTES, target,
                )
                self.assertEqual(order[0], target)
                self.assertEqual(set(order), expected)
                self.assertEqual(len(order), len(expected))

    def test_cycles_and_duplicate_routes_do_not_add_unreachable_pages(self):
        pages = {name: [name] for name in ("目标", "父页", "祖先", "无关A", "无关B")}
        routes = [{"from": source, "to": dest} for source, dest in (
            ("父页", "目标"), ("父页", "目标"), ("祖先", "父页"), ("父页", "祖先"),
            ("目标", "目标"), ("无关A", "无关B"), ("无关B", "无关A"),
        )]
        self.assertEqual(self.action._build_detection_order(pages, routes, "目标"),
                         ["目标", "父页", "祖先"])

    def test_custom_aliases_and_multileg_routes_are_merged_before_pruning(self):
        params = {
            "aliases": {"终点别名": "自定义目标", "来源别名": "自定义来源"},
            "pages": {"终点别名": ["target"], "来源别名": ["source"], "自定义中转": ["middle"]},
            "routes": [{"from": "来源别名", "to": "自定义中转", "tasks": ["first"]},
                       {"from": "自定义中转", "to": "终点别名", "tasks": ["second"]}],
        }
        order = self.action._build_detection_order(
            self.action._build_pages(params), self.action._build_routes(params),
            self.action._normalize_page("终点别名", params),
        )
        self.assertEqual(order, ["自定义目标", "自定义中转", "自定义来源"])

    def test_arena_custom_entry_keeps_all_sources_with_existing_routes(self):
        nodes = json.loads((ROOT / "assets/resource/pipeline/arena.json").read_text(encoding="utf-8"))
        params = nodes["竞技场任务链准备"]["action"]["param"]["custom_action_param"]
        order = self.action._build_detection_order(
            self.action._build_pages(params), self.action._build_routes(params), params["target"],
        )
        self.assertEqual(order[:2], ["竞技场入口区域", "独自"])
        self.assertEqual(set(order), {"竞技场入口区域", "独自", "结伴", "主页面", "竞技场",
                                      "主线故事", "时空回廊", "心阶", "搭配评选赛"})

    def test_scan_logs_candidate_and_node_counts_once_per_attempt(self):
        self.action._screencap = Mock(return_value="screen")
        self.action._recognize = Mock(return_value=False)
        with patch.object(navigation.logger, "debug") as log:
            self.assertIsNone(self.action._detect_current_page(
                None, navigation.DEFAULT_PAGES, 1, 1,
                ordered_pages=["主页面"], target_page="主页面",
            ))
        log.assert_called_once()
        self.assertIn(f"candidates=1/{len(navigation.DEFAULT_PAGES)}, checked_nodes=1, detected=None", log.call_args.args[0])

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

    def prepare_route(self, initial_page):
        self.action._wait_while_intermediate = Mock(return_value=True)
        self.action._detect_current_page = Mock(return_value=initial_page)
        self.action._run_task = Mock(return_value=True)
        self.action._wait_for_page = Mock(return_value=True)
        self.action._run_fallback_step = Mock(return_value=True)

    def test_final_arrival_finishes_within_one_step_without_redetection(self):
        self.prepare_route("独自")
        argv = SimpleNamespace(custom_action_param=json.dumps({
            "target": "竞技场", "max_steps": 1,
        }))
        self.assertTrue(self.action.run(Mock(), argv))
        self.action._detect_current_page.assert_called_once()
        self.action._wait_while_intermediate.assert_called_once()
        self.action._wait_for_page.assert_called_once()
        self.action._run_fallback_step.assert_not_called()

    def test_final_arrival_still_requires_two_consecutive_ready_screens(self):
        argv = SimpleNamespace(custom_action_param=json.dumps({
            "target": "竞技场", "max_steps": 1,
        }))
        for screens in (["loading", "target", "missing", "target", "target"],
                        ["target", "loading", "target", "target"]):
            with self.subTest(screens=screens):
                self.prepare_route("独自")
                # Exercise the actual confirmation loop after route completion.
                self.action._wait_for_page = navigation.PageNavigateAction._wait_for_page.__get__(self.action)
                self.action._screencap = Mock(side_effect=screens)
                self.action._intermediate_reason = Mock(side_effect=lambda ctx, image, nodes: image == "loading")
                self.action._recognize = Mock(side_effect=lambda ctx, image, node: image == "target")
                with patch.object(navigation.time, "sleep"):
                    self.assertTrue(self.action.run(Mock(), argv))
                self.assertEqual(self.action._screencap.call_count, len(screens))
                self.action._detect_current_page.assert_called_once()
                self.action._run_task.assert_called_once()

    def test_multileg_route_reuses_confirmation_and_keeps_between_task_waits(self):
        self.prepare_route("主页面")
        argv = SimpleNamespace(custom_action_param=json.dumps({
            "target": "竞技场", "max_steps": 2,
        }))
        self.assertTrue(self.action.run(Mock(), argv))
        self.assertEqual([call.args[1] for call in self.action._run_task.call_args_list], [
            "导航确保底部菜单打开", "导航点击开始旅程入口", "导航确保独自页面",
            "导航点击钻石竞技场",
        ])
        self.assertEqual([call.args[2] for call in self.action._wait_for_page.call_args_list], [
            "独自", "竞技场",
        ])
        self.action._detect_current_page.assert_called_once()
        # Initial loading check plus the two waits between homepage route tasks.
        self.assertEqual(self.action._wait_while_intermediate.call_count, 3)

    def test_between_task_loading_timeout_stops_before_next_click(self):
        self.prepare_route("主页面")
        self.action._wait_while_intermediate.side_effect = [True, False]
        argv = SimpleNamespace(custom_action_param=json.dumps({"target": "竞技场"}))
        self.assertFalse(self.action.run(Mock(), argv))
        self.action._run_task.assert_called_once()
        self.assertEqual(self.action._run_task.call_args.args[1], "导航确保底部菜单打开")
        self.action._wait_for_page.assert_not_called()
        self.action._run_fallback_step.assert_not_called()

    def test_fallback_requires_fresh_page_detection(self):
        self.prepare_route(None)
        self.action._detect_current_page.side_effect = [None, "主页面"]
        argv = SimpleNamespace(custom_action_param=json.dumps({"target": "任务"}))
        self.assertTrue(self.action.run(Mock(), argv))
        self.action._run_fallback_step.assert_called_once()
        self.assertEqual(self.action._detect_current_page.call_count, 2)
        self.assertEqual([call.args[1] for call in self.action._run_task.call_args_list], [
            "导航确保侧边菜单打开", "导航点击任务入口",
        ])

    def test_consecutive_returns_scan_only_home_on_fresh_images(self):
        state = ["first_detail"]
        context = Mock()
        context.tasker.stopping = False
        context.clone.return_value = context
        context.run_recognition.return_value = SimpleNamespace(hit=True, box=Rect(24, 51, 36, 37))
        destinations = iter(["second_detail", "home"])

        def return_task(*args, **kwargs):
            state[0] = next(destinations)
            return SimpleNamespace(status=SimpleNamespace(succeeded=True))

        context.run_task.side_effect = return_task
        self.action._wait_while_intermediate = Mock(return_value=True)
        self.action._intermediate_reason = Mock(return_value="")
        self.action._screencap = Mock(side_effect=lambda ctx: state[0])
        self.action._recognize = Mock(side_effect=lambda ctx, image, node:
                                      node == "是否是主页面" and image == "home")
        self.action._build_detection_order = Mock(wraps=self.action._build_detection_order)
        argv = SimpleNamespace(custom_action_param=json.dumps({"target": "主页", "max_steps": 3}))
        with patch.object(navigation.time, "sleep"):
            self.assertTrue(self.action.run(context, argv))
        self.action._build_detection_order.assert_called_once()
        self.assertEqual(context.run_task.call_count, 2)
        self.assertEqual([(call.args[1], call.args[2]) for call in self.action._recognize.call_args_list], [
            ("first_detail", "是否是主页面"), ("first_detail", "是否是主页面"),
            ("second_detail", "是否是主页面"), ("second_detail", "是否是主页面"),
            ("home", "是否是主页面"), ("home", "是否是主页面"), ("home", "是否是主页面"),
        ])

    def test_target_reached_during_scan_is_confirmed_without_return(self):
        state = ["detail"]
        context = Mock()
        context.tasker.stopping = False
        self.action._wait_while_intermediate = Mock(return_value=True)
        self.action._intermediate_reason = Mock(return_value="")
        self.action._screencap = Mock(side_effect=lambda ctx: state[0])

        def recognize(ctx, image, node):
            # First scan misses its snapshot; the following fallback snapshot is home.
            state[0] = "home"
            return image == "home"

        self.action._recognize = Mock(side_effect=recognize)
        argv = SimpleNamespace(custom_action_param=json.dumps({"target": "主页面", "max_steps": 2}))
        with patch.object(navigation.time, "sleep"):
            self.assertTrue(self.action.run(context, argv))
        self.assertEqual(self.action._screencap.call_count, 5)
        context.run_recognition.assert_not_called()
        context.run_task.assert_not_called()
        context.run_action_direct.assert_not_called()

    def test_direct_target_hit_requires_two_consecutive_non_loading_screens(self):
        for screens in (["home", "home", "missing", "home", "home"],
                        ["home", "home", "loading", "home", "home"]):
            with self.subTest(screens=screens):
                self.action._wait_while_intermediate = Mock(return_value=True)
                self.action._screencap = Mock(side_effect=screens)
                self.action._intermediate_reason = Mock(side_effect=lambda ctx, image, nodes:
                                                        "在加载页面" if image == "loading" else "")
                self.action._recognize = Mock(side_effect=lambda ctx, image, node: image == "home")
                self.action._run_task = Mock()
                self.action._run_fallback_step = Mock()
                argv = SimpleNamespace(custom_action_param=json.dumps({"target": "主页面", "max_steps": 1}))
                with patch.object(navigation.time, "sleep"):
                    self.assertTrue(self.action.run(Mock(), argv))
                self.assertEqual(self.action._screencap.call_count, len(screens))
                self.action._run_task.assert_not_called()
                self.action._run_fallback_step.assert_not_called()

    def test_direct_target_confirmation_timeout_does_not_fallback(self):
        self.prepare_route("主页面")
        self.action._wait_for_page.return_value = False
        argv = SimpleNamespace(custom_action_param=json.dumps({"target": "主页面"}))
        self.assertFalse(self.action.run(Mock(), argv))
        self.action._run_task.assert_not_called()
        self.action._run_fallback_step.assert_not_called()

    def test_unknown_pages_stop_at_max_steps_without_warning_each_scan(self):
        self.prepare_route(None)
        argv = SimpleNamespace(custom_action_param=json.dumps({"target": "主页面", "max_steps": 2}))
        with patch.object(navigation.logger, "warning") as warning:
            self.assertFalse(self.action.run(Mock(), argv))
        warning.assert_not_called()
        self.assertEqual(self.action._run_fallback_step.call_count, 2)
        for call in self.action._detect_current_page.call_args_list:
            self.assertEqual(call.kwargs["ordered_pages"], ["主页面"])

    def test_missing_safe_fallback_fails_without_route_action(self):
        self.prepare_route(None)
        self.action._run_fallback_step.return_value = False
        argv = SimpleNamespace(custom_action_param=json.dumps({"target": "任务"}))
        self.assertFalse(self.action.run(Mock(), argv))
        self.action._run_task.assert_not_called()
        self.action._run_fallback_step.assert_called_once()

    def test_unknown_page_uses_one_scan_before_fallback_by_default(self):
        for options, expected in (({}, 1), ({"retry_detect": 2}, 2)):
            with self.subTest(options=options):
                self.prepare_route(None)
                self.action._detect_current_page.side_effect = [None, "任务"]
                argv = SimpleNamespace(custom_action_param=json.dumps({"target": "任务", **options}))
                self.assertTrue(self.action.run(Mock(), argv))
                self.assertEqual(self.action._detect_current_page.call_args_list[0].kwargs["retry"], expected)
                self.action._run_fallback_step.assert_called_once()

    def test_confirmed_page_is_not_shared_between_invocations(self):
        self.prepare_route("独自")
        self.action._detect_current_page.side_effect = ["独自", "竞技场"]
        argv = SimpleNamespace(custom_action_param=json.dumps({"target": "竞技场"}))
        self.assertTrue(self.action.run(Mock(), argv))
        self.assertTrue(self.action.run(Mock(), argv))
        self.assertEqual(self.action._detect_current_page.call_count, 2)
        self.action._run_task.assert_called_once()
        self.assertEqual(self.action._wait_for_page.call_count, 2)

    def test_route_task_failure_never_confirms_arrival(self):
        self.prepare_route("独自")
        self.action._run_task.return_value = False
        argv = SimpleNamespace(custom_action_param=json.dumps({"target": "竞技场"}))
        self.assertFalse(self.action.run(Mock(), argv))
        self.action._wait_for_page.assert_not_called()
        self.action._run_fallback_step.assert_not_called()

    def test_missing_or_failed_task_result_is_failure(self):
        for result in (None, SimpleNamespace(status=SimpleNamespace(succeeded=False))):
            self.assertFalse(self.action._run_task(SimpleNamespace(run_task=lambda _: result), "node"))
        result = SimpleNamespace(status=SimpleNamespace(succeeded=True))
        self.assertTrue(self.action._run_task(SimpleNamespace(run_task=lambda _: result), "node"))

    def test_route_references_and_menu_guards(self):
        nodes = {}
        for path in (ROOT / "assets/resource/pipeline").rglob("*.json"):
            nodes.update(load_jsonc(path))
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


class NavigationFallbackTests(unittest.TestCase):
    def setUp(self):
        self.action = navigation.PageNavigateAction()
        self.context = Mock()
        self.context.tasker.stopping = False
        self.context.clone.return_value = self.context
        self.box = Rect(24, 51, 36, 37)
        self.context.run_recognition.return_value = SimpleNamespace(hit=True, box=self.box, raw_detail={"best": {}})
        self.context.run_task.return_value = SimpleNamespace(status=SimpleNamespace(succeeded=True))
        self.action._screencap = Mock(return_value="fresh")
        self.action._intermediate_reason = Mock(return_value="")

    def test_fresh_target_hit_skips_return_and_other_fallbacks(self):
        self.action._recognize = Mock(return_value=True)
        self.assertTrue(self.action._run_fallback_step(
            self.context, ["点击返回", "点击esc"], target_nodes=["target"],
        ))
        self.action._recognize.assert_called_once_with(self.context, "fresh", "target")
        self.context.run_recognition.assert_not_called()
        self.context.run_task.assert_not_called()

    def test_target_miss_keeps_recognized_return_action(self):
        self.action._recognize = Mock(return_value=False)
        self.assertTrue(self.action._run_fallback_step(
            self.context, ["点击返回"], target_nodes=["target"],
        ))
        self.action._recognize.assert_called_once_with(self.context, "fresh", "target")
        self.context.run_recognition.assert_called_once_with("点击返回", "fresh")
        self.context.run_task.assert_called_once()

    def test_loading_timeout_never_probes_target_or_clicks(self):
        self.action._intermediate_reason.return_value = "在加载页面"
        self.action._recognize = Mock(return_value=True)
        self.assertFalse(self.action._run_fallback_step(
            self.context, ["点击返回"], timeout=0, target_nodes=["target"],
        ))
        self.action._recognize.assert_not_called()
        self.context.run_task.assert_not_called()

    def test_return_reuses_box_and_only_overrides_local_pre_delay(self):
        self.assertTrue(self.action._run_fallback_step(self.context, ["点击返回"]))
        self.context.run_recognition.assert_called_once_with("点击返回", "fresh")
        self.context.run_task.assert_called_once_with(
            "点击返回", pipeline_override={"点击返回": {
                "recognition": "DirectHit", "action": {"param": {"target": [24, 51, 36, 37]}},
                "pre_delay": 0,
            }},
        )
        self.context.clone.assert_called_once()

    def test_loading_after_scan_finishes_with_redetection_before_any_return(self):
        self.action._screencap.side_effect = ["loading", "fresh"]
        self.action._intermediate_reason.side_effect = ["在加载页面", ""]
        with patch.object(navigation.time, "sleep"):
            self.assertTrue(self.action._run_fallback_step(self.context, ["点击返回"]))
        self.context.run_recognition.assert_not_called()
        self.context.run_task.assert_not_called()

    def test_loading_timeout_and_cancellation_never_click(self):
        self.action._intermediate_reason.return_value = "在加载页面"
        self.assertFalse(self.action._run_fallback_step(self.context, ["点击返回"], timeout=0))
        self.context.run_recognition.assert_not_called()
        self.context.tasker.stopping = True
        self.assertFalse(self.action._run_fallback_step(self.context, ["点击返回"]))
        self.context.run_task.assert_not_called()

    def test_missing_or_invalid_button_never_clicks(self):
        for result in (None, SimpleNamespace(hit=False), SimpleNamespace(hit=True, box=None),
                       SimpleNamespace(hit=True, box=Rect(1, 2, 0, 4))):
            with self.subTest(result=result):
                self.context.run_recognition.return_value = result
                self.assertFalse(self.action._run_fallback_step(self.context, ["点击返回"]))
                self.context.run_task.assert_not_called()

    def test_failed_return_action_does_not_click_another_stale_candidate(self):
        for result in (None, SimpleNamespace(status=SimpleNamespace(succeeded=False))):
            self.context.run_task.reset_mock()
            self.context.run_task.return_value = result
            self.assertFalse(self.action._run_fallback_step(self.context, ["点击返回", "点击esc"]))
            self.context.run_task.assert_called_once()
            self.assertEqual(self.context.run_task.call_args.args[0], "点击返回")

    def test_other_fallbacks_keep_their_task_chains(self):
        self.context.run_recognition.return_value = SimpleNamespace(hit=False)
        self.action._recognize = Mock(return_value=True)
        self.action._run_task = Mock(return_value=True)
        self.assertTrue(self.action._run_fallback_step(self.context, ["点击返回", "点击esc"]))
        self.action._run_task.assert_called_once_with(self.context, "点击esc")


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
                 loading=lambda t: False, button=lambda t: True, params=None):
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
        options = {"source_nodes": ["source"], "target_nodes": ["target"]}
        options.update(params or {})
        argv = SimpleNamespace(node_name="button", custom_action_param=json.dumps(options))
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

    def test_dreamweaver_loader_is_local_and_blocks_clicks_until_ready(self):
        dream_nodes = load_jsonc(ROOT / "assets/resource/pipeline/dreamweaver_space.json")
        nav_nodes = json.loads((ROOT / "assets/resource/pipeline/navigation.json").read_text(encoding="utf-8"))
        entry_params = dream_nodes["织梦空间任务链准备"]["action"]["param"]["custom_action_param"]
        click_params = nav_nodes["导航识别并点击织梦空间"]["action"]["param"]["custom_action_param"]
        loader = "织梦空间专属加载中"
        self.assertNotIn(loader, navigation.DEFAULT_WAIT_RECOGNITIONS)
        self.assertNotIn(loader, navigation.PageNavigateAction()._build_wait_nodes({}))
        self.assertIn(loader, navigation.PageNavigateAction()._build_wait_nodes(entry_params))

        for target_seen_while_loading in (False, True):
            with self.subTest(target_seen_while_loading=target_seen_while_loading):
                now = [0.0]
                context = Mock()
                def recognize(self, ctx, image, name):
                    if name == loader:
                        return now[0] < 2
                    if name == "导航在织梦空间页面":
                        return target_seen_while_loading or now[0] >= 2
                    return name == "是否是主页面"
                with patch.object(navigation.time, "monotonic", side_effect=lambda: now[0]), \
                     patch.object(navigation.time, "sleep", side_effect=lambda dt: now.__setitem__(0, now[0] + dt)), \
                     patch.object(navigation.PageNavigateAction, "_screencap", return_value=object()), \
                     patch.object(navigation.PageNavigateAction, "_recognize", new=recognize):
                    ok = navigation.NavigationClickAction().run(context, SimpleNamespace(
                        node_name="导航识别并点击织梦空间", custom_action_param=json.dumps(click_params)))
                self.assertTrue(ok)
                self.assertEqual(now[0], 2)
                context.run_action_direct.assert_not_called()

    def test_friend_allows_slow_loading_without_extra_clicks(self):
        nodes = json.loads((ROOT / "assets/resource/pipeline/navigation.json").read_text(encoding="utf-8"))
        params = nodes["导航点击好友"]["action"]["param"]["custom_action_param"]
        ok, clicks, _ = self.simulate(target=lambda t: t >= 22,
                                     loading=lambda t: 0 < t < 22,
                                     params={"timeout": params["timeout"]})
        self.assertTrue(ok)
        self.assertEqual([t for t, _ in clicks], [0])
        ok, clicks, _ = self.simulate(target=lambda t: t >= 22,
                                     loading=lambda t: 0 < t < 22)
        self.assertFalse(ok)
        self.assertEqual([t for t, _ in clicks], [0])

    def test_friend_loading_stops_at_thirty_seconds(self):
        ok, clicks, _ = self.simulate(loading=lambda t: t > 0,
                                     target=lambda t: t >= 31, params={"timeout": 30000})
        self.assertFalse(ok)
        self.assertEqual([t for t, _ in clicks], [0])

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
            all_nodes.update(load_jsonc(path))
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
        self.assertEqual(len(guarded), 27)


if __name__ == "__main__":
    unittest.main()
