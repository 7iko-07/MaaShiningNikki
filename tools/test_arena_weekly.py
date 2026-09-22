"""离线验证竞技场周周期、账号隔离及成功后记账，不连接游戏。"""

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from maa.agent.agent_server import AgentServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
from utils import arena_state as state


def load_action(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"agent/custom/{name}.py")
    module = importlib.util.module_from_spec(spec)
    with patch.object(AgentServer, "register_custom_action", return_value=True):
        spec.loader.exec_module(module)
    return module


weekly = load_action("arena_weekly")
switch = load_action("switch_account")
arena = load_action("arena_action")


def beijing(value):
    return datetime.fromisoformat(value).replace(tzinfo=state.BEIJING)


class ArenaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        patcher = patch.object(state, "STATE_PATH", Path(self.temp.name) / "arena.sqlite3")
        patcher.start()
        self.addCleanup(patcher.stop)
        state._sessions.clear()
        self.context = Mock()
        self.context.tasker.controller.uuid = "device-a"
        self.context.tasker.get_task_detail.return_value = SimpleNamespace(
            status=SimpleNamespace(succeeded=True))
        self.context.override_next.return_value = True
        self.action = weekly.ArenaWeekly()
        self.now = beijing("2026-09-16T10:00:00")
        clock = patch.object(state, "now_beijing", side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)

    def run_action(self, operation, mode="weekly", task_id=20):
        return self.action.run(self.context, SimpleNamespace(
            node_name=operation, task_detail=SimpleNamespace(task_id=task_id),
            custom_action_param=json.dumps({"operation": operation, "mode": mode})))

    def login(self, suffix=None, server="千羽森林"):
        state.begin_login(self.context, 10, suffix is not None)
        if suffix is not None:
            state.set_login_suffix(self.context, suffix)
            state.set_login_server(self.context, server)

    def next_nodes(self):
        return self.context.override_next.call_args.args[1]

    def test_settlement_boundaries_and_timezone(self):
        cases = [("2026-09-19T20:59:59", False), ("2026-09-19T21:00:00", True),
                 ("2026-09-20T11:59:59", True), ("2026-09-20T12:00:00", False)]
        for value, expected in cases:
            now = beijing(value)
            self.assertEqual(state.in_settlement(now), expected)
            self.assertEqual(state.in_settlement(now.astimezone(timezone.utc)), expected)
        self.assertEqual(state.week_key(beijing("2026-09-20T11:59:59")),
                         "2026-09-13T12:00:00+08:00")
        self.assertEqual(state.week_key(beijing("2026-09-20T12:00:00")),
                         "2026-09-20T12:00:00+08:00")

    def test_year_boundary(self):
        self.assertEqual(state.week_key(beijing("2027-01-01T00:00:00")),
                         "2026-12-27T12:00:00+08:00")

    def test_all_modes_stop_before_navigation_and_reset_next_later(self):
        for mode in ("never", "always", "weekly"):
            self.now = beijing("2026-09-19T21:00:00")
            self.assertTrue(self.run_action("enter", mode))
            self.assertEqual(self.next_nodes(), [])
            self.now = beijing("2026-09-20T12:00:00")
            self.assertTrue(self.run_action("enter", mode))
            self.assertIn("竞技场任务链准备", self.next_nodes())

    def test_incomplete_then_success_persists_across_action_instances(self):
        self.login()
        self.run_action("enter")
        self.run_action("choose")
        self.assertIn("我的搭配", self.next_nodes())
        self.assertFalse(state.StateStore().completed("default", state.week_key()))
        # 模拟失败/进程重启：没有执行 complete 就不能跳过。
        self.action = weekly.ArenaWeekly()
        self.run_action("enter")
        self.run_action("choose")
        self.assertIn("我的搭配", self.next_nodes())
        self.assertTrue(self.run_action("complete"))
        self.action = weekly.ArenaWeekly()
        self.run_action("enter")
        self.run_action("choose")
        self.assertNotIn("我的搭配", self.next_nodes())
        self.now = beijing("2026-09-20T12:00:00")
        self.run_action("enter")
        self.run_action("choose")
        self.assertIn("我的搭配", self.next_nodes())

    def test_never_and_always_ignore_weekly_completion(self):
        self.login()
        state.StateStore().mark_completed("default", state.week_key(), self.now)
        self.run_action("enter")
        self.run_action("choose", "always")
        self.assertIn("我的搭配", self.next_nodes())
        self.run_action("choose", "never")
        self.assertNotIn("我的搭配", self.next_nodes())
        self.assertFalse(self.run_action("complete"))

    def test_accounts_servers_and_default_are_isolated(self):
        self.login("0012")
        self.run_action("enter")
        self.run_action("choose")
        self.run_action("complete")
        for suffix, server, skip in [("0012", "千羽森林", True),
                                     ("0013", "千羽森林", False),
                                     ("0012", "抖落繁星", False),
                                     (None, "", False)]:
            self.login(suffix, server)
            self.run_action("enter")
            self.run_action("choose")
            self.assertEqual("我的搭配" not in self.next_nodes(), skip)

    def test_multi_identity_not_reused_without_start(self):
        self.login("0012")
        self.assertIsNotNone(state.take_account(self.context))
        self.assertIsNone(state.take_account(self.context))
        # 重启只能恢复多账号模式，不能恢复已过期的身份。
        state._sessions.clear()
        self.assertIsNone(state.take_account(self.context))

    def test_failed_login_unknown_server_and_first_run_have_no_identity(self):
        self.assertIsNone(state.take_account(self.context))
        self.login("0012", "")
        self.assertIsNone(state.take_account(self.context))
        self.login("0012")
        self.context.tasker.get_task_detail.return_value.status.succeeded = False
        self.assertIsNone(state.take_account(self.context))
        self.assertIsNone(state.take_account(self.context))

    def test_device_sessions_are_isolated(self):
        self.login("0012")
        self.context.tasker.controller.uuid = "device-b"
        self.assertIsNone(state.take_account(self.context))
        self.context.tasker.controller.uuid = "device-a"
        self.assertIsNotNone(state.take_account(self.context))

    def test_unknown_identity_dresses_without_writing(self):
        self.login("0012", "")
        self.run_action("enter")
        self.run_action("choose")
        self.assertIn("我的搭配", self.next_nodes())
        with patch.object(state.StateStore, "mark_completed") as save:
            self.run_action("complete")
            save.assert_not_called()

    def test_corrupt_storage_does_not_stop_challenges_or_skip_outfit(self):
        self.login()
        self.run_action("enter")
        state.STATE_PATH.write_bytes(b"corrupt database")
        self.assertTrue(self.run_action("choose"))
        self.assertIn("我的搭配", self.next_nodes())
        self.assertTrue(self.run_action("complete"))
        self.assertEqual(self.next_nodes(), ["竞技场挑战策略分流"])

    def test_settlement_and_week_crossing_never_mark_completion(self):
        for finish in ("2026-09-19T21:00:00", "2026-09-20T12:00:00"):
            self.now = beijing("2026-09-19T20:59:00")
            self.login()
            self.run_action("enter")
            self.run_action("choose")
            self.now = beijing(finish)
            with patch.object(state.StateStore, "mark_completed") as save:
                self.run_action("complete")
                save.assert_not_called()

    def test_complete_requires_matching_task_and_pending_outfit(self):
        self.assertFalse(self.run_action("complete"))
        self.login()
        self.run_action("enter")
        self.assertFalse(self.run_action("complete"))
        self.run_action("choose")
        self.assertFalse(self.run_action("complete", task_id=21))
        self.assertTrue(self.run_action("complete"))
        self.assertFalse(self.run_action("complete"))

    def test_server_requires_unique_full_name_and_respects_expected(self):
        self.login("0012", "")
        action = switch.SwitchServer()
        for texts, expected, result in [(["千 羽 森 林"], "", "千羽森林"),
                                         (["千羽森林", "抖落繁星"], "", ""),
                                         (["千羽森林"], "抖落繁星", ""),
                                         (["千羽"], "", "")]:
            action._ocr = Mock(return_value=[SimpleNamespace(text=t) for t in texts])
            action._record_current_server(self.context, None, expected)
            self.assertEqual(state._sessions["device-a"].server, result)

    def test_existing_challenge_action_stops_during_settlement(self):
        self.now = beijing("2026-09-19T21:00:00")
        action = arena.ArenaCompareAction()
        action._read_challenge_times = Mock()
        self.assertTrue(action.run(self.context, SimpleNamespace(node_name="竞技场稳守段位")))
        action._read_challenge_times.assert_not_called()
        self.assertEqual(self.next_nodes(), [])

    def test_crossing_settlement_while_reading_power_never_clicks_opponent(self):
        self.now = beijing("2026-09-19T20:59:59")
        action = arena.ArenaCompareAction()

        def read_power(*args):
            self.now = beijing("2026-09-19T21:00:00")
            return 100, None

        action._read_player_power = read_power
        action._click_opponent = Mock()
        with patch.object(arena, "read_ocr_number", return_value=10):
            self.assertIs(action._challenge_once(
                self.context, self.context.tasker.controller,
                [0, 0, 10, 10], [[0, 0, 10, 10]], [0, 0, 10, 10],
                5, "点击OK-竞技场", 3, 1, 5, 1), arena.ChallengeOutcome.STOPPED)
        action._click_opponent.assert_not_called()

    def test_start_again_clears_previous_identity_before_login(self):
        self.login("0012")
        state.begin_login(self.context, 11, True)
        self.assertIsNone(state.take_account(self.context))

    def test_pipeline_completion_follows_full_chain_and_options_keep_legacy_ids(self):
        pipeline = json.loads((ROOT / "assets/resource/pipeline/arena.json").read_text(encoding="utf-8"))
        node = "我的搭配"
        visited = []
        while node != "竞技场搭配完成确认":
            self.assertNotIn(node, visited)
            visited.append(node)
            node = next(n for n in pipeline[node]["next"] if not n.startswith("[JumpBack]"))
        for n in ("推荐搭配", "确定竞技场影召1", "确定竞技场影召2", "确定竞技场影召3", "返回竞技场战力对比页面"):
            self.assertIn(n, visited)
        self.assertEqual(pipeline[node]["recognition"]["type"], "OCR")
        interface = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
        option = interface["option"]["竞技场是否搭配服装"]
        self.assertEqual(option["default_case"], "No")
        self.assertEqual([c["name"] for c in option["cases"]], ["No", "Yes", "Weekly"])


if __name__ == "__main__":
    unittest.main()
