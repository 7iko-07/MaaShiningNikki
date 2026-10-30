import json
import unittest
from collections import Counter
from pathlib import Path


INTERFACE = Path(__file__).resolve().parents[1] / "assets" / "interface.json"


class TaskPresetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.interface = json.loads(INTERFACE.read_text(encoding="utf-8"))
        cls.presets = {preset["name"]: preset for preset in cls.interface["preset"]}

    def names(self, preset):
        return [task["name"] for task in self.presets[preset]["task"]]

    def test_two_named_presets(self):
        self.assertEqual(list(self.presets), ["全部任务", "完整日常"])

    def test_task_references_are_unique_and_exist(self):
        available = {task["name"] for task in self.interface["task"]}
        for name in self.presets:
            with self.subTest(preset=name):
                tasks = self.names(name)
                self.assertEqual(len(tasks), len(set(tasks)))
                self.assertTrue(set(tasks) <= available)
        self.assertEqual(set(self.names("全部任务")), available)

    def test_start_and_end_tasks_and_execution_metadata(self):
        definitions = {task["name"]: task for task in self.interface["task"]}
        self.assertFalse({"启动模拟器", "关闭模拟器", "关闭本程序"} & definitions.keys())
        self.assertEqual(definitions["关闭闪耀暖暖"]["__mfa_end_action"], "Pipeline")
        self.assertEqual(definitions["关闭闪耀暖暖"]["entry"], "关闭闪暖")
        startup = definitions["启动闪耀暖暖"]
        self.assertTrue(startup["__mfa_startup"])
        self.assertEqual(startup["label"], "🚀 启动前执行")
        self.assertEqual(startup["entry"], "启动账号记录初始化")
        self.assertNotIn("group", startup)
        shutdown = definitions["关闭闪耀暖暖"]
        self.assertTrue(shutdown["__mfa_shutdown"])
        self.assertEqual(shutdown["label"], "🛑 结束后执行")
        self.assertNotIn("group", shutdown)
        for name, preset in self.presets.items():
            with self.subTest(preset=name):
                self.assertNotIn("启动前执行", [group["name"] for group in preset["group"]])
                self.assertEqual(preset["task"][0],
                                 {"name": "启动闪耀暖暖", "enabled": True, "group": []})
                self.assertNotIn("__mfa_end_tasks", [group["name"] for group in preset["group"]])
                self.assertEqual(preset["task"][1],
                                 {"name": "关闭闪耀暖暖", "enabled": True, "group": []})

    def test_daily_contents(self):
        self.assertEqual(self.names("完整日常"), [
            "启动闪耀暖暖", "关闭闪耀暖暖", "好友", "联盟", "邮件", "福利",
            "感应谜之海两次", "回家", "时光钟表铺", "不落的帷幕", "印象旅航",
            "制作目标服装", "美甲", "情报屋", "竞技场", "搭配评选赛", "分享",
            "一键领取", "回忆小铺", "月卡领取", "特权卡领取", "活动：寻阅山河", "织梦空间",
        ])

    def test_declared_groups_match_tasks(self):
        for name, preset in self.presets.items():
            with self.subTest(preset=name):
                declared = [group["name"] for group in preset["group"]]
                memberships = Counter(group for task in preset["task"] for group in task["group"])
                self.assertEqual(set(declared), set(memberships))
                self.assertTrue(all(count > 1 for count in memberships.values()))
                runs = []
                for task in preset["task"]:
                    group = task["group"]
                    if group and (not runs or runs[-1] != group[0]):
                        runs.append(group[0])
                self.assertEqual(runs, declared)
        self.assertEqual(self.presets["完整日常"]["group"], [])
        self.assertTrue(all(task["group"] == [] for task in self.presets["完整日常"]["task"]))

    def test_preview_preserves_optional_task_defaults(self):
        tasks = {task["name"]: task for task in self.presets["全部任务"]["task"]}
        for name in ["活动：湘间君子女", "活动：寻阅山河", "剧情自动点击"]:
            self.assertFalse(tasks[name]["enabled"])
        self.assertTrue(all(task["enabled"] for task in self.presets["完整日常"]["task"]))


if __name__ == "__main__":
    unittest.main()
