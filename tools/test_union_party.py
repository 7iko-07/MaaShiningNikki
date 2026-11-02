"""入夜派对时段、题库串题、重复发送与输入顺序的离线验证。"""

import importlib.util
import json
import re
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from maa.agent.agent_server import AgentServer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
spec = importlib.util.spec_from_file_location("union_party_under_test", ROOT / "agent/custom/union_party.py")
party = importlib.util.module_from_spec(spec)
with patch.object(AgentServer, "register_custom_action", return_value=True):
    spec.loader.exec_module(party)


class PartyTests(unittest.TestCase):
    def setUp(self):
        self.context = Mock()
        self.context.tasker.stopping = False
        self.clock = 0.
        for patcher in (
            patch.object(party.time, "monotonic", side_effect=lambda: self.clock),
            patch.object(party.time, "sleep", side_effect=self.advance),
            patch.object(party, "logger"),
            patch.object(party, "debug_exception"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def advance(self, seconds):
        self.clock += seconds

    def test_session_boundaries_and_utc(self):
        for hour in (10, 19, 20):
            for suffix, opened in (("09:59", False), ("10:00", True), ("19:59", True), ("20:00", False)):
                now = datetime.fromisoformat(f"2026-11-01T{hour}:{suffix}+08:00")
                for value in (now, now.astimezone(timezone.utc)):
                    end = party.session_end(value)
                    self.assertEqual(end is not None, opened)
                    if opened:
                        self.assertEqual(end, now.replace(minute=20, second=0))
        self.assertIsNone(party.session_end(datetime.fromisoformat("2026-11-01T11:10:00+08:00")))

    def test_closed_session_skips_without_navigation_and_cancel_stops(self):
        with patch.object(party, "now_beijing", return_value=datetime.fromisoformat("2026-11-01T10:20:00+08:00")):
            self.assertTrue(party.UnionPartyAction().run(self.context, SimpleNamespace()))
            party.logger.info.assert_called_once_with("入夜派对尚未开放")
            self.context.run_task.assert_not_called()
            self.context.tasker.controller.post_screencap.assert_not_called()
            self.context.tasker.stopping = True
            self.assertFalse(party.UnionPartyAction().run(self.context, SimpleNamespace()))

    def test_deadline_is_session_end_plus_thirty_seconds_not_ten_minutes_from_start(self):
        now = datetime.fromisoformat("2026-11-01T19:18:15+08:00")
        with patch.object(party, "now_beijing", return_value=now), patch.object(party, "PartyRunner") as runner:
            runner.return_value.run.return_value = True
            self.assertTrue(party.UnionPartyAction().run(self.context, SimpleNamespace()))
            runner.assert_called_once_with(self.context, 135.)

    def test_keywords_match_only_their_own_question(self):
        self.assertEqual(sum(isinstance(answer, str) for _, _, answer in party.QUESTIONS), 191)
        self.assertEqual(sorted(answer for _, _, answer in party.QUESTIONS if isinstance(answer, int)), [4, 9, 9])
        for i, (text, pattern, _) in enumerate(party.QUESTIONS):
            with self.subTest(text=text):
                self.assertEqual(party.match_question(text), i)
                hits = [j for j, (other, _, _) in enumerate(party.QUESTIONS)
                        if re.search(pattern, party.normalize(other))]
                self.assertEqual(hits, [i])

    def test_multiline_punctuation_fullwidth_and_conflicting_questions(self):
        for text, answer in (("暖暖在足球\n校队中踢哪个位置？", "守门员"),
                             ("ＢＪ４ 是什么意思？", "不解释"),
                             ("哪个国家覆盖雪面积最大", "北地王国"),
                             ("哪种水果势力最差", "芒果")):
            self.assertEqual(party.QUESTIONS[party.match_question(text)][2], answer)
        self.assertIsNone(party.match_question("暖暖的星座，足球校队"))
        self.assertIsNone(party.match_question("未收录的新问题"))

    def test_all_twenty_seven_missing_questions_match_their_answers(self):
        cases = (
            ("蜜蜂停在日历上，猜一成语", "风和日丽"),
            ("最坚不可摧的门怕什么", "钥匙"),
            ("什么水果最忙", "芒果"),
            ("哪个月有28天", "每个月"),
            ("什么动物最没有方向感", "麋鹿"),
            ("一般来说失眠的人都在想什么", "睡觉"),
            ("大熊猫一生的遗憾是没有什么", "彩色照片"),
            ("13个人捉迷藏，捉了10个，还剩几个", "2"),
            ("失败是成功之母，那成功是失败的什么", "反义词"),
            ("小明的妈妈只生了三个孩子，老大叫大毛，老二叫二毛，老三叫什么", "小明"),
            ("一口井7米深，有只蜗牛从井底往上爬，白天爬3米，晚上往下坠2米，问几天能爬出", "5"),
            ("涂抹", "色彩"), ("电线杆", "多嘴"), ("盘底", "调味品"),
            ("春天", "花儿"), ("右边画彩虹", "龙"), ("慢动作", "重播"),
            ("____生南国，春来发几枝", "红豆"),
            ("氢氦锂铍硼，碳氮氧氟氖，钠镁铝硅磷，硫氯氩钾____", "钙"),
            ("汉语言文字的历史最早可以追溯到夏朝还是商朝", "商朝"),
            ("大江东去浪淘尽千古风流", "人物"), ("夜来风雨声____知多少", "花落"),
            ("哈姆雷特的作者", "莎士比亚"), ("海市蜃楼形成的原因是因为光的折射还是反射", "折射"),
            ("上北下南左西右____", "东"), ("丝织品由桑蚕丝制作而成，必须在多少摄氏度水温下单独手洗", "40"),
            ("珍珠需要与什么接触", "空气"),
        )
        self.assertEqual(len(cases), 27)
        for text, answer in cases:
            with self.subTest(text=text):
                index = party.match_question(text)
                self.assertIsNotNone(index)
                self.assertEqual(party.QUESTIONS[index][2], answer)

    def test_missing_question_log_fragments_and_reversed_ocr_lines(self):
        for text, answer in (
            ("盘底\n今日派对剩余06:18", "调味品"),
            ("第5/20题\n大江东去浪淘尽千古风流\n已答对2题", "人物"),
            ("知多少。\n夜来风雨声，\n已答对4题", "花落"),
            ("失败是成功之母，那成功是\n失败的什么?", "反义词"),
        ):
            self.assertEqual(party.QUESTIONS[party.match_question(text)][2], answer)

    def test_number_puzzles_do_not_answer_different_conditions(self):
        for text in (
            "12个人捉迷藏捉了10个还剩几个", "13个人捉迷藏捉了9个还剩几个",
            "113个人捉迷藏捉了10个还剩几个", "13个人捉迷藏捉了110个还剩几个",
            "一口井8米深蜗牛白天爬3米晚上下坠2米几天爬出",
            "一口井7米深蜗牛白天爬4米晚上下坠2米几天爬出",
            "一口井17米深蜗牛白天爬3米晚上下坠2米几天爬出",
            "一口井7米深蜗牛白天爬13米晚上下坠2米几天爬出",
            "把18分成两半是多少", "把80分成两半是多少",
            "2到10哪个不开车", "1到100哪个不开车",
            "四个小朋友猜拳一共多少手指", "13个小朋友猜拳一共多少手指",
        ):
            with self.subTest(text=text):
                self.assertIsNone(party.match_question(text))

    def test_confirmed_answers_and_excluded_unconfirmed_question(self):
        for text, answer in (
            ("遇见你我就姓____", "福"), ("你属____", "属于我"),
            ("超人保护地球____保护城市", "屏幕"),
            ("台风天要带多少钱才能出门", "40000000"),
            ("把８分成两半是多少", "0"),
            ("吃苹果时吃到几条虫子最可怕", "半条"),
            ("狐狸为什么站不起来因为它____", "狡猾"),
        ):
            with self.subTest(text=text):
                index = party.match_question(text)
                self.assertIsNotNone(index)
                self.assertEqual(party.QUESTIONS[index][2], answer)
                self.assertIsInstance(party.QUESTIONS[index][2], str)
        self.assertIsNone(party.match_question("两好三坏猜一个地名"))

    def test_reverse_and_similar_question_wordings_do_not_conflict(self):
        for text, answer in (
            ("流影镇的镇长是谁", "罗德尔"),
            ("罗德尔是什么镇的镇长", "流影"),
            ("叶苏夏是苹果联邦流影镇几年级的小学生", "2"),
            ("春风送暖入____", "屠苏"),
            ("爆竹声中一岁除春风送暖入屠苏里面的屠苏指什么", "酒"),
            ("我买几个橘子去你就在此地不要走动是朱自清写给谁的文章", "父亲"),
            ("我买几个橘子你就在此地不要动出自哪篇文章", "背影"),
            ("蜜蜂停在日历上打一成语", "风和日丽"),
            ("哪种动物的名字颠倒过来可以从动物变成食物", "蜜蜂"),
            ("什么人不怕冷", "雪人"), ("哪种动物最怕冷", "鸭子"),
        ):
            with self.subTest(text=text):
                index = party.match_question(text)
                self.assertIsNotNone(index)
                self.assertEqual(party.QUESTIONS[index][2], answer)

    def test_historical_source_variants_and_multiline_questions(self):
        for text, answer in (
            ("To be or not to be 是谁的一句名言", "莎士比亚"),
            ("生存还是毁灭这是一个问题是谁的一句名言", "莎士比亚"),
            ("秦衣又被称为____干颜", "卿"),
            ("秦衣又称作____千面", "卿"),
            ("阿海最喜欢的设计师是谁", "李尔里德"),
            ("图鉴资料库全名", "忆海资料馆"),
            ("夏梦露设计的完美原型叫", "爱衣"),
            ("暖暖的洋芋片是什么口味", "番茄"),
            ("一颗好友的爱心可以得多少体力", "2"),
            ("暖暖设计中心的地图标示在谁的上方", "薇薇安"),
            ("与暖暖亲密度达到几等时可以解锁设计工具", "20"),
            ("第一次回家时暖暖所穿的套装名称为", "欲望之音"),
            ("暖暖看的电视剧的女主角的恋人是爸爸的", "仇人的儿子"),
            ("哪些字母人最喜欢听", "CD"),
            ("如果有一辆计程车你是司机小明是乘客计程车是谁的", "如果"),
            ("什么东西以2000公里每小时的速度载人而不用加油和其他燃料", "地球"),
            ("人以铜为镜可以正衣冠以古为镜可以见____", "兴替"),
            ("飓风天要带多少钱才能出门", "40000000"),
            ("3个小朋友猜拳一个出剪刀一个出石头一个出布共有几根手指", "30"),
            ("年纪不算大胡子一大把不分大和小只会叫妈妈猜一动物", "羊"),
            ("一边是绿一边是红一边喜雨一边喜风猜一字", "秋"),
            ("哪两个英文字母合在一起会爆炸", "OK"),
            ("第4/20题\n浮梦立方是宙与海的\n已答对2题", "实验基地"),
            ("在哪里可以看到已拥有的设计师之影", "方舟"),
            ("鱼吐泡泡猜一种颜色", "蓝色"),
            ("海哲的服装店全名", "苹果联邦服装集团"),
            ("小明妈妈有三个儿子大儿子叫一明二儿子叫二明三儿子叫什么", "小明"),
        ):
            with self.subTest(text=text):
                index = party.match_question(text)
                self.assertIsNotNone(index)
                self.assertEqual(party.QUESTIONS[index][2], answer)

    def test_picture_question_matches_without_digit_one_or_any_option_digits(self):
        for text in (
            "请选出图中服饰的正确\n名称。(输入、2、3、\n4进行回答）",
            "请选出图中服饰的正确名称，输入一、2、3、4进行回答",
            "请选出图中服饰的正确名称，输入l、2、3、4进行回答",
            "请选出图中服饰的正确名称",
        ):
            with self.subTest(text=text):
                index = party.match_question(text)
                self.assertIsNotNone(index)
                self.assertEqual(party.QUESTIONS[index][2], 4)

    def test_controls_do_not_depend_on_always_visible_correct_or_missing_one(self):
        self.assertEqual(party.panel_state("已答对12题"), "question")
        self.assertEqual(party.panel_state("3 秒 后 进入 下 题，已答对12题"), "waiting")
        self.assertEqual(party.panel_state("本次答题结束了，已答对12题"), "ended")

    def loop(self, frames, deadline=630):
        runner = party.PartyRunner(self.context, deadline)
        position = 0

        def capture():
            nonlocal position
            runner.check()
            frame = frames[min(position, len(frames) - 1)]
            position += 1
            return frame

        runner.capture = capture
        runner.read_panel = lambda image: image
        runner.wait_button = Mock(return_value=(350, 1080, 120, 30))
        runner.task = Mock()
        runner.action = Mock()
        runner.send = Mock(return_value=True)
        runner.enter = Mock(return_value=True)
        return runner

    def test_wrong_same_question_blank_frames_and_score_do_not_resend(self):
        text = "暖暖是哪个星座的？"
        runner = self.loop([text, "", "已答对0题", text + "已答对0题", text, "本次答题结束了"])
        self.assertTrue(runner.run())
        runner.send.assert_called_once_with(party.match_question(text), "射手座")

    def test_unknown_question_is_logged_skipped_and_next_question_still_answered(self):
        runner = self.loop(["新题目", "新题目", "5秒后进入下题", "", "暖暖是哪个星座的？", "本次答题结束了"])
        self.assertTrue(runner.run())
        self.assertEqual(runner.send.call_count, 1)
        party.logger.debug.assert_any_call("入夜派对：未匹配题目='新题目'")

    def test_countdown_allows_the_same_question_in_a_new_round(self):
        text = "暖暖是哪个星座的？"
        runner = self.loop([text, text, "3秒后进入下一题", "1秒后进入下题", "", text, text, "本次答题结束了"])
        self.assertTrue(runner.run())
        self.assertEqual(runner.send.call_count, 2)

    def test_new_question_can_be_answered_when_countdown_was_not_captured(self):
        runner = self.loop(["暖暖是哪个星座的？", "谁是万兽之王？", "本次答题结束了"])
        self.assertTrue(runner.run())
        self.assertEqual([item.args[1] for item in runner.send.call_args_list], ["射手座", "动物园园长"])

    def test_random_question_sends_each_digit_once_and_stops_after_nine(self):
        for text in ("猜猜大喵今天吃了几块五花肉？", "猜猜大喵今天收到了几个红包？"):
            with self.subTest(text=text):
                frames = [text] * 5 + ["", "已答对0题"] + [text] * 15
                runner = self.loop(frames + ["本次答题结束了"])
                self.assertTrue(runner.run())
                self.assertEqual([item.args[1] for item in runner.send.call_args_list], list("123456789"))

    def test_random_question_stops_on_countdown_or_end_or_new_question(self):
        for text in ("猜猜大喵今天吃了几块五花肉？", "猜猜大喵今天收到了几个红包？"):
            for next_text in ("5秒后进入下题", "本次答题结束了", "暖暖是哪个星座的？"):
                with self.subTest(text=text, next_text=next_text):
                    runner = self.loop([text, text, next_text, "本次答题结束了"])
                    self.assertTrue(runner.run())
                    answers = [item.args[1] for item in runner.send.call_args_list]
                    self.assertEqual(answers[:2], ["1", "2"])
                    self.assertNotIn("3", answers)

    def test_red_packet_next_round_restarts_and_fixed_numeric_answer_is_sent_once(self):
        text = "猜猜大喵今天收到了几个红包？"
        runner = self.loop([text, text, "5秒后进入下题", text, text,
                            "台风天要带多少钱才能出门", "", "台风天要带多少钱才能出门",
                            "本次答题结束了"])
        self.assertTrue(runner.run())
        self.assertEqual([item.args[1] for item in runner.send.call_args_list],
                         ["1", "2", "1", "2", "40000000"])

    def test_picture_question_sends_one_to_four_once_despite_blank_frames(self):
        text = "请选出图中服饰的正确名称，输入、2、3、4进行回答"
        runner = self.loop([text, text, "", text, "已答对5题", text] + [text] * 10 + ["本次答题结束了"])
        self.assertTrue(runner.run())
        self.assertEqual([item.args[1] for item in runner.send.call_args_list], list("1234"))

    def test_picture_question_stops_on_countdown_or_end_or_new_question(self):
        text = "请选出图中服饰的正确名称"
        for next_text in ("5秒后进入下题", "本次答题结束了", "暖暖是哪个星座的？"):
            runner = self.loop([text, text, next_text, "本次答题结束了"])
            self.assertTrue(runner.run())
            answers = [item.args[1] for item in runner.send.call_args_list]
            self.assertEqual(answers[:2], ["1", "2"])
            self.assertNotIn("3", answers)

    def test_picture_question_next_round_restarts_from_one(self):
        text = "请选出图中服饰的正确名称"
        runner = self.loop([text, text, "5秒后进入下题", text, "本次答题结束了"])
        self.assertTrue(runner.run())
        self.assertEqual([item.args[1] for item in runner.send.call_args_list], ["1", "2", "1"])

    def test_run_reentry_does_not_keep_previous_account_progress(self):
        runner = self.loop(["暖暖是哪个星座的？", "本次答题结束了"])
        self.assertTrue(runner.run())
        runner.capture = Mock(side_effect=["暖暖是哪个星座的？", "本次答题结束了"])
        self.assertTrue(runner.run())
        self.assertEqual(runner.send.call_count, 2)

    def test_grace_period_only_observes_and_missing_end_fails(self):
        runner = self.loop(["暖暖是哪个星座的？"], deadline=30)
        with self.assertRaisesRegex(RuntimeError, "未识别到答题结束提示"):
            runner.run()
        runner.send.assert_not_called()
        self.assertLess(self.clock, 31)

    def test_send_sequence_closes_keyboard_then_sends_and_clicks_twice(self):
        runner = party.PartyRunner(self.context, 630)
        box = (1, 2, 30, 40)
        runner.wait_button = Mock(return_value=box)
        runner.task = Mock()
        runner.action = Mock()
        self.assertTrue(runner.send(0, "设计师情报屋"))
        self.assertEqual(runner.action.call_args_list, [
            call("入夜派对输入框", box), call("入夜派对点击空白"),
            call("入夜派对发送", box), call("入夜派对点击空白"), call("入夜派对点击空白"),
        ])
        args = runner.task.call_args.args
        self.assertEqual(args[1]["入夜派对输入答案"]["action"]["param"]["custom_action_param"],
                         {"input_text": "设计师情报屋"})

    def test_input_button_waits_and_never_blind_clicks(self):
        runner = party.PartyRunner(self.context, 630)
        runner.capture = Mock(return_value=object())
        runner.read_panel = Mock(return_value="暖暖是哪个星座的？")
        self.context.run_recognition.return_value = SimpleNamespace(hit=False)
        with self.assertRaisesRegex(RuntimeError, "未找到输入或发送按钮"):
            runner.wait_button("入夜派对输入框", party.match_question("星座"))
        self.context.run_action.assert_not_called()
        self.assertLess(self.clock, 31)

    def test_question_transition_while_waiting_for_button_aborts_send(self):
        runner = party.PartyRunner(self.context, 630)
        runner.capture = Mock(return_value=object())
        for text in ("3秒后进入下题", "本次答题结束了", "谁是万兽之王？"):
            runner.read_panel = Mock(return_value=text)
            self.assertFalse(runner.send(party.match_question("星座"), "射手座"))
        self.context.run_action.assert_not_called()
        self.context.run_task.assert_not_called()

    def test_multiline_panel_uses_all_results_in_reading_order(self):
        runner = party.PartyRunner(self.context, 630)
        items = [SimpleNamespace(text="校队的位置", box=(200, 80, 100, 20)),
                 SimpleNamespace(text="暖暖在足球", box=(200, 40, 100, 20))]
        self.context.run_recognition.return_value = SimpleNamespace(hit=True, all_results=items)
        self.assertEqual(runner.read_panel(object()), "暖暖在足球\n校队的位置")

    def test_real_capture_click_input_recognition_failures_and_cancellation(self):
        runner = party.PartyRunner(self.context, 630)
        self.context.tasker.controller.post_screencap.return_value.wait.return_value.succeeded = False
        with self.assertRaisesRegex(RuntimeError, "截图失败"):
            runner.capture()
        self.context.run_action.return_value = SimpleNamespace(success=False)
        with self.assertRaisesRegex(RuntimeError, "点击失败"):
            runner.action("入夜派对发送", (1, 2, 30, 40))
        self.context.run_task.return_value = SimpleNamespace(status=SimpleNamespace(succeeded=False))
        with self.assertRaisesRegex(RuntimeError, "文本输入失败"):
            runner.task("入夜派对输入答案")
        self.context.run_recognition.return_value = None
        with self.assertRaisesRegex(RuntimeError, "识别执行失败"):
            runner.recognize("入夜派对读取题目", object())
        self.context.tasker.stopping = True
        with self.assertRaisesRegex(RuntimeError, "任务已停止"):
            runner.check()

    def test_pipeline_input_reuses_dreamweaver_and_all_subtasks_are_leaves(self):
        pipeline = json.loads((ROOT / "assets/resource/pipeline/union_party.json").read_text(encoding="utf-8"))
        self.assertTrue(all(not node.get("next") for node in pipeline.values()))
        self.assertEqual(pipeline["入夜派对输入答案"]["action"]["param"]["custom_action"], "dreamweaver_input_text")
        self.assertEqual(pipeline["入夜派对读取题目"]["recognition"]["param"]["roi"], [227, 151, 493, 375])
        interface = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
        task = next(task for task in interface["task"] if task["name"] == "入夜派对")
        self.assertIn(task["entry"], pipeline)

    def test_party_page_title_accepts_missing_initial_character(self):
        pipeline = json.loads((ROOT / "assets/resource/pipeline/union_party.json").read_text(encoding="utf-8"))
        params = pipeline["导航在入夜派对页面"]["recognition"]["param"]
        self.assertEqual(params["roi"], [381, 0, 339, 162])
        for text in ("入夜派对", "夜派对"):
            self.assertIsNotNone(re.fullmatch(params["expected"][0], text))
        for text in ("派对", "联盟", "入夜"):
            self.assertIsNone(re.fullmatch(params["expected"][0], text))
        self.assertEqual(pipeline["入夜派对任务链准备"]["action"]["param"]["custom_action_param"],
                         {"target": "入夜派对说明"})

    def entry_runner(self, frames):
        runner = party.PartyRunner(self.context, 630)
        position = 0

        def capture():
            nonlocal position
            runner.check()
            frame = frames[min(position, len(frames) - 1)]
            position += 1
            return frame

        runner.capture = Mock(side_effect=capture)
        runner.recognize = Mock(side_effect=lambda node, image: SimpleNamespace(hit=node in image))
        runner.task = Mock()
        runner.wait_button = Mock(return_value=(302, 780, 120, 30))
        runner.action = Mock()
        return runner

    def test_one_frame_participated_toast_is_captured_before_any_delay(self):
        runner = self.entry_runner([set(), {"入夜派对今日已参与"}, set()])
        self.assertFalse(runner.enter())
        self.assertEqual(self.clock, 0)
        self.assertEqual(runner.capture.call_count, 2)
        runner.action.assert_called_once_with("入夜派对点击前往派对", (302, 780, 120, 30))
        party.logger.info.assert_called_once_with("入夜派对：今日已参与，跳过")

    def test_participated_is_successful_skip_without_reading_questions(self):
        runner = self.entry_runner([set(), {"入夜派对今日已参与"}])
        runner.read_panel = Mock()
        runner.send = Mock()
        self.assertTrue(runner.run())
        runner.read_panel.assert_not_called()
        runner.send.assert_not_called()

    def test_already_in_answer_page_skips_invitation_and_entry_click(self):
        runner = self.entry_runner([{"导航在入夜派对页面"}])
        self.assertTrue(runner.enter())
        runner.task.assert_not_called()
        runner.wait_button.assert_not_called()
        runner.action.assert_not_called()

    def test_entry_waits_for_answer_page_and_does_not_require_toast(self):
        runner = self.entry_runner([set(), set(), {"导航在入夜派对页面"}])
        self.assertTrue(runner.enter())
        runner.task.assert_called_once_with("入夜派对任务链准备")
        self.assertAlmostEqual(self.clock, .1)
        self.assertEqual(runner.action.call_count, 1)

    def test_missing_toast_and_answer_page_times_out_without_repeated_click(self):
        runner = self.entry_runner([set()])
        with self.assertRaisesRegex(RuntimeError, "未能进入答题页面"):
            runner.enter()
        self.assertLess(self.clock, 31)
        self.assertEqual(runner.action.call_count, 1)
        party.logger.info.assert_not_called()

    def test_participated_ocr_uses_keywords_without_punctuation(self):
        pipeline = json.loads((ROOT / "assets/resource/pipeline/union_party.json").read_text(encoding="utf-8"))
        keywords = pipeline["入夜派对今日已参与"]["recognition"]["param"]["expected"]
        for text in ("今日已参与，请明天再来吧。", "今日已参与请明天再来吧", "今日已参与", "请明天再来吧"):
            self.assertTrue(any(re.search(keyword, text) for keyword in keywords))
        self.assertFalse(any(re.search(keyword, "派对已开启，赶快前往参加吧") for keyword in keywords))
        self.assertTrue(all(not any(char in keyword for char in "，。,.！？") for keyword in keywords))
        self.assertEqual(pipeline["入夜派对点击前往派对"]["post_delay"], 0)

    def test_alliance_switch_defaults_off_and_both_entries_share_action(self):
        interface = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
        union = json.loads((ROOT / "assets/resource/pipeline/union.json").read_text(encoding="utf-8"))
        pipeline = json.loads((ROOT / "assets/resource/pipeline/union_party.json").read_text(encoding="utf-8"))
        alliance_task = next(task for task in interface["task"] if task["name"] == "联盟")
        self.assertIn("联盟入夜派对", alliance_task["option"])
        option = interface["option"]["联盟入夜派对"]
        self.assertEqual(option["default_case"], "No")
        cases = {case["name"]: case["pipeline_override"] for case in option["cases"]}
        self.assertEqual(cases["Yes"]["退出联盟福利"]["next"], ["联盟入夜派对"])
        self.assertEqual(cases["No"]["退出联盟福利"]["next"], ["返回主页面"])
        self.assertEqual(union["退出联盟福利"]["next"], ["返回主页面"])
        self.assertEqual(union["联盟入夜派对"]["action"], pipeline["入夜派对任务链入口"]["action"])
        self.assertEqual(union["联盟入夜派对"]["next"], ["返回主页面"])

    def test_standalone_task_and_preset_belong_to_tools(self):
        interface = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
        task = next(task for task in interface["task"] if task["name"] == "入夜派对")
        self.assertEqual(task["group"], ["辅助工具"])
        preset = next(item for item in interface["preset"] if item["name"] == "全部任务")
        self.assertEqual(next(item for item in preset["task"] if item["name"] == "入夜派对")["group"],
                         ["辅助工具"])


if __name__ == "__main__":
    unittest.main()
