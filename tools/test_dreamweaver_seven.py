"""七日妙想的当前卡片、重试、制作及多选衔接测试。"""

import itertools
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools.test_dreamweaver_space import ROOT, dream
from tools.validate_schema import load_jsonc


class SevenTests(unittest.TestCase):
    def runner(self):
        context = Mock()
        context.tasker.stopping = False
        context.run_recognition.return_value = SimpleNamespace(hit=False, filtered_results=[])
        return dream.DreamRunner(context)

    def result(self, *items):
        return SimpleNamespace(hit=bool(items), filtered_results=list(items))

    def item(self, text, column=0):
        return SimpleNamespace(text=text, box=(71 + 300 * column, 252, 180, 32))

    def clock(self, now):
        return (
            patch.object(dream.time, "monotonic", side_effect=lambda: now[0]),
            patch.object(dream.time, "sleep", side_effect=lambda dt: now.__setitem__(0, now[0] + dt)),
        )

    def test_titles_cover_current_four_cards_and_exclude_partial_fifth(self):
        runner = self.runner()
        runner.reco = Mock(return_value=self.result(
            self.item("制作 １ 件 上 衣", 2),
            self.item("制作1件连衣裙", 0),
            self.item("分享1次定制成衣", 3),
            self.item("制作1件下装", 4),
        ))
        self.assertEqual(runner.seven_titles(object()),
                         [(0, "制作1件连衣裙"), (2, "制作1件上衣"), (3, "分享1次定制成衣")])

    def test_button_is_matched_only_inside_current_task_column(self):
        runner = self.runner()
        runner.seven_titles = Mock(return_value=[
            (0, "完成2次灵感拾趣"), (1, "制作1件上衣"), (2, "制作1件下装"),
        ])
        runner.box = Mock(return_value=(505, 642, 101, 32))
        frame = object()
        self.assertEqual(runner.seven_candidate(frame, set()),
                         ("上衣", 1, (505, 642, 101, 32)))
        runner.box.assert_called_once_with("七日妙想前往完成", frame, [355, 631, 260, 67])

    def test_completed_unknown_quantity_and_missing_button_are_not_clicked(self):
        runner = self.runner()
        runner.seven_titles = Mock(return_value=[
            (0, "制作1件连衣裙"), (1, "制作2件上衣"), (2, "制作1件外套"), (3, "制作1件下装"),
        ])
        runner.box = Mock(return_value=None)
        self.assertIsNone(runner.seven_candidate(object(), {"连衣裙"}))
        self.assertEqual(runner.box.call_count, 1)

    def test_share_candidate_uses_current_column_and_is_not_repeated(self):
        runner = self.runner()
        runner.seven_titles = Mock(return_value=[(2, "分享1次定制成衣")])
        runner.box = Mock(return_value=(808, 645, 100, 30))
        frame = object()
        self.assertEqual(runner.seven_candidate(frame, set()), ("分享", 2, (808, 645, 100, 30)))
        runner.box.assert_called_once_with("七日妙想前往完成", frame, [655, 631, 260, 67])
        self.assertIsNone(runner.seven_candidate(frame, {"分享"}))
        runner.seven_titles.return_value = [(2, "分享2次定制成衣")]
        self.assertIsNone(runner.seven_candidate(frame, set()))

    def test_first_complete_card_ends_all_tasks_before_reading_other_cards(self):
        runner = self.runner()
        runner.seven_card_complete = Mock(return_value=True)
        runner.seven_titles = Mock()
        runner.box = Mock()
        self.assertEqual(runner.seven_candidate(object(), set()), ("全部完成", None, None))
        runner.seven_card_complete.assert_called_once()
        self.assertEqual(runner.seven_card_complete.call_args.args[1], 0)
        runner.seven_titles.assert_not_called()
        runner.box.assert_not_called()

    def test_first_complete_card_needs_page_and_stable_confirmation_not_titles(self):
        runner = self.runner()
        runner.image = Mock(return_value=object())
        runner.seven_ready = Mock(side_effect=[False, True, True])
        runner.seven_card_complete = Mock(return_value=True)
        runner.seven_titles = Mock()
        now = [0.0]
        time_patch, sleep_patch = self.clock(now)
        with time_patch, sleep_patch:
            self.assertEqual(runner.seven_current(set()), ("全部完成", None, None))
        self.assertEqual(runner.seven_card_complete.call_count, 2)
        runner.seven_titles.assert_not_called()

    def test_later_complete_card_does_not_end_all_tasks_or_repeat_mannequin(self):
        runner = self.runner()
        runner.seven_titles = Mock(return_value=[(0, "完成1次灵感拾趣"), (3, "进行1次人台换衣")])
        runner.seven_card_complete = Mock(side_effect=lambda image, column: column == 3)
        self.assertEqual(runner.seven_candidate(object(), set()), ("拾趣", 0, None))
        runner.seven_card_complete.return_value = False
        runner.seven_card_complete.side_effect = None
        self.assertEqual(runner.seven_candidate(object(), {"拾趣"}), ("换衣", 3, None))
        self.assertIsNone(runner.seven_candidate(object(), {"换衣", "拾趣"}))

    def test_mannequin_first_name_is_required_without_binding_clothes_name(self):
        runner = self.runner()
        runner.hit = Mock(return_value=True)
        runner.reco = Mock(return_value=self.result(SimpleNamespace(text="另一个衣服名称", box=(852, 316, 95, 25))))
        self.assertTrue(runner.seven_mannequin_list(object()))
        runner.reco.return_value = self.result(SimpleNamespace(text="第二件衣服", box=(852, 470, 95, 25)))
        self.assertIsNone(runner.seven_mannequin_list(object()))
        runner.hit.return_value = False
        self.assertIsNone(runner.seven_mannequin_list(object()))

    def test_mannequin_save_requires_selected_first_clothes_and_waits_loading(self):
        runner = self.runner()
        runner.button = Mock()
        runner.click = Mock()
        runner.seven_action = Mock()
        runner.wait = Mock(return_value=(1165, 652, 47, 27))
        runner.seven_change_mannequin()
        runner.seven_action.assert_called_once_with("七日人台选择首件")
        self.assertEqual([call.args[0] for call in runner.button.call_args_list],
                         ["七日人台换衣入口", "七日妙想入口"])
        save = runner.wait.call_args_list[3].args[0]
        runner.box = Mock(return_value=(1165, 652, 47, 27))
        runner.hit = Mock(side_effect=[True, False])
        self.assertIsNone(save(object()))
        runner.box.assert_not_called()
        runner.hit = Mock(side_effect=[True, True])
        self.assertEqual(save(object()), (1165, 652, 47, 27))
        loaded = runner.wait.call_args_list[4].args[0]
        exit_loaded = runner.wait.call_args_list[1].args[0]
        self.assertIs(exit_loaded, loaded)
        runner.hit = Mock(return_value=True)
        runner.space_ready = Mock(return_value=True)
        self.assertFalse(loaded(object()))
        runner.space_ready.assert_not_called()
        runner.hit.return_value = False
        self.assertTrue(loaded(object()))

    def test_exit_loading_timeout_does_not_enter_mannequin(self):
        runner = self.runner()
        runner.image = Mock(return_value=object())
        runner.click = Mock()
        runner.button = Mock()
        runner.seven_action = Mock()
        runner.hit = Mock(return_value=True)
        runner.space_ready = Mock(return_value=True)
        original_wait = runner.wait
        def wait(predicate, description, **kwargs):
            if description == "退出七日妙想的返回箭头":
                return (29, 31, 55, 59)
            return original_wait(predicate, description, **kwargs)
        runner.wait = Mock(side_effect=wait)
        now = [0.0]
        time_patch, sleep_patch = self.clock(now)
        with time_patch, sleep_patch, self.assertRaises(dream.DreamError):
            runner.seven_change_mannequin()
        runner.click.assert_called_once_with((29, 31, 55, 59))
        runner.button.assert_not_called()
        runner.seven_action.assert_not_called()
        runner.space_ready.assert_not_called()
        self.assertLess(now[0], 30.3)

    def test_unselected_mannequin_times_out_without_saving_or_returning(self):
        runner = self.runner()
        runner.image = Mock(return_value=object())
        runner.click = Mock()
        runner.button = Mock()
        runner.seven_action = Mock()
        runner.hit = Mock(side_effect=lambda node, image: node == "七日人台展示换新")
        runner.box = Mock()
        original_wait = runner.wait
        def wait(predicate, description, **kwargs):
            if description == "首件衣服已选中及保存按钮":
                return original_wait(predicate, description, **kwargs)
            return (29, 31, 55, 59)
        runner.wait = Mock(side_effect=wait)
        now = [0.0]
        time_patch, sleep_patch = self.clock(now)
        with time_patch, sleep_patch, self.assertRaises(dream.DreamError):
            runner.seven_change_mannequin()
        runner.click.assert_called_once()
        runner.box.assert_not_called()
        runner.button.assert_called_once_with("七日人台换衣入口")
        self.assertLess(now[0], 30.3)

    def test_mannequin_dispatch_refreshes_cards_and_all_done_does_not_click(self):
        runner = self.runner()
        runner.enter_seven = Mock()
        runner.seven_current = Mock(side_effect=[("换衣", 3, None), ("全部完成", None, None)])
        runner.seven_change_mannequin = Mock()
        runner.seven_open_task = Mock()
        runner.seven_make = Mock()
        runner.seven_share = Mock()
        runner.seven()
        runner.seven_change_mannequin.assert_called_once_with()
        runner.seven_open_task.assert_not_called()
        runner.seven_make.assert_not_called()
        runner.seven_share.assert_not_called()
        self.assertEqual(runner.seven_current.call_count, 2)

    def test_card_animation_must_settle_before_deciding_no_task(self):
        runner = self.runner()
        runner.image = Mock(return_value=object())
        runner.seven_ready = Mock(return_value=True)
        runner.seven_titles = Mock(side_effect=[[], [(0, "分享1次定制成衣")],
                                               [(0, "分享1次定制成衣")], [(0, "分享1次定制成衣")]])
        runner.seven_candidate = Mock(return_value=None)
        now = [0.0]
        time_patch, sleep_patch = self.clock(now)
        with time_patch, sleep_patch:
            self.assertIsNone(runner.seven_current(set()))
        self.assertEqual(runner.image.call_count, 4)
        runner.context.run_action_direct.assert_not_called()

    def test_unreadable_card_list_times_out_instead_of_reporting_no_tasks(self):
        runner = self.runner()
        runner.image = Mock(return_value=object())
        runner.seven_ready = Mock(return_value=True)
        runner.seven_titles = Mock(return_value=[])
        now = [0.0]
        time_patch, sleep_patch = self.clock(now)
        with time_patch, sleep_patch, self.assertRaises(dream.DreamError):
            runner.seven_current(set())
        runner.context.run_action_direct.assert_not_called()
        self.assertLess(now[0], 30.3)

    def test_open_task_retries_only_while_same_task_is_visible(self):
        runner = self.runner()
        runner.click = Mock()
        runner.image = Mock(side_effect=["same", "transition", "same", "same", "clothes"])
        runner.hit = Mock(side_effect=lambda node, image: image == "clothes")
        runner.seven_ready = Mock(side_effect=lambda image: image == "same")
        runner.seven_titles = Mock(return_value=[(1, "制作1件上衣")])
        runner.box = Mock(return_value=(505, 642, 101, 32))
        now = [0.0]
        time_patch, sleep_patch = self.clock(now)
        with time_patch, sleep_patch:
            runner.seven_open_task("上衣", 1, (500, 640, 100, 30))
        self.assertEqual([call.args[0] for call in runner.click.call_args_list],
                         [(500, 640, 100, 30), (505, 642, 101, 32)])

    def test_changed_card_prevents_retry_and_has_finite_timeout(self):
        runner = self.runner()
        runner.click = Mock()
        runner.image = Mock(return_value=object())
        runner.hit = Mock(return_value=False)
        runner.seven_ready = Mock(return_value=True)
        runner.seven_titles = Mock(return_value=[(0, "分享1次定制成衣")])
        runner.box = Mock()
        now = [0.0]
        time_patch, sleep_patch = self.clock(now)
        with time_patch, sleep_patch, self.assertRaises(dream.DreamError):
            runner.seven_open_task("连衣裙", 0, (205, 642, 102, 34))
        runner.click.assert_called_once()
        runner.box.assert_not_called()

    def test_share_open_retries_same_share_card_before_entering_gallery(self):
        runner = self.runner()
        runner.click = Mock()
        runner.image = Mock(side_effect=["same", "same", "gallery"])
        runner.hit = Mock(side_effect=lambda node, image: image == "gallery")
        runner.seven_ready = Mock(return_value=True)
        runner.seven_titles = Mock(return_value=[(1, "分享1次定制成衣")])
        runner.box = Mock(return_value=(505, 642, 101, 32))
        now = [0.0]
        time_patch, sleep_patch = self.clock(now)
        with time_patch, sleep_patch:
            runner.seven_open_task("分享", 1, (500, 640, 100, 30))
        self.assertEqual(runner.click.call_count, 2)

    def test_first_clothes_name_is_required_and_is_not_bound_to_specific_text(self):
        runner = self.runner()
        runner.hit = Mock(return_value=True)
        runner.reco = Mock(return_value=self.result(
            SimpleNamespace(text="切换部位", box=(1147, 33, 108, 27)),
            SimpleNamespace(text="每周不同的衣服", box=(1145, 202, 105, 22)),
        ))
        self.assertTrue(runner.seven_clothes_list(object()))
        runner.reco.return_value = self.result(SimpleNamespace(text="第二件", box=(1145, 357, 105, 22)))
        self.assertIsNone(runner.seven_clothes_list(object()))
        runner.hit.return_value = False
        self.assertIsNone(runner.seven_clothes_list(object()))

    def test_material_requires_both_tab_and_generate_button(self):
        runner = self.runner()
        for hits, expected in (([False], False), ([True, False], False), ([True, True], True)):
            runner.hit = Mock(side_effect=hits)
            self.assertEqual(runner.seven_material_ready(object()), expected)

    def test_custom_retries_ignored_click_with_fresh_box_and_half_second_interval(self):
        runner = self.runner()
        now = [0.0]
        clicks = []
        runner.image = Mock(side_effect=lambda: runner.image.call_count)
        runner.hit = Mock(side_effect=lambda node, image: node != "七日制衣作品页" and len(clicks) == 2)
        runner.box = Mock(side_effect=lambda node, image: (image, 350, 97, 24))
        runner.click = Mock(side_effect=lambda box: clicks.append((now[0], box)))
        with self.clock(now)[0], self.clock(now)[1]:
            runner.seven_choose_custom("上衣")
        self.assertEqual(len(clicks), 2)
        self.assertGreaterEqual(clicks[1][0] - clicks[0][0], 0.5)
        self.assertNotEqual(clicks[0][1], clicks[1][1])
        self.assertIn("七日制衣上衣", [call.args[0] for call in runner.hit.call_args_list])

    def test_custom_retries_beyond_four_clicks_with_one_deadline(self):
        runner = self.runner()
        now = [0.0]
        clicks = []
        runner.image = Mock(return_value=object())
        runner.hit = Mock(return_value=False)
        runner.box = Mock(return_value=(1085, 350, 97, 24))
        runner.click = Mock(side_effect=lambda box: clicks.append(now[0]))
        with self.clock(now)[0], self.clock(now)[1], self.assertRaisesRegex(dream.DreamError, "已点击"):
            runner.seven_choose_custom("连衣裙")
        self.assertGreater(len(clicks), 4)
        self.assertTrue(all(b - a >= 0.5 - 1e-9 for a, b in zip(clicks, clicks[1:])))
        self.assertLess(now[0], 30.2)

    def test_custom_missing_button_never_clicks_and_times_out(self):
        runner = self.runner()
        now = [0.0]
        runner.image = Mock(return_value=object())
        runner.hit = Mock(return_value=False)
        runner.box = Mock(return_value=None)
        runner.click = Mock()
        with self.clock(now)[0], self.clock(now)[1], self.assertRaisesRegex(dream.DreamError, "已点击0次"):
            runner.seven_choose_custom("下装")
        runner.click.assert_not_called()

    def test_custom_destination_after_dialog_has_priority_over_old_button(self):
        runner = self.runner()
        runner.image = Mock(side_effect=["dialog", "destination", "destination"])
        runner.hit = Mock(side_effect=lambda node, image: node != "七日制衣作品页" and image == "destination")
        runner.box = Mock(return_value=(1085, 350, 97, 24))
        runner.click = Mock()
        runner.seven_choose_custom("下装")
        runner.box.assert_called_once()
        runner.click.assert_called_once()
        self.assertEqual(runner.image.call_count, 3)

    def test_old_gallery_category_before_delayed_dialog_cannot_skip_custom(self):
        runner = self.runner()
        now = [0.0]
        runner.image = Mock(side_effect=["gallery", "gallery", "dialog", "gallery", "destination", "destination"])
        runner.hit = Mock(side_effect=lambda node, image:
                          image == "gallery" if node == "七日制衣作品页" else image in ("gallery", "destination"))
        runner.box = Mock(side_effect=lambda node, image: (1085, 350, 97, 24) if image == "dialog" else None)
        runner.click = Mock()
        with self.clock(now)[0], self.clock(now)[1]:
            runner.seven_choose_custom("连衣裙")
        runner.click.assert_called_once_with((1085, 350, 97, 24))
        self.assertEqual(runner.image.call_count, 6)

    def test_destination_requires_consecutive_frames_and_no_gallery(self):
        runner = self.runner()
        now = [0.0]
        runner.image = Mock(side_effect=["dialog", "destination", "transition", "destination", "destination"])
        runner.hit = Mock(side_effect=lambda node, image: node != "七日制衣作品页" and image == "destination")
        runner.box = Mock(side_effect=lambda node, image: (1085, 350, 97, 24) if image == "dialog" else None)
        runner.click = Mock()
        with self.clock(now)[0], self.clock(now)[1]:
            runner.seven_choose_custom("连衣裙")
        runner.click.assert_called_once()
        self.assertEqual(runner.image.call_count, 5)

    def test_custom_disappearing_button_during_transition_prevents_retry(self):
        runner = self.runner()
        now = [0.0]
        runner.image = Mock(side_effect=lambda: "dialog" if now[0] < 0.1 else "transition")
        runner.hit = Mock(side_effect=lambda node, image: node != "七日制衣作品页" and now[0] >= 1)
        runner.box = Mock(side_effect=lambda node, image: (1085, 350, 97, 24) if image == "dialog" else None)
        runner.click = Mock()
        with self.clock(now)[0], self.clock(now)[1]:
            runner.seven_choose_custom("上衣")
        runner.click.assert_called_once_with((1085, 350, 97, 24))

    def test_custom_slow_recognition_cannot_click_after_deadline(self):
        runner = self.runner()
        now = [0.0]
        runner.image = Mock(return_value=object())
        runner.hit = Mock(return_value=False)
        def slow_box(node, image):
            now[0] += 31
            return (1085, 350, 97, 24)
        runner.box = Mock(side_effect=slow_box)
        runner.click = Mock()
        with self.clock(now)[0], self.clock(now)[1], self.assertRaisesRegex(dream.DreamError, "已点击0次"):
            runner.seven_choose_custom("上衣")
        runner.click.assert_not_called()

    def test_custom_click_failure_stops_before_category_selection(self):
        runner = self.runner()
        runner.image = Mock(return_value=object())
        runner.hit = Mock(return_value=False)
        runner.box = Mock(return_value=(1085, 350, 97, 24))
        runner.click = Mock(side_effect=dream.DreamError("点击失败或任务取消"))
        runner.button = Mock()
        with self.assertRaises(dream.DreamError):
            runner.seven_make("上衣")
        runner.click.assert_called_once()
        runner.button.assert_called_once_with("七日制衣新建")

    def test_creation_order_and_loading_and_return_contract(self):
        runner = self.runner()
        trace = Mock()
        for method in ("button", "seven_action", "click", "wait", "seven_choose_custom", "seven_rename"):
            setattr(runner, method, Mock(return_value=(955, 645, 90, 28)))
            trace.attach_mock(getattr(runner, method), method)
        runner.seven_make("下装")
        self.assertEqual([call.args[0] for call in runner.button.call_args_list],
                         ["七日制衣新建", "七日制衣下装",
                          "七日制衣生成成衣"])
        self.assertEqual([call.args[0] for call in runner.seven_action.call_args_list],
                         ["七日制衣选择首件", "七日制衣选择首块面料"])
        self.assertEqual([call[0] for call in trace.method_calls],
                         ["button", "seven_choose_custom", "button", "wait", "seven_action", "wait",
                          "seven_action", "button", "seven_rename", "wait", "click", "wait"])
        finished = runner.wait.call_args_list[2].args[0]
        runner.hit = Mock(side_effect=lambda node, image: image == "loading")
        runner.box = Mock(return_value=(955, 645, 90, 28))
        self.assertIsNone(finished("loading"))
        runner.box.assert_not_called()
        self.assertEqual(finished("result"), (955, 645, 90, 28))
        self.assertEqual(runner.wait.call_args_list[-1].args[0], runner.seven_ready)

    def test_rename_closes_input_before_verifying_name_and_confirming(self):
        runner = self.runner()
        trace = Mock()
        confirm_box = (704, 429, 54, 31)
        for method in ("button", "seven_input_name", "click", "wait"):
            setattr(runner, method, Mock(return_value=confirm_box))
            trace.attach_mock(getattr(runner, method), method)
        runner.seven_rename()
        self.assertEqual([call[0] for call in trace.method_calls],
                         ["wait", "button", "seven_input_name", "wait", "click", "wait", "button"])
        runner.button.assert_any_call("七日制衣点击名称框")
        runner.seven_input_name.assert_called_once_with()
        runner.click.assert_called_once_with(confirm_box)
        self.assertEqual(runner.button.call_args_list[-1].args, ("七日制衣名称确定",))
        ready = runner.wait.call_args_list[-1].args[0]
        for hits, expected in ((set(), False), ({"七日制衣名称弹窗"}, False),
                               ({"七日制衣名称已输入"}, False),
                               ({"七日制衣名称弹窗", "七日制衣名称已输入"}, True)):
            runner.hit = Mock(side_effect=lambda node, image: node in hits)
            self.assertEqual(ready(object()), expected)

    def test_rename_does_not_submit_if_input_or_closing_or_verification_fails(self):
        for failing_method, wait_failure in (("seven_input_name", 0), ("click", 0), ("wait", 1), ("wait", 2)):
            runner = self.runner()
            runner.button = Mock()
            runner.seven_input_name = Mock()
            runner.click = Mock()
            runner.wait = Mock(return_value=(704, 429, 54, 31))
            failure = dream.DreamError("输入或收起输入框失败")
            if wait_failure:
                runner.wait.side_effect = [(704, 429, 54, 31)] * wait_failure + [failure]
            else:
                getattr(runner, failing_method).side_effect = failure
            with self.assertRaises(dream.DreamError):
                runner.seven_rename()
            runner.button.assert_called_once_with("七日制衣点击名称框")
            if wait_failure == 1:
                runner.click.assert_not_called()

    def test_input_failure_or_cancellation_does_not_press_confirm(self):
        for stopping, result in ((False, None), (False, SimpleNamespace(status=SimpleNamespace(succeeded=False))), (True, None)):
            runner = self.runner()
            runner.context.tasker.stopping = stopping
            runner.context.run_task.return_value = result
            with self.assertRaises(dream.DreamError):
                runner.seven_input_name()
            if stopping:
                runner.context.run_task.assert_not_called()

    def test_input_runs_leaf_task_with_recognition_before_custom_action(self):
        runner = self.runner()
        runner.context.run_task.return_value = SimpleNamespace(status=SimpleNamespace(succeeded=True))
        runner.seven_input_name()
        runner.context.run_task.assert_called_once_with("七日制衣输入临时")
        runner.context.run_action.assert_not_called()
        node = load_jsonc(ROOT / "assets/resource/pipeline/dreamweaver_space.json")["七日制衣输入临时"]
        self.assertFalse(node.get("next"))
        self.assertFalse(node.get("on_error"))

    def input_context(self, *, index=3):
        context = self.runner().context
        context.tasker.controller.info = {
            "type": "adb",
            "config": {"extras": {"mumu": {"enable": True, "path": "D:/MuMuPlayer", "index": index}}},
        }
        argv = SimpleNamespace(custom_action_param=json.dumps({"input_text": "临时"}))
        return context, argv

    def test_mumu_input_uses_current_instance_and_checks_command_response(self):
        context, argv = self.input_context()
        with patch.object(dream.Path, "is_file", return_value=True), \
             patch.object(dream.subprocess, "run", return_value=SimpleNamespace(
                 returncode=0, stdout='{"errcode":0,"errmsg":""}',
             )) as run:
            self.assertTrue(dream.DreamweaverInputTextAction().run(context, argv))
        command = run.call_args.args[0]
        self.assertEqual(dream.Path(command[0]), dream.Path("D:/MuMuPlayer/nx_main/MuMuManager.exe"))
        self.assertEqual(command[1:], ["control", "-v", "3", "tool", "cmd", "--cmd", "input_text", "--text", "临时"])
        self.assertEqual(run.call_args.kwargs["timeout"], 10)
        context.run_action_direct.assert_not_called()

    def test_mumu_input_failure_never_falls_back_to_broken_clipboard(self):
        for result in (
            SimpleNamespace(returncode=1, stdout='{"errcode":0}'),
            SimpleNamespace(returncode=0, stdout='{"errcode":-1,"errmsg":"failed"}'),
            SimpleNamespace(returncode=0, stdout='{}'),
            SimpleNamespace(returncode=0, stdout='invalid json'),
            dream.subprocess.TimeoutExpired("MuMuManager.exe", 10),
        ):
            with self.subTest(result=result):
                context, argv = self.input_context()
                with patch.object(dream.Path, "is_file", return_value=True), \
                     patch.object(dream.subprocess, "run", side_effect=[result]):
                    self.assertFalse(dream.DreamweaverInputTextAction().run(context, argv))
                context.run_action_direct.assert_not_called()

    def test_mumu_input_requires_target_configuration_and_honors_cancellation(self):
        for index, exists, stopping in ((None, True, False), (-1, True, False),
                                        (3, False, False), (3, True, True)):
            context, argv = self.input_context(index=index)
            context.tasker.stopping = stopping
            with patch.object(dream.Path, "is_file", return_value=exists), \
                 patch.object(dream.subprocess, "run") as run:
                self.assertFalse(dream.DreamweaverInputTextAction().run(context, argv))
            run.assert_not_called()
            context.run_action_direct.assert_not_called()

    def test_non_mumu_keeps_framework_input_and_checks_actual_success(self):
        for result, expected in ((None, False), (SimpleNamespace(success=False), False),
                                 (SimpleNamespace(success=True), True)):
            context, argv = self.input_context()
            context.tasker.controller.info = {"type": "adb", "config": {}}
            context.run_action_direct.return_value = result
            with patch.object(dream.subprocess, "run") as run:
                self.assertEqual(dream.DreamweaverInputTextAction().run(context, argv), expected)
            run.assert_not_called()
            self.assertEqual(context.run_action_direct.call_args.args[0], "InputText")
            self.assertEqual(context.run_action_direct.call_args.args[1].input_text, "临时")

    def test_each_return_refreshes_card_positions_and_none_ends_without_swipe(self):
        runner = self.runner()
        runner.enter_seven = Mock()
        runner.seven_current = Mock(side_effect=[
            ("连衣裙", 0, (205, 642, 102, 34)),
            ("上衣", 0, (205, 642, 102, 34)),
            ("下装", 0, (205, 642, 102, 34)), None,
        ])
        observations = []
        results = iter(runner.seven_current.side_effect)
        runner.seven_current.side_effect = lambda completed: observations.append(set(completed)) or next(results)
        runner.seven_open_task = Mock()
        runner.seven_make = Mock()
        runner.seven()
        self.assertEqual(observations, [set(), {"连衣裙"}, {"连衣裙", "上衣"}, {"连衣裙", "上衣", "下装"}])
        self.assertEqual([call.args[0] for call in runner.seven_make.call_args_list], ["连衣裙", "上衣", "下装"])
        runner.context.run_action_direct.assert_not_called()

    def test_creation_failure_stops_before_next_task(self):
        runner = self.runner()
        runner.enter_seven = Mock()
        runner.seven_current = Mock(return_value=("连衣裙", 0, (205, 642, 102, 34)))
        runner.seven_open_task = Mock()
        runner.seven_make = Mock(side_effect=dream.DreamError("未生成成衣"))
        with self.assertRaises(dream.DreamError):
            runner.seven()
        runner.seven_current.assert_called_once()

    def test_work_count_distinguishes_empty_from_unready_and_rejects_bad_counts(self):
        runner = self.runner()
        runner.hit = Mock(return_value=True)
        for text, expected in (("0/5", (0,)), ("1/5", (1,)), ("6/5", None), ("1/0", None), ("临时", None)):
            runner.reco = Mock(return_value=self.result(SimpleNamespace(text=text)))
            self.assertEqual(runner.seven_works(object()), expected)
        runner.hit.return_value = False
        runner.reco.reset_mock()
        self.assertIsNone(runner.seven_works(object()))
        runner.reco.assert_not_called()

    def test_share_empty_work_does_not_click_a_work_or_share(self):
        runner = self.runner()
        runner.button = Mock()
        runner.wait = Mock(side_effect=[True, (0,)])
        runner.seven_action = Mock()
        runner.click = Mock()
        with self.assertRaises(dream.DreamError):
            runner.seven_share()
        runner.seven_action.assert_called_once_with("七日制衣选择首件")
        runner.click.assert_not_called()

    def test_qq_requires_both_labels_and_game_return_requires_both_buttons(self):
        runner = self.runner()
        for predicate in (runner.seven_qq_ready, runner.seven_share_page):
            for hits, expected in (([False], False), ([True, False], False), ([True, True], True)):
                runner.hit = Mock(side_effect=hits)
                self.assertEqual(predicate(object()), expected)

    def test_share_icons_are_guarded_by_selected_work_and_result_page(self):
        runner = self.runner()
        runner.button = Mock()
        runner.seven_action = Mock()
        runner.click = Mock()
        runner.wait = Mock(side_effect=[True, (1,), (980, 643, 49, 48), (850, 637, 45, 49),
                                        True, True, True, True])
        runner.seven_share()
        selected = runner.wait.call_args_list[2].args[0]
        result_share = runner.wait.call_args_list[3].args[0]
        runner.box = Mock(return_value=(850, 637, 45, 49))
        runner.hit = Mock(side_effect=[True, False])
        self.assertIsNone(selected(object()))
        runner.box.assert_not_called()
        runner.hit = Mock(side_effect=[True, True])
        self.assertEqual(selected(object()), (850, 637, 45, 49))
        runner.box.reset_mock()
        runner.hit = Mock(side_effect=[True])
        self.assertIsNone(result_share(object()))
        runner.hit = Mock(side_effect=[False, False])
        self.assertIsNone(result_share(object()))
        runner.box.assert_not_called()
        runner.hit = Mock(side_effect=[False, True])
        self.assertEqual(result_share(object()), (850, 637, 45, 49))

    def test_unrecognized_qq_times_out_without_closing_app_or_clicking_return(self):
        runner = self.runner()
        runner.button = Mock()
        runner.seven_action = Mock()
        runner.click = Mock()
        runner.image = Mock(return_value=object())
        runner.seven_qq_ready = Mock(return_value=False)
        original_wait = runner.wait
        def wait(predicate, label, **kwargs):
            if predicate == runner.seven_qq_ready:
                return original_wait(predicate, label, **kwargs)
            return (1,) if predicate == runner.seven_works else (850, 637, 45, 49)
        runner.wait = Mock(side_effect=wait)
        now = [0.0]
        time_patch, sleep_patch = self.clock(now)
        with time_patch, sleep_patch, self.assertRaises(dream.DreamError):
            runner.seven_share()
        self.assertLess(now[0], 30.3)
        self.assertNotIn("七日分享关闭QQ", [call.args[0] for call in runner.seven_action.call_args_list])
        self.assertNotIn("七日分享返回按钮", [call.args[0] for call in runner.button.call_args_list])

    def test_share_return_failure_stops_before_back_and_leave(self):
        runner = self.runner()
        runner.button = Mock()
        runner.seven_action = Mock()
        runner.click = Mock()
        runner.wait = Mock(side_effect=[True, (1,), (980, 643, 49, 48), (850, 637, 45, 49),
                                        True, True, dream.DreamError("QQ未返回游戏")])
        with self.assertRaises(dream.DreamError):
            runner.seven_share()
        self.assertEqual(runner.seven_action.call_args_list[-1].args, ("七日分享关闭QQ",))
        self.assertNotIn("七日分享返回按钮", [call.args[0] for call in runner.button.call_args_list])

    def test_share_success_returns_before_recording_and_refreshes_cards(self):
        runner = self.runner()
        runner.enter_seven = Mock()
        runner.seven_current = Mock(side_effect=[("分享", 0, (205, 642, 102, 34)),
                                                ("下装", 0, (205, 642, 102, 34)), None])
        runner.seven_open_task = Mock()
        runner.seven_share = Mock()
        runner.seven_make = Mock()
        runner.seven()
        runner.seven_share.assert_called_once_with()
        runner.seven_make.assert_called_once_with("下装")
        self.assertEqual(runner.seven_current.call_count, 3)
        runner.context.run_action_direct.assert_not_called()

    def test_navigation_returns_from_each_management_endpoint(self):
        for source, expected_buttons in (
            ("空间", ["七日妙想入口"]),
            ("经营", ["织梦经营返回按钮", "七日妙想入口"]),
            ("订单", ["织梦订单返回按钮", "织梦经营返回按钮", "七日妙想入口"]),
            ("橱窗", ["织梦橱窗返回按钮", "织梦经营返回按钮", "七日妙想入口"]),
        ):
            with self.subTest(source=source):
                runner = self.runner()
                page = [source]
                runner.seven_ready = Mock(side_effect=lambda image: page[0] == "七日")
                runner.space_ready = Mock(side_effect=lambda image: page[0] == "空间")
                runner.shop_frame = Mock(side_effect=lambda image: image if page[0] == "橱窗" else None)
                runner.map_ready = Mock(side_effect=lambda image: (1, 5) if page[0] == "订单" else None)
                runner.management_ready = Mock(side_effect=lambda image: page[0] == "经营")
                def click(node, **kwargs):
                    page[0] = "七日" if node == "七日妙想入口" else ("空间" if node == "织梦经营返回按钮" else "经营")
                runner.button = Mock(side_effect=click)
                runner.wait = Mock(side_effect=lambda predicate, label, **kwargs: predicate(object()))
                runner.enter_seven()
                self.assertEqual([call.args[0] for call in runner.button.call_args_list], expected_buttons)
                deadlines = {call.kwargs["deadline"] for call in runner.wait.call_args_list}
                self.assertEqual(len(deadlines), 1)

    def test_all_selection_combinations_keep_seven_first_and_independent(self):
        names = ("织梦选择七日妙想", "织梦选择今日签到", "织梦选择每日能量晶矿", "织梦选择订单驿站", "织梦选择售卖橱窗")
        methods = ("seven", "sign_in", "mine", "orders", "shop")
        for selected in itertools.product((False, True), repeat=5):
            flags = {name for name, yes in zip(names, selected) if yes} | {"织梦选择经营"}
            context = Mock()
            context.get_node_data.side_effect = lambda node: {"enabled": node in flags}
            with patch.object(dream, "DreamRunner") as factory:
                self.assertTrue(dream.DreamweaverSpaceAction().run(context, SimpleNamespace()))
                expected = [method for method, yes in zip(methods, selected) if yes]
                if selected[0] and any(selected[1:]):
                    expected.insert(1, "return_from_seven")
                self.assertEqual([call[0] for call in factory.return_value.method_calls],
                                 expected)
        context.get_node_data.side_effect = lambda node: {"enabled": node == "织梦选择七日妙想"}
        with patch.object(dream, "DreamRunner") as factory:
            self.assertTrue(dream.DreamweaverSpaceAction().run(context, SimpleNamespace()))
            factory.return_value.seven.assert_called_once_with()
            factory.return_value.return_from_seven.assert_not_called()

    def test_main_checkbox_display_order_and_management_nested_selection(self):
        interface = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
        pipeline = load_jsonc(ROOT / "assets/resource/pipeline/dreamweaver_space.json")
        option = interface["option"]["织梦空间子任务"]
        self.assertEqual(option["type"], "checkbox")
        self.assertEqual([case["name"] for case in option["cases"]], ["七日妙想", "今日签到", "织梦经营"])
        self.assertEqual(option["cases"][2]["option"], ["织梦经营子任务"])
        self.assertEqual([case["name"] for case in interface["option"]["织梦经营子任务"]["cases"]],
                         ["每日能量晶矿", "订单驿站", "售卖橱窗"])
        for case in option["cases"]:
            node, = case["pipeline_override"]
            self.assertFalse(pipeline[node]["enabled"])
            self.assertTrue(case["pipeline_override"][node]["enabled"])

    def test_seven_to_space_waits_for_loading_before_following_tasks(self):
        runner = self.runner()
        now = [0.0]
        page = ["seven"]
        runner.image = Mock(side_effect=lambda: page[0])
        runner.hit = Mock(side_effect=lambda node, image: image == "loading")
        runner.space_ready = Mock(side_effect=lambda image: image == "space")
        runner.seven_ready = Mock(side_effect=lambda image: image == "seven")
        runner.box = Mock(return_value=(31, 34, 52, 53))
        runner.click = Mock(side_effect=lambda box: page.__setitem__(0, "loading"))
        def sleep(dt):
            now[0] += dt
            if now[0] >= 0.6:
                page[0] = "space"
        with patch.object(dream.time, "monotonic", side_effect=lambda: now[0]), patch.object(dream.time, "sleep", side_effect=sleep):
            runner.return_from_seven()
        runner.click.assert_called_once_with((31, 34, 52, 53))
        self.assertGreaterEqual(now[0], 0.8)
        self.assertTrue(all(call.args[0] != "loading" for call in runner.space_ready.call_args_list))

    def test_seven_return_already_in_space_does_not_click_arrow(self):
        runner = self.runner()
        runner.image = Mock(return_value=object())
        runner.hit = Mock(return_value=False)
        runner.space_ready = Mock(return_value=True)
        runner.click = Mock()
        runner.box = Mock()
        now = [0.0]
        with self.clock(now)[0], self.clock(now)[1]:
            runner.return_from_seven()
        runner.click.assert_not_called()
        runner.box.assert_not_called()

    def test_missing_seven_return_arrow_times_out_without_clicking(self):
        runner = self.runner()
        runner.image = Mock(return_value=object())
        runner.hit = Mock(return_value=False)
        runner.space_ready = Mock(return_value=False)
        runner.seven_ready = Mock(return_value=True)
        runner.box = Mock(return_value=None)
        runner.click = Mock()
        now = [0.0]
        with self.clock(now)[0], self.clock(now)[1], self.assertRaises(dream.DreamError):
            runner.return_from_seven()
        runner.click.assert_not_called()
        self.assertLess(now[0], 30.3)

    def test_failed_seven_return_stops_before_sign_and_management(self):
        context = Mock()
        context.get_node_data.return_value = {"enabled": True}
        with patch.object(dream, "DreamRunner") as factory:
            factory.return_value.return_from_seven.side_effect = dream.DreamError("未返回织梦空间")
            self.assertFalse(dream.DreamweaverSpaceAction().run(context, SimpleNamespace()))
            self.assertEqual([call[0] for call in factory.return_value.method_calls], ["seven", "return_from_seven"])

    def test_pipeline_category_or_uses_full_column_and_no_explicit_defaults(self):
        pipeline = load_jsonc(ROOT / "assets/resource/pipeline/dreamweaver_space.json")
        for category in ("连衣裙", "上衣", "下装"):
            recognition = pipeline["七日制衣" + category]["recognition"]
            self.assertEqual(recognition["type"], "Or")
            children = recognition["param"]["any_of"]
            self.assertEqual([child["recognition"]["type"] for child in children], ["OCR", "TemplateMatch"])
            for child in children:
                param = child["recognition"]["param"]
                self.assertEqual(param["roi"], [1110, 0, 170, 720])
                self.assertNotIn("threshold", param)
                self.assertNotIn("index", param)
            self.assertTrue((ROOT / "assets/resource/image" / ("七日制衣" + category + ".png")).is_file())
        for node, data in pipeline.items():
            if node.startswith("七日"):
                self.assertNotIn("post_delay", data)
                self.assertNotIn("pre_delay", data)


if __name__ == "__main__":
    unittest.main()
