"""成衣清理的多名称匹配、审核跳过、管理状态及删除结果。"""

import json
import re
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from tools.test_dreamweaver_space import ROOT, dream
from tools.validate_schema import load_jsonc


class CleanupTests(unittest.TestCase):
    BOX = (723, 175, 94, 93)
    NAME_BOX = (751, 277, 36, 20)
    CLICKED = [("临时", NAME_BOX)]

    def cleanup(self, name="临时"):
        context = Mock()
        context.tasker.stopping = False
        runner = dream.DreamRunner(context)
        runner.image = Mock(return_value=object())
        runner.hit = Mock(return_value=False)
        runner.click = Mock()
        return dream.CleanupRunner(runner, name)

    def test_invalid_or_empty_name_fails_before_any_ui_action(self):
        for name in (None, 1, "", "  ", "临时\n其他", "\r", ",", "，", "临时,", "临时， ，分享成衣"):
            with self.subTest(name=name), self.assertRaises(dream.DreamError):
                self.cleanup(name)

    def test_name_is_literal_and_does_not_remove_internal_spaces(self):
        self.assertEqual(self.cleanup(" 临 时 ").names, ("临 时",))
        self.assertEqual(self.cleanup(".*").names, (".*",))
        self.assertEqual(self.cleanup(" 临时， 分享成衣,临时 ").names, ("临时", "分享成衣"))

    def test_exact_name_only_selects_matching_literal(self):
        cleanup = self.cleanup(".*")
        cleanup.review = Mock(return_value="可清理")
        cleanup.items = Mock(return_value=[(".*", self.NAME_BOX)])
        result = cleanup.select_batch(object(), [("临时", self.NAME_BOX), (".*", self.NAME_BOX), (".*2", self.NAME_BOX)])
        self.assertEqual(result, [(".*", self.NAME_BOX)])
        cleanup.r.click.assert_called_once_with((769, 287, 1, 1))

    def test_multiple_names_select_all_matching_cards_once(self):
        cleanup = self.cleanup("临时，分享成衣")
        other_box = (859, 277, 72, 20)
        items = [("临时", self.NAME_BOX), ("分享成衣", other_box), ("分享成衣2", (984, 277, 90, 20))]
        cleanup.items = Mock(return_value=items)
        cleanup.review = Mock(return_value="可清理")
        cleanup.r.reco = Mock(side_effect=AssertionError("不应识别选中状态"))
        cleanup.wait = Mock()
        self.assertEqual(cleanup.select_batch(object(), [*items, items[0]]), items[:2])
        self.assertEqual([call.args[0] for call in cleanup.r.click.call_args_list],
                         [(769, 287, 1, 1), (895, 287, 1, 1)])
        cleanup.wait.assert_not_called()
        cleanup.r.reco.assert_not_called()

    def test_pending_or_uncertain_review_is_skipped_before_selection(self):
        for status in ("审核中", "状态不明"):
            cleanup = self.cleanup()
            cleanup.review = Mock(return_value=status)
            cleanup.items = Mock(return_value=self.CLICKED)
            cleanup.skip = Mock()
            self.assertEqual(cleanup.select_batch(object(), [("临时", self.NAME_BOX)]), [])
            cleanup.skip.assert_called_once_with("临时", status)
            cleanup.r.click.assert_not_called()

    def test_review_blocks_either_shen_or_he_but_not_zhong_or_other_text(self):
        cleanup = self.cleanup()
        image = SimpleNamespace(shape=(720, 1280, 3))
        cleanup.items = Mock()
        for text in ("审核中", "审", "核", "审核", "核中"):
            cleanup.items.return_value = [(text, (743, 244, 53, 21))]
            self.assertEqual(cleanup.review(image, self.NAME_BOX), "审核中", text)
        for text in ("中", "Nikki", "临时"):
            cleanup.items.return_value = [(text, (743, 244, 53, 21))]
            self.assertEqual(cleanup.review(image, self.NAME_BOX), "可清理", text)
        cleanup.items.return_value = []
        cleanup.r.hit.return_value = True
        self.assertEqual(cleanup.review(image, self.NAME_BOX), "可清理")
        cleanup.r.hit.assert_not_called()  # 不再做红色兜底检测。

    def test_review_region_is_relative_to_each_name_center(self):
        cleanup = self.cleanup()
        image = SimpleNamespace(shape=(720, 1280, 3))
        cleanup.items = Mock(return_value=[])
        self.assertEqual(cleanup.review(image, self.NAME_BOX), "可清理")
        cleanup.items.assert_called_with(image, "清理成衣作品文字", (737, 241, 64, 28))
        cleanup.review(image, (859, 277, 72, 20))
        cleanup.items.assert_called_with(image, "清理成衣作品文字", (863, 241, 64, 28))
        cleanup.items.reset_mock()
        self.assertEqual(cleanup.review(image, (751, 100, 36, 20)), "状态不明")
        cleanup.items.assert_not_called()

    def test_changed_name_is_not_selected(self):
        cleanup = self.cleanup()
        cleanup.review = Mock(return_value="可清理")
        cleanup.items = Mock(return_value=[("临时2", self.NAME_BOX)])
        cleanup.skip = Mock()
        self.assertEqual(cleanup.select_batch(object(), [("临时", self.NAME_BOX)]), [])
        cleanup.r.click.assert_not_called()

    def test_click_effect_does_not_trigger_post_selection_recognition_or_undo(self):
        cleanup = self.cleanup()
        cleanup.review = Mock(return_value="可清理")
        cleanup.items = Mock(side_effect=[[('临时', self.NAME_BOX)], AssertionError('不应识别点击后的文字')])
        cleanup.wait = Mock()
        self.assertEqual(cleanup.select_batch(object(), self.CLICKED), self.CLICKED)
        cleanup.r.click.assert_called_once_with((769, 287, 1, 1))
        cleanup.wait.assert_not_called()

    def test_empty_gallery_does_not_try_to_enter_management(self):
        cleanup = self.cleanup()
        cleanup.wait = Mock(return_value=True)
        cleanup.count = Mock(return_value=0)
        cleanup.manage = Mock()
        cleanup.garment()
        cleanup.manage.assert_not_called()

    def test_cancellation_or_total_timeout_stops_before_click(self):
        for stopping, deadline in ((True, float("inf")), (False, 0)):
            cleanup = self.cleanup()
            cleanup.r.context.tasker.stopping = stopping
            cleanup.deadline = deadline
            with self.assertRaises(dream.DreamError):
                cleanup.center(self.BOX)
            cleanup.r.click.assert_not_called()

    def prepared_delete(self):
        cleanup = self.cleanup()
        cleanup.count = Mock(return_value=2)
        cleanup.r.hit = Mock(side_effect=lambda node, image: node in ("清理成衣取消管理", "清理成衣删除可用"))
        cleanup.button = Mock()
        cleanup.wait = Mock(return_value=True)
        cleanup.skip = Mock()
        return cleanup

    def test_invalid_click_record_or_count_change_never_clicks_delete(self):
        for clicked, actual_count, expected_count in (([], 2, 2), (self.CLICKED * 3, 2, 2),
                                                     (self.CLICKED, 1, 2), (self.CLICKED, 0, 0),
                                                     ([("临时2", self.NAME_BOX)], 2, 2)):
            cleanup = self.prepared_delete()
            cleanup.count.return_value = actual_count
            with self.assertRaises(dream.DreamError):
                cleanup.delete_batch(clicked, expected_count)
            cleanup.button.assert_not_called()

    def test_disabled_delete_is_skipped_without_confirmation(self):
        cleanup = self.prepared_delete()
        cleanup.r.hit.side_effect = lambda node, image: node == "清理成衣取消管理"
        self.assertFalse(cleanup.delete_batch(self.CLICKED, 2))
        cleanup.button.assert_not_called()

    def test_missing_confirmation_or_unchanged_count_does_not_repeat_deletion(self):
        for outcomes, expected in (([dream.DreamError("无确认弹窗")], ["清理成衣删除可用"]),
                                   ([True, dream.DreamError("作品数量未减少")], ["清理成衣删除可用", "清理成衣确认删除"])):
            cleanup = self.prepared_delete()
            cleanup.wait.side_effect = outcomes
            with self.assertRaises(dream.DreamError):
                cleanup.delete_batch(self.CLICKED, 2)
            self.assertEqual([call.args[0] for call in cleanup.button.call_args_list], expected)
            self.assertEqual(cleanup.deleted, 0)

    def test_deletion_requires_exact_count_reduction(self):
        cleanup = self.prepared_delete()
        self.assertTrue(cleanup.delete_batch(self.CLICKED, 2))
        changed = cleanup.wait.call_args_list[-1].args[0]
        cleanup.r.hit.side_effect = lambda node, image: node == "清理成衣删除弹窗"
        self.assertFalse(changed(object()))
        cleanup.r.hit.side_effect = lambda node, image: node == "七日制衣切换部位"
        cleanup.count.return_value = 2
        self.assertFalse(changed(object()))
        cleanup.count.return_value = None
        self.assertFalse(changed(object()))  # 尚有作品时不能把面板消失当成成功。
        cleanup.count.return_value = 1
        self.assertTrue(changed(object()))
        self.assertEqual(cleanup.deleted, 1)

    def test_last_work_can_finish_by_automatic_return(self):
        cleanup = self.prepared_delete()
        cleanup.count.return_value = 1
        cleanup.delete_batch(self.CLICKED, 1)
        changed = cleanup.wait.call_args_list[-1].args[0]
        cleanup.count.return_value = None
        cleanup.r.hit.side_effect = lambda node, image: node == "七日制衣切换部位"
        self.assertTrue(changed(object()))
        cleanup.r.hit.side_effect = lambda node, image: node == "清理成衣成衣标签"
        self.assertTrue(changed(object()))
        cleanup.r.hit.return_value = False
        cleanup.r.hit.side_effect = None
        self.assertFalse(changed(object()))

    def test_category_return_does_not_prove_partial_deletion(self):
        cleanup = self.prepared_delete()
        cleanup.delete_batch(self.CLICKED, 2)
        changed = cleanup.wait.call_args_list[-1].args[0]
        cleanup.count.return_value = None
        cleanup.r.hit.side_effect = lambda node, image: node == "清理成衣成衣标签"
        self.assertFalse(changed(object()))

    def test_category_snapshot_survives_delete_return_and_position_changes(self):
        cleanup = self.cleanup()
        first = ("甲", (1150, 205, 70, 20))
        second = ("乙", (1150, 355, 70, 20))
        fresh_second = ("乙", (1150, 205, 70, 20))
        state = {"category": False}
        cleanup.r.hit.side_effect = lambda node, image: state["category"]
        cleanup.stable = Mock(side_effect=[(object(), [first, second]), (object(), [first, second]),
                                          (object(), [fresh_second, ("新增服装", (1150, 355, 70, 20))])])
        cleanup.reopen_category = Mock(side_effect=lambda category: state.update(category=False))
        cleanup.garment = Mock(side_effect=lambda: state.update(category=True))
        cleanup.button = Mock()
        cleanup.clean_category("连衣裙")
        cleanup.reopen_category.assert_called_once_with("连衣裙")
        self.assertEqual([call.args[0] for call in cleanup.r.click.call_args_list],
                         [(1185, 215, 1, 1), (1185, 215, 1, 1)])
        self.assertEqual(cleanup.garment.call_count, 2)
        cleanup.button.assert_not_called()  # 最后一件已返回类别页，不多点切换部位。
        self.assertEqual(cleanup.location, "连衣裙/乙")

    def test_cannot_delete_continues_to_next_then_switches_part_once(self):
        cleanup = self.cleanup()
        garments = [("甲", (1150, 205, 70, 20)), ("乙", (1150, 355, 70, 20))]
        cleanup.stable = Mock(return_value=(object(), garments))
        cleanup.reopen_category = Mock()
        cleanup.garment = Mock()  # 审核中/无匹配时留在管理页面。
        cleanup.button = Mock()
        cleanup.wait = Mock()
        cleanup.clean_category("上衣")
        self.assertEqual(cleanup.garment.call_count, 2)
        cleanup.reopen_category.assert_not_called()
        cleanup.button.assert_called_once_with("七日制衣切换部位")

    def test_missing_recorded_garment_is_skipped_without_clicking_wrong_row(self):
        cleanup = self.cleanup()
        cleanup.stable = Mock(side_effect=[(object(), [("甲", (1150, 205, 70, 20))]),
                                          (object(), [("其他", (1150, 205, 70, 20))])])
        cleanup.garment = Mock()
        cleanup.button = Mock()
        cleanup.wait = Mock()
        cleanup.clean_category("下装")
        cleanup.garment.assert_not_called()
        cleanup.r.click.assert_not_called()
        cleanup.button.assert_called_once_with("七日制衣切换部位")

    def test_reopen_category_uses_fresh_recognition_and_rejects_missing_name(self):
        cleanup = self.cleanup()
        box = (1155, 290, 100, 75)
        cleanup.stable = Mock(return_value=(object(), [("上衣", box)]))
        cleanup.wait = Mock()
        cleanup.reopen_category("上衣")
        cleanup.r.click.assert_called_once_with((1205, 327, 1, 1))
        cleanup.r.click.reset_mock()
        with self.assertRaises(dream.DreamError):
            cleanup.reopen_category("下装")
        cleanup.r.click.assert_not_called()

    def test_sidebar_uses_fresh_positions_and_finishes_without_swiping(self):
        cleanup = self.cleanup()
        items = [("甲", (1150, 205, 70, 20)), ("乙", (1150, 355, 70, 20))]
        cleanup.stable = Mock(side_effect=lambda reader, label: (object(), items))
        iterator = cleanup.walk_sidebar(categories=False)
        self.assertEqual(next(iterator)[0], "甲")
        items[:] = [("乙", (1150, 220, 70, 20))]
        self.assertEqual(next(iterator), ("乙", (1150, 220, 70, 20)))
        self.assertEqual(list(iterator), [])
        cleanup.r.context.run_action_direct.assert_not_called()

    def test_misread_rarity_labels_are_not_garment_candidates(self):
        cleanup = self.cleanup()
        cleanup.r.hit.return_value = True
        garments = [("花漾微风", (1162, 203, 70, 21)), ("轻甜仲夏", (1161, 356, 71, 21))]
        cleanup.items = Mock(return_value=[("称有", (1213, 89, 30, 21)), *garments,
                                          ("冈耀", (1213, 242, 30, 21))])
        self.assertEqual(cleanup.sidebar(object(), categories=False), garments)

    def test_coat_template_keeps_category_when_ocr_reads_only_tao(self):
        cleanup = self.cleanup()
        cleanup.r.hit.return_value = True
        boxes = {"七日制衣连衣裙": (1150, 95, 103, 70), "清理成衣外套": (1150, 184, 103, 70),
                 "七日制衣上衣": (1150, 286, 103, 70), "七日制衣下装": (1150, 378, 103, 70)}
        cleanup.r.box = Mock(side_effect=lambda node, image: boxes[node])
        cleanup.items = Mock(return_value=[("套", (1218, 231, 19, 18))])
        categories = cleanup.sidebar(object(), categories=True)
        self.assertEqual([name for name, _ in categories], ["连衣裙", "外套", "上衣", "下装"])
        self.assertEqual(categories[1][1], boxes["清理成衣外套"])

    def test_all_recognized_categories_including_coat_enter_cleanup(self):
        cleanup = self.cleanup()
        cleanup.wait = Mock()
        cleanup.button = Mock()
        cleanup.center = Mock()
        categories = [(name, (1150, 95 + i * 95, 103, 70))
                      for i, name in enumerate(("连衣裙", "外套", "上衣", "下装"))]
        cleanup.walk_sidebar = Mock(return_value=iter(categories))
        cleanup.clean_category = Mock()
        cleanup.run()
        self.assertEqual([call.args[0] for call in cleanup.clean_category.call_args_list],
                         ["连衣裙", "外套", "上衣", "下装"])

    def test_management_is_entered_once_and_never_cancelled(self):
        cleanup = self.cleanup()
        state = {"managed": False}
        cleanup.r.hit.side_effect = lambda node, image: state["managed"]
        cleanup.button = Mock(side_effect=lambda node: state.update(managed=True))
        cleanup.wait = Mock()
        for _ in range(3):
            self.assertTrue(cleanup.manage())
        cleanup.button.assert_called_once_with("清理成衣管理")
        cleanup.wait.assert_called_once()

    def test_no_matching_visible_works_keeps_management_without_swiping(self):
        cleanup = self.cleanup()
        cleanup.wait = Mock(return_value=True)
        cleanup.count = Mock(return_value=1)
        cleanup.manage = Mock(return_value=True)
        cleanup.stable = Mock(return_value=(object(), [("其他", self.NAME_BOX)]))
        cleanup.select_batch = Mock(return_value=[])
        cleanup.button = Mock()
        cleanup.garment()
        cleanup.button.assert_not_called()
        cleanup.r.context.run_action_direct.assert_not_called()

    def test_delete_reflow_keeps_management_and_reads_new_visible_items(self):
        cleanup = self.cleanup()
        cleanup.wait = Mock(return_value=True)
        cleanup.count = Mock(side_effect=[2, 2, 1])
        cleanup.manage = Mock(return_value=True)
        cleanup.stable = Mock(side_effect=[(object(), self.CLICKED), (object(), [("其他", self.NAME_BOX)])])
        cleanup.select_batch = Mock(side_effect=[self.CLICKED, []])
        cleanup.delete_batch = Mock(return_value=True)
        cleanup.button = Mock()
        cleanup.garment()
        self.assertEqual(cleanup.manage.call_count, 2)
        cleanup.delete_batch.assert_called_once_with(self.CLICKED, 2)
        cleanup.button.assert_not_called()

    def test_cleanup_is_opt_in_with_multiple_literal_names_and_weekly_hint(self):
        interface = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
        pipeline = load_jsonc(ROOT / "assets/resource/pipeline/dreamweaver_space.json")
        options = interface["option"]
        case = next(case for case in options["织梦空间子任务"]["cases"] if case["name"] == "清理成衣作品")
        self.assertNotIn(case["name"], options["织梦空间子任务"].get("default_case", []))
        self.assertFalse(pipeline["织梦选择清理成衣"]["enabled"])
        self.assertIn("每周", case["description"])
        option = options["清理成衣作品名称"]
        self.assertEqual(option["inputs"][0]["default"], "临时")
        self.assertIn("逗号", option["description"])
        self.assertNotIn("清理成衣已选边框", pipeline)
        self.assertNotIn("清理成衣审核红色", pipeline)
        verify = re.compile(option["inputs"][0]["verify"])
        for text in ("临时", " 临时，分享成衣 ", "临 时,.*"):
            self.assertIsNotNone(verify.fullmatch(text), text)
        for text in ("", "  ", "临时,", "临时， ，分享成衣", "临时,\n分享成衣"):
            self.assertIsNone(verify.fullmatch(text), text)
        self.assertEqual(option["pipeline_override"], {"清理成衣设置": {"attach": {"name": "{cleanup_name}"}}})


if __name__ == "__main__":
    unittest.main()
