"""Regression checks using supplied screenshot crops and complete glyph rows."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from maa.agent.agent_server import AgentServer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
from utils import stage_number
from utils.stage_number import load_templates, read_stage_candidates, read_stage_labels

spec = importlib.util.spec_from_file_location(
    "main_storyline_under_test", ROOT / "agent/custom/main_storyline.py"
)
module = importlib.util.module_from_spec(spec)
with patch.object(AgentServer, "register_custom_action", return_value=True):
    spec.loader.exec_module(module)


def make_label(text):
    """Compose original strokes to test combinations absent from the screenshots."""
    glyphs = [next(mask for char, mask in load_templates() if char == c) for c in text]
    height = max(mask.shape[0] for mask in glyphs)
    image = np.zeros((height + 12, sum(mask.shape[1] + 6 for mask in glyphs) + 6, 3), dtype=np.uint8)
    x = 6
    for char, mask in zip(text, glyphs):
        y = 6 + (round(height * 0.55) if char == "-" else height - mask.shape[0])
        image[y:y + mask.shape[0], x:x + mask.shape[1]][mask] = 255
        x += mask.shape[1] + 6
    return image


class StageNumberTemplatesTests(unittest.TestCase):
    def test_templates_include_every_digit_and_dash(self):
        self.assertEqual({char for char, _ in load_templates()}, set("0123456789-"))

    def test_all_ten_supplied_screenshot_labels(self):
        with np.load(ROOT / "tools/fixtures/main_storyline_stage_samples.npz", allow_pickle=False) as samples:
            for target in samples.files:
                with self.subTest(target=target):
                    self.assertEqual([text for text, _ in read_stage_labels(samples[target])], [target])

    def test_complete_labels_and_two_digit_suffixes(self):
        for target in ("1-1", "1-10", "1-11", "1-12", "2-2", "2-12", "2-21", "3-13", "12-2", "3-99"):
            with self.subTest(target=target):
                self.assertEqual([text for text, _ in read_stage_labels(make_label(target))], [target])

    def test_three_digit_suffix_is_not_a_substring_match(self):
        self.assertEqual(read_stage_labels(make_label("1-123")), [])

    def test_leading_zero_is_not_a_valid_stage(self):
        for text in ("1-02", "01-2", "1-0"):
            with self.subTest(text=text):
                self.assertEqual(read_stage_labels(make_label(text)), [])

    def test_unknown_suffix_glyph_invalidates_whole_label(self):
        image = make_label("2-21")
        # Replace the last 1 with a filled block, keeping the complete row.
        image[:, -18:] = 0
        image[6:26, -15:-8] = 255
        self.assertEqual(read_stage_labels(image), [])

    def test_missing_physical_digit_or_dash_is_not_reconstructed(self):
        for text in ("-2", "2-", "212"):
            with self.subTest(text=text):
                self.assertEqual(read_stage_labels(make_label(text)), [])

    def test_clipped_label_is_rejected(self):
        image = make_label("1-10")
        self.assertEqual(read_stage_labels(image[:, 6:]), [])

    def test_scale_variants(self):
        image = make_label("2-12")
        for scale in (0.8, 1.25, 1.5, 2.0):
            with self.subTest(scale=scale):
                ys = np.minimum((np.arange(round(image.shape[0] * scale)) / scale).astype(int), image.shape[0] - 1)
                xs = np.minimum((np.arange(round(image.shape[1] * scale)) / scale).astype(int), image.shape[1] - 1)
                scaled = image[np.ix_(ys, xs)]
                self.assertEqual([text for text, _ in read_stage_labels(scaled)], ["2-12"])

    def test_no_labels_on_empty_or_colour_only_image(self):
        for image in (np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((80, 100, 3), dtype=np.uint8), np.full((80, 100, 3), [255, 150, 100], dtype=np.uint8)):
            with self.subTest(shape=image.shape):
                self.assertEqual(read_stage_labels(image), [])

    def test_low_scores_preserve_whole_row_counts_and_unknown_positions(self):
        original = stage_number.classify_digit_detail

        def uncertain_two(mask):
            char, scores = original(mask)
            return (None, (("2", 0.69), ("3", 0.60))) if char == "2" else (char, scores)

        with patch.object(stage_number, "classify_digit_detail", side_effect=uncertain_two):
            candidates = read_stage_candidates(make_label("3-12"))
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.text, "3-1?")
        self.assertEqual((len(candidate.left), len(candidate.right)), (1, 2))
        self.assertFalse(candidate.complete)
        self.assertEqual(candidate.check_target("3-12"), "")
        self.assertIn("数字数量不符", candidate.check_target("3-1"))
        self.assertIn("字符冲突", candidate.check_target("2-12"))


class FindStageTests(unittest.TestCase):
    def setUp(self):
        self.action = module.MainStorylineFindStageAction()
        self.context = MagicMock()
        self.controller = MagicMock()
        self.context.run_recognition_direct.return_value = SimpleNamespace(hit=False)

    def find(self, image, target, ocr_text=None):
        h, w = image.shape[:2]
        items = [] if ocr_text is None else [SimpleNamespace(text=ocr_text, box=[4, 4, w - 8, h - 8])]
        with patch.object(self.action, "_ocr_items", return_value=items):
            return self.action._find_stage(self.context, self.controller, [0, 0, w, h], target, 0.3, image=image)

    def uncertain_twos(self):
        original = stage_number.classify_digit_detail

        def uncertain(mask):
            char, scores = original(mask)
            return (None, (("2", 0.69), ("3", 0.60))) if char == "2" else (char, scores)

        return patch.object(stage_number, "classify_digit_detail", side_effect=uncertain)

    def local_ocr(self, text):
        self.context.run_recognition_direct.return_value = SimpleNamespace(
            hit=bool(text), best_result=SimpleNamespace(text=text), filtered_results=[], all_results=[]
        )

    def test_even_exact_ocr_cannot_override_complete_visual_number(self):
        for actual, wrong in (("1-10", "1-1"), ("2-12", "2-2"), ("2-21", "2-2"), ("1-11", "1-1")):
            with self.subTest(actual=actual, wrong=wrong):
                self.assertIsNone(self.find(make_label(actual), wrong, wrong))
        self.controller.post_click.assert_not_called()

    def test_missing_ocr_dash_and_one_use_digit_templates(self):
        for ocr in (None, "212", "2-2", "-12", "2-", "2-12"):
            with self.subTest(ocr=ocr):
                self.assertIsNotNone(self.find(make_label("2-12"), "2-12", ocr))
        self.controller.post_click.assert_not_called()

    def test_local_recognition_enlarges_label_and_skips_detector(self):
        image = make_label("2-12")
        self.context.run_recognition_direct.return_value = SimpleNamespace(
            hit=True, filtered_results=[SimpleNamespace(text="2-12")], all_results=[], best_result=None
        )
        with self.uncertain_twos():
            match = self.find(image, "2-12", "2-2")
        args = self.context.run_recognition_direct.call_args.args
        self.assertTrue(args[1].only_rec)
        self.assertEqual(args[1].roi, (0, 0, args[2].shape[1], args[2].shape[0]))
        self.assertGreater(args[2].shape[0], match[3] * 3)
        self.assertGreater(args[2].shape[1], match[2] * 3)

    def test_exact_ocr_agrees_with_templates_without_local_retry(self):
        self.assertIsNotNone(self.find(make_label("2-12"), "2-12", "2-12"))
        self.context.run_recognition_direct.assert_not_called()

    def test_native_result_views_do_not_repeat_the_same_text(self):
        # The SDK constructs separate objects for best/filtered/all views.
        self.context.run_recognition_direct.return_value = SimpleNamespace(
            hit=True,
            best_result=SimpleNamespace(text="2-12"),
            filtered_results=[SimpleNamespace(text="2-12")],
            all_results=[SimpleNamespace(text="2-12")],
        )
        image = make_label("2-12")
        box = read_stage_labels(image)[0][1]
        self.assertEqual(self.action._read_stage_crop(self.context, image, box, 0.3), "2-12")

    def test_dashless_ocr_still_requires_visual_separator(self):
        self.assertIsNotNone(self.find(make_label("2-12"), "2-12", "212"))
        self.assertIsNone(self.find(make_label("212"), "2-12", "212"))

    def test_return_box_respects_roi_offset(self):
        label = make_label("3-13")
        image = np.zeros((label.shape[0] + 50, label.shape[1] + 100, 3), dtype=np.uint8)
        image[20:20 + label.shape[0], 70:70 + label.shape[1]] = label
        local_box = read_stage_labels(label)[0][1]
        with patch.object(self.action, "_ocr_items", return_value=[]):
            box = self.action._find_stage(self.context, self.controller, [70, 20, label.shape[1], label.shape[0]], "3-13", 0.3, image=image)
        self.assertEqual(box, [local_box[0] + 70, local_box[1] + 20, local_box[2], local_box[3]])

    def test_invalid_roi_returns_no_match(self):
        for roi in ([0, 0, 0, 30], [0, 0, 30, -1], [1000, 1000, 40, 40]):
            with self.subTest(roi=roi):
                self.assertIsNone(self.action._find_stage(self.context, self.controller, roi, "1-1", 0.3, image=make_label("1-1")))

    def test_matching_never_makes_a_digit_optional(self):
        for text in ("2-2", "2-112", "12-12", "22", "12"):
            self.assertFalse(self.action._stage_text_matches(text, "2-12"))
        for text in ("2-12", "2－12", "2 - 12", "212"):
            self.assertTrue(self.action._stage_text_matches(text, "2-12"))
        for text in ("2-12 3-13", "前往2-12", "2-12关卡"):
            self.assertFalse(self.action._stage_text_matches(text, "2-12"))

    def test_local_ocr_rescues_a_low_score_digit_in_3_12(self):
        self.local_ocr("3-12")
        with self.uncertain_twos():
            self.assertIsNotNone(self.find(make_label("3-12"), "3-12", "3-2"))
        self.context.run_recognition_direct.assert_called_once()
        self.controller.post_click.assert_not_called()

    def test_local_dashless_ocr_can_rescue_the_same_complete_row(self):
        self.local_ocr("312")
        with self.uncertain_twos():
            self.assertIsNotNone(self.find(make_label("3-12"), "3-12", "3-2"))

    def test_global_ocr_alone_cannot_rescue_a_low_score_digit(self):
        for local in ("", "3-2", "3-13", "前往3-12"):
            with self.subTest(local=local):
                self.local_ocr(local)
                with self.uncertain_twos():
                    self.assertIsNone(self.find(make_label("3-12"), "3-12", "3-12"))

    def test_uncertain_extra_digit_cannot_be_discarded_to_match_shorter_suffix(self):
        self.local_ocr("3-1")
        with self.uncertain_twos():
            self.assertIsNone(self.find(make_label("3-12"), "3-1", "3-1"))
        self.context.run_recognition_direct.assert_not_called()

    def test_confirmed_digit_conflict_rejects_even_matching_local_ocr(self):
        self.local_ocr("3-13")
        self.assertIsNone(self.find(make_label("3-12"), "3-13", "3-13"))
        self.context.run_recognition_direct.assert_not_called()

    def test_no_template_candidate_still_runs_global_and_local_ocr_but_cannot_click(self):
        self.local_ocr("3-12")
        with patch.object(module, "read_stage_candidates", return_value=[]):
            self.assertIsNone(self.find(make_label("3-12"), "3-12", "3-12"))
        self.context.run_recognition_direct.assert_called_once()

    def test_ocr_box_crop_can_recover_segmentation_and_verify_locally(self):
        self.local_ocr("3-12")
        image = make_label("3-12")
        candidates = read_stage_candidates(image)
        with patch.object(module, "read_stage_candidates", side_effect=[[], candidates]):
            self.assertEqual(self.find(image, "3-12", "3-12"), candidates[0].box)
        self.context.run_recognition_direct.assert_called_once()

    def test_rejected_candidate_does_not_log_search_progress(self):
        self.local_ocr("3-2")
        with self.uncertain_twos(), patch.object(module, "logger") as logger:
            self.assertIsNone(self.find(make_label("3-12"), "3-12", "3-2"))
        logger.info.assert_not_called()
        logger.debug.assert_not_called()


class FindStageFlowTests(unittest.TestCase):
    def setUp(self):
        self.action = module.MainStorylineFindStageAction()
        self.context = MagicMock()
        self.context.run_action_direct.return_value = SimpleNamespace(success=True)
        self.argv = SimpleNamespace(node_name="主线查找第三章关卡", custom_action_param=json.dumps({
            "chapter_number": 3, "stage_suffix": 12, "wait_after_swipe": 0,
            "rounds": 1, "max_scan_swipes": 1,
        }))

    def test_initial_screenshot_hit_never_resets_or_scans(self):
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        label = make_label("3-12")
        image[320:320 + label.shape[0], 140:140 + label.shape[1]] = label
        with patch.object(self.action, "_screencap", return_value=image) as screencap, \
             patch.object(self.action, "_open_stage", return_value=True), \
             patch.object(self.action, "_is_non_main_panel", return_value=False):
            self.assertTrue(self.action.run(self.context, self.argv))
        screencap.assert_called_once()
        self.context.run_action_direct.assert_not_called()

    def test_miss_then_three_default_resets_then_scan(self):
        events, swipes = [], []
        results = iter([None, None, [140, 320, 55, 20]])

        def find(*args, **kwargs):
            events.append("查找")
            return next(results)

        def swipe(kind, params):
            swipes.append(params)
            events.append("复位" if params.begin == [61, 500] else "扫描")
            return SimpleNamespace(success=True)

        self.context.run_action_direct.side_effect = swipe
        with patch.object(self.action, "_find_stage", side_effect=find), \
             patch.object(self.action, "_screencap", return_value=np.zeros((1280, 720, 3), dtype=np.uint8)), \
             patch.object(self.action, "_open_stage", return_value=True), \
             patch.object(self.action, "_is_non_main_panel", return_value=False):
            self.assertTrue(self.action.run(self.context, self.argv))
        self.assertEqual(events, ["查找", "复位", "复位", "复位", "查找", "扫描", "查找"])
        for params in swipes[:3]:
            self.assertEqual(params.end, [698, 500])
            self.assertEqual(params.duration, module.JSwipe().duration)
            self.assertEqual(params.end_hold, module.JSwipe().end_hold)
        self.assertEqual(swipes[-1].begin, [614, 602])
        self.assertEqual(swipes[-1].end, [0, 602])
        self.assertEqual(swipes[-1].duration, module.JSwipe().duration)
        self.assertEqual(swipes[-1].end_hold, 500)

    def test_all_chapter_pipeline_nodes_use_the_new_scan(self):
        nodes = json.loads((ROOT / "assets/resource/pipeline/main_storyline.json").read_text(encoding="utf-8"))
        for name in ("主线查找第一章关卡", "主线查找第二章关卡", "主线查找第三章关卡"):
            with self.subTest(name=name):
                params = nodes[name]["action"]["param"]["custom_action_param"]
                self.assertEqual(params["scan_begin"], [614, 602])
                self.assertEqual(params["scan_end"], [0, 602])
                self.assertEqual(params["end_hold"], 500)
                self.assertNotIn("duration", params)
                self.assertEqual(params["reset_swipes"], 3)

    def test_explicit_slow_challenge_scan_parameters_are_preserved(self):
        self.argv.custom_action_param = json.dumps({
            "chapter_number": 2, "stage_suffix": 1, "required_panel_title": "挑战关卡2-1",
            "scan_begin": [698, 500], "scan_end": [61, 500], "duration": 6000, "end_hold": 2000,
        })
        with patch.object(self.action, "_scan_for_required_panel", return_value=True) as scan:
            self.assertTrue(self.action.run(self.context, self.argv))
        self.assertEqual(scan.call_args.kwargs["duration"], 6000)
        self.assertEqual(scan.call_args.kwargs["end_hold"], 2000)
        self.assertEqual(scan.call_args.kwargs["scan_begin"], [698, 500])


if __name__ == "__main__":
    unittest.main()
