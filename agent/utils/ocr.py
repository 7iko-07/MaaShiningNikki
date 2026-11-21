import re
import time
import unicodedata
from numbers import Integral

from maa.pipeline import JOCR, JRecognitionType

from .logger import logger

__all__ = [
    "extract_ocr_text",
    "iter_recognition_results",
    "extract_text_from_raw_detail",
    "extract_number_before_slash",
    "read_ocr_text",
    "read_ocr_number",
    "read_number_from_controller",
    "read_ocr_fraction",
    "read_fraction_from_controller",
]


def extract_ocr_text(result):
    if not result or not getattr(result, "hit", False):
        return ""

    for candidate in iter_recognition_results(result):
        text = getattr(candidate, "text", None)
        if text:
            return str(text)

        detail = getattr(candidate, "detail", None)
        if detail:
            return str(detail)

    return extract_text_from_raw_detail(getattr(result, "raw_detail", None))


def iter_recognition_results(result):
    best_result = getattr(result, "best_result", None)
    if best_result is not None:
        yield best_result

    for attr in ("filtered_results", "all_results"):
        for item in getattr(result, attr, []) or []:
            if item is not None:
                yield item


def extract_text_from_raw_detail(raw_detail):
    if isinstance(raw_detail, dict):
        for key in ("text", "detail"):
            value = raw_detail.get(key)
            if value:
                return str(value)

        for key in ("best", "filtered", "all"):
            value = raw_detail.get(key)
            text = extract_text_from_raw_detail(value)
            if text:
                return text

    if isinstance(raw_detail, list):
        for item in raw_detail:
            text = extract_text_from_raw_detail(item)
            if text:
                return text

    return ""


def extract_number_before_slash(text):
    text = str(text).replace(",", "").replace(" ", "")
    num_str = re.sub(r"[^\d]", "", text.split("/", 1)[0])
    return int(num_str) if num_str else None


def read_ocr_text(context, image, node_name, roi, expected, log_prefix="ocr"):
    try:
        result = context.run_recognition(
            node_name,
            image,
            pipeline_override={
                node_name: {
                    "recognition": "OCR",
                    "roi": roi,
                    "expected": expected,
                }
            },
        )
        return extract_ocr_text(result)
    except Exception as e:
        logger.debug(f"{log_prefix}: OCR error: {e}")
        return ""


def read_ocr_number(context, image, node_name, roi, expected, log_prefix="ocr"):
    text = read_ocr_text(context, image, node_name, roi, expected, log_prefix)
    return extract_number_before_slash(text) if text else None


def read_number_from_controller(
    context,
    controller,
    roi,
    node_name,
    expected,
    retry=1,
    retry_delay=0.0,
    log_prefix="ocr",
    retry_label="number",
):
    try:
        retry = max(1, int(retry))
    except (TypeError, ValueError):
        retry = 1

    try:
        retry_delay = float(retry_delay)
    except (TypeError, ValueError):
        retry_delay = 0.0

    for attempt in range(retry):
        controller.post_screencap().wait()
        number = read_ocr_number(
            context,
            controller.cached_image,
            node_name,
            roi,
            expected,
            log_prefix,
        )
        if number is not None:
            return number

        if attempt < retry - 1:
            logger.debug(
                f"{log_prefix}: failed to OCR {retry_label} on "
                f"attempt {attempt + 1}/{retry}"
            )
            time.sleep(retry_delay)

    return None


def _ocr_box(value):
    """OCR result boxes may be JSON lists even when the SDK annotates Rect."""
    if not isinstance(value, (list, tuple)):
        value = tuple(getattr(value, name, None) for name in ("x", "y", "w", "h"))
    if len(value) != 4 or any(isinstance(part, bool) or not isinstance(part, Integral) for part in value):
        return None
    x, y, w, h = value
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _fraction_value(result, label=""):
    """Accept a complete fraction or two intact, horizontally aligned number boxes."""
    if result is None or not result.hit:
        return None
    candidates = getattr(result, "all_results", None) or getattr(result, "filtered_results", None) or []
    if not candidates and getattr(result, "best_result", None) is not None:
        candidates = [result.best_result]
    parts = []
    for item in candidates:
        text = unicodedata.normalize("NFKC", str(getattr(item, "text", ""))).strip()
        fraction_text = re.sub(rf"^{re.escape(label)}\s*:?\s*", "", text) if label else text
        match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", fraction_text)
        if match and int(match[2]) > 0:
            parts.append((item, text, int(match[1])))
        elif text:
            parts.append((item, text, None))
    if len(parts) == 1:
        return parts[0][2]
    if len(parts) not in (2, 3):
        return None
    if any(getattr(item, "score", 1.0) < .7 or getattr(item, "box", None) is None
           for item, _, _ in parts):
        return None
    positioned = [(_ocr_box(item.box), text) for item, text, _ in parts]
    if any(box is None for box, _ in positioned):
        return None
    positioned.sort(key=lambda part: part[0][0])
    boxes = [box for box, _ in positioned]
    if max(box[1] for box in boxes) >= min(box[1] + box[3] for box in boxes):
        return None
    if any(left[0] + left[2] > right[0] for left, right in zip(boxes, boxes[1:])):
        return None
    texts = [text for _, text in positioned]
    if len(texts) == 3:
        if texts[1] != "/" or not all(re.fullmatch(r"\d+", text) for text in (texts[0], texts[2])):
            return None
        text = "".join(texts)
    elif all(re.fullmatch(r"\d+", text) for text in texts):
        text = "/".join(texts)
    elif ((re.fullmatch(r"\d+", texts[0]) and re.fullmatch(r"/\s*\d+", texts[1]))
          or (re.fullmatch(r"\d+\s*/", texts[0]) and re.fullmatch(r"\d+", texts[1]))):
        text = "".join(texts)
    else:
        return None
    match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", text)
    if match and int(match[2]) > 0:
        return int(match[1])
    return None


def read_ocr_fraction(context, image, roi, label=""):
    """Read the numerator, retrying the same region as a single line if needed."""
    for only_rec in (False, True):
        result = context.run_recognition_direct(JRecognitionType.OCR, JOCR(
            roi=tuple(roi), expected=[".*"], only_rec=only_rec,
        ), image)
        value = _fraction_value(result, label)
        if value is not None:
            return value
    return None


def read_fraction_from_controller(context, controller, roi, retry, retry_delay):
    for attempt in range(retry):
        if context.tasker.stopping or not controller.post_screencap().wait().succeeded:
            return None
        value = read_ocr_fraction(context, controller.cached_image, roi)
        if value is not None:
            return value
        logger.debug(f"体力识别未命中，尝试次数 {attempt + 1}/{retry}")
        if attempt < retry - 1 and retry_delay > 0:
            time.sleep(retry_delay)
    return None
