import re
import time

from .logger import logger

__all__ = [
    "extract_ocr_text",
    "iter_recognition_results",
    "extract_text_from_raw_detail",
    "extract_number_before_slash",
    "read_ocr_text",
    "read_ocr_number",
    "read_number_from_controller",
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
        logger.warning(f"{log_prefix}: OCR error: {e}")
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
            logger.warning(
                f"{log_prefix}: failed to OCR {retry_label} on "
                f"attempt {attempt + 1}/{retry}"
            )
            time.sleep(retry_delay)

    return None
