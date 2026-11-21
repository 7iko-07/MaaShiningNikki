import json
import re
import sqlite3
import time
import unicodedata
from datetime import timedelta

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.custom_recognition import CustomRecognition
from maa.context import Context
from maa.pipeline import JClick, JOCR

from utils import logger
from utils import arena_state as state
from utils.ocr import extract_ocr_text, extract_text_from_raw_detail, iter_recognition_results


PROGRESS_PATTERN = re.compile(r"(?<!\d)(\d+)\s*[/／]\s*(\d+)(?!\d)")
PROGRESS_OCR_NODE = "机密任务今日执行次数OCR"
DEFAULT_SINGLE_BUTTON_NODE = "机密任务执行一次按钮OCR"
DEFAULT_MULTIPLE_BUTTON_NODE = "机密任务执行多次按钮OCR"


def parse_execution_progress(text):
    """Return (remaining, total) from text such as '今日执行次数：1/5'."""
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    match = PROGRESS_PATTERN.search(normalized)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def is_execution_complete(text):
    progress = parse_execution_progress(text)
    return bool(progress and progress[0] < 1)


def execution_button_node_for_remaining(
    remaining,
    single_button_node=DEFAULT_SINGLE_BUTTON_NODE,
    multiple_button_node=DEFAULT_MULTIPLE_BUTTON_NODE,
):
    if remaining < 1:
        return None
    if remaining == 1:
        return single_button_node
    return multiple_button_node


def _collect_ocr_text(result):
    texts = []
    for candidate in iter_recognition_results(result):
        text = getattr(candidate, "text", None)
        if text:
            texts.append(str(text))

    raw_text = extract_text_from_raw_detail(getattr(result, "raw_detail", None))
    if raw_text:
        texts.append(raw_text)
    return " ".join(texts)


@AgentServer.custom_action("union_secret_task_execute")
class UnionSecretTaskExecuteAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            return self._execute(context, argv)
        except Exception as e:
            logger.debug(f"union_secret_task_execute: 执行失败: {e}")
            logger.error("联盟机密任务执行异常")
            return False

    def _execute(self, context, argv):
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        progress_node = params.get("progress_node", PROGRESS_OCR_NODE)
        completed_next = params.get("completed_next", "点击返回联盟")
        single_button_node = params.get(
            "single_button_node",
            DEFAULT_SINGLE_BUTTON_NODE,
        )
        multiple_button_node = params.get(
            "multiple_button_node",
            DEFAULT_MULTIPLE_BUTTON_NODE,
        )
        purchase_popup_node = params.get(
            "purchase_popup_node",
            "机密任务购买次数弹窗",
        )
        cancel_purchase_node = params.get(
            "cancel_purchase_node",
            "取消购买机密任务次数",
        )
        reward_node = params.get("reward_node", "联盟执行多次奖励")
        weekly_node = params.get("weekly_popup_node", "关闭机密任务周奖励弹窗")
        close_reward_node = params.get("close_reward_node", "机密任务关闭奖励弹窗")
        deadline = time.monotonic() + max(0, int(params.get("timeout", 30000))) / 1000
        controller = context.tasker.controller
        stable_progress, stable_frames = None, 0
        clicked_remaining = None
        reward_seen = False
        next_click = 0.0
        first_poll = True

        def active():
            return not context.tasker.stopping and time.monotonic() < deadline

        while active():
            if not first_poll:
                time.sleep(min(0.25, max(0, deadline - time.monotonic())))
            first_poll = False
            if not active():
                break
            image = self._screencap(controller)
            if image is None or not active():
                return False
            # 弹窗优先于背景中的次数和按钮；仅执行关闭动作，不进入其 next 子链。
            if self._recognize(context, image, purchase_popup_node):
                if not active():
                    break
                logger.debug("union_secret_task_execute: 出现购买次数弹窗，取消购买并终止任务")
                logger.error("联盟机密任务：出现购买次数弹窗，取消购买并停止任务")
                result = context.run_action(cancel_purchase_node)
                if not result or not result.success:
                    logger.debug("union_secret_task_execute: 取消购买次数失败")
                    logger.error("联盟机密任务：未能取消购买，请手动检查")
                return False
            weekly = self._recognize(context, image, weekly_node)
            reward = not weekly and self._recognize(context, image, reward_node)
            if weekly or reward:
                stable_progress, stable_frames = None, 0
                if reward:
                    reward_seen = True
                if active() and time.monotonic() >= next_click:
                    result = context.run_action(close_reward_node)
                    if not result or not result.success:
                        logger.debug("union_secret_task_execute: 关闭奖励弹窗失败")
                        logger.error("联盟机密任务：关闭奖励弹窗失败")
                        return False
                    next_click = time.monotonic() + 1
                continue

            result = self._run_recognition(context, image, progress_node)
            progress = (parse_execution_progress(_collect_ocr_text(result))
                        if result and result.hit else None)
            if not progress:
                stable_progress, stable_frames = None, 0
                continue
            remaining, total = progress
            target = None
            if remaining > 0:
                button_node = execution_button_node_for_remaining(
                    remaining, single_button_node, multiple_button_node,
                )
                button = self._run_recognition(context, image, button_node)
                target = self._execution_text_box(button)
                if not self._valid_target(target):
                    stable_progress, stable_frames = None, 0
                    continue
            if progress == stable_progress:
                stable_frames += 1
            else:
                stable_progress, stable_frames = progress, 1
            if stable_frames < 2 or not active():
                continue
            if remaining == 0:
                if not context.override_next(argv.node_name, [completed_next]):
                    logger.debug("union_secret_task_execute: 完成后继设置失败")
                    logger.error("联盟机密任务：未能正常结束")
                    return False
                logger.debug(f"联盟机密任务: 今日剩余次数={remaining}/{total}，执行完成")
                logger.info(f"联盟机密任务：执行完成，今日剩余 {remaining}/{total} 次")
                return True
            if clicked_remaining is not None:
                if remaining > clicked_remaining:
                    logger.debug("union_secret_task_execute: 剩余次数异常增加，停止执行")
                    logger.error("联盟机密任务：剩余次数异常，停止执行")
                    return False
                if remaining < clicked_remaining:
                    logger.debug(f"union_secret_task_execute: 剩余次数 {clicked_remaining} -> {remaining}")
                    clicked_remaining = None
                    reward_seen = False
                elif reward_seen:
                    # 已确认执行奖励，次数尚未刷新时不能重复执行。
                    continue
            if active() and time.monotonic() >= next_click:
                logger.debug(f"union_secret_task_execute: remaining={remaining}/{total}, "
                             f"button_node={button_node!r}, recognized_box={target}")
                if not self._click_roi(controller, target):
                    logger.debug("union_secret_task_execute: 点击执行失败")
                    logger.error("联盟机密任务：点击执行失败")
                    return False
                clicked_remaining = remaining
                stable_progress, stable_frames = None, 0
                next_click = time.monotonic() + 1

        logger.debug("union_secret_task_execute: 任务取消或等待执行结果超时，未确认完成")
        logger.error("联盟机密任务：未能及时确认执行结果，任务已停止")
        return False

    def _recognize(self, context, image, node_name):
        result = self._run_recognition(context, image, node_name)
        if result is None:
            raise RuntimeError(f"无法检查弹窗：{node_name}")
        return bool(result.hit)

    def _run_recognition(self, context, image, node_name):
        try:
            return context.run_recognition(node_name, image)
        except Exception as e:
            logger.debug(
                f"union_secret_task_execute: 识别 {node_name!r} 失败: {e}"
            )
            return None

    def _execution_text_box(self, result):
        if not result or not getattr(result, "hit", False):
            return None

        for candidate in iter_recognition_results(result):
            text = unicodedata.normalize(
                "NFKC",
                str(getattr(candidate, "text", "") or ""),
            ).replace(" ", "")
            if "执行" not in text:
                continue

            box = getattr(candidate, "box", None)
            if isinstance(box, (list, tuple)) and len(box) == 4:
                return list(box)
            if box is not None and all(
                hasattr(box, field) for field in ("x", "y", "w", "h")
            ):
                return [box.x, box.y, box.w, box.h]
        return None

    def _screencap(self, controller):
        capture = controller.post_screencap().wait()
        if not capture.succeeded or controller.cached_image is None:
            logger.debug("union_secret_task_execute: 截图失败")
            logger.error("联盟机密任务：获取画面失败")
            return None
        return controller.cached_image.copy()

    def _click_roi(self, controller, roi):
        x, y, w, h = roi
        return controller.post_click(x + w // 2, y + h // 2).wait().succeeded

    def _valid_target(self, target):
        return (
            isinstance(target, (list, tuple))
            and len(target) == 4
            and all(isinstance(value, (int, float)) for value in target)
            and target[2] > 0
            and target[3] > 0
        )


@AgentServer.custom_recognition("union_secret_task_ready")
class UnionSecretTaskReadyRecognition(CustomRecognition):
    """Only enter the action when both the count and its button are visible.

    A miss lets the parent retry its entire candidate list within its timeout,
    including popups and the user-selected manual execution branch.
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult | None:
        helper = UnionSecretTaskExecuteAction()
        result = helper._run_recognition(context, argv.image, PROGRESS_OCR_NODE)
        if not result or not getattr(result, "hit", False):
            return None
        progress = parse_execution_progress(_collect_ocr_text(result))
        if not progress or progress[0] < 1:
            return None

        button_node = execution_button_node_for_remaining(progress[0])
        button = helper._run_recognition(context, argv.image, button_node)
        target = helper._execution_text_box(button)
        if not helper._valid_target(target):
            return None
        return CustomRecognition.AnalyzeResult(
            box=tuple(target),
            detail={"remaining": progress[0], "total": progress[1]},
        )


@AgentServer.custom_recognition("union_secret_task_complete")
class UnionSecretTaskCompleteRecognition(CustomRecognition):
    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult | None:
        try:
            result = context.run_recognition(PROGRESS_OCR_NODE, argv.image)
        except Exception as e:
            logger.debug(f"union_secret_task_complete: OCR error: {e}")
            return None

        text = _collect_ocr_text(result)
        progress = parse_execution_progress(text)
        if not progress:
            return None

        remaining, total = progress
        if remaining >= 1:
            return None

        logger.debug(
            "联盟机密任务: "
            f"今日剩余次数={remaining}/{total}"
        )
        return CustomRecognition.AnalyzeResult(
            box=(0, 1210, 235, 70),
            detail={
                "text": text,
                "remaining": remaining,
                "total": total,
            },
        )


BURST_MODEL = "ppocr_v6"
BURST_PARTS = {
    "裙子": "连衣裙|上下装",
    "发型": "发型",
    "耳饰": "耳饰",
    "项圈": "颈饰|项圈",
    "头饰": "头饰",
    "面饰": "面饰",
    "手饰": "手饰",
    "手持": "手持",
    "背饰": "背饰",
    "摆件": "摆件",
    "鞋袜": "鞋袜",
}
BURST_PART_PATTERN = "^[一-龥]{1,6}评分$"
BURST_START_OVERRIDE = {
    "联盟机密任务入口分流": {"next": ["点击开始执行"]},
    "等待机密任务比拼中": {"next": []},
    **{
        node: {"recognition": {"param": {"model": BURST_MODEL}}}
        for node in (
            "点击开始执行", "点击竞技场-推荐搭配", "点击竞技场-代理评选",
            "点击竞技场-下一步", "点击竞技场-确定影召",
        )
    },
}


def secret_task_week_key(now):
    now = now.astimezone(state.BEIJING)
    monday = (now - timedelta(days=now.weekday())).replace(
        hour=5, minute=0, second=0, microsecond=0,
    )
    if now < monday:
        monday -= timedelta(days=7)
    return monday.isoformat()


def secret_task_settings(context, mode):
    settings = {"mode": mode}
    if mode == "burst":
        options = context.get_node_data("机密任务刷爆发")["attach"]
        settings.update(
            total=options["total"],
            required=sorted(part for part in BURST_PARTS if options.get(part)),
            minutes=options["minutes"],
            continue_on_timeout=options.get("continue_on_timeout", False),
        )
    return json.dumps(settings, ensure_ascii=False, sort_keys=True)


@AgentServer.custom_action("union_secret_task_weekly")
class UnionSecretTaskWeeklyAction(CustomAction):
    def __init__(self):
        super().__init__()
        self._runs = {}

    def run(self, context, argv):
        params = json.loads(argv.custom_action_param or "{}")
        operation = params["operation"]
        device = context.tasker.controller.uuid
        task_id = argv.task_detail.task_id
        now = state.now_beijing()
        if operation == "enter":
            options = context.get_node_data(argv.node_name).get("attach", {})
            mode = options.get("mode", "execute")
            self._runs[device] = {
                "task_id": task_id,
                "account": state.take_account(context, "union"),
                "week": secret_task_week_key(now),
                "settings": secret_task_settings(context, mode),
                "mode": mode,
                "weekly": options.get("weekly", False),
                "pending": False,
            }
            return True

        run = self._runs.get(device)
        if not run or run["task_id"] != task_id:
            logger.error("联盟机密任务：请从任务列表重新运行")
            return False
        if operation == "choose":
            if run["mode"] not in ("start", "burst"):
                return context.override_next(argv.node_name, ["点击执行多次"])
            if run["weekly"] and run["account"]:
                try:
                    if state.StateStore().secret_task_completed(
                        run["account"], run["week"], run["settings"],
                    ):
                        logger.info("联盟机密任务：本周已按当前设置完成，直接执行多次")
                        return context.override_next(argv.node_name, ["点击执行多次"])
                except (OSError, sqlite3.Error) as error:
                    logger.debug(f"机密任务周记录读取失败：{error}")
                    logger.warning("联盟机密任务：无法读取本周记录，本次照常执行")
            if run["weekly"] and run["account"] is None:
                logger.warning("联盟机密任务：账号未确认，请先运行启动任务以使用每周记录")
            next_node = "机密任务刷爆发" if run["mode"] == "burst" else "点击开始执行"
            if not context.override_next(argv.node_name, [next_node]):
                return False
            run["pending"] = True
            return True

        if operation == "complete":
            if not run["pending"]:
                logger.error("联盟机密任务：本次比拼尚未确认，未保存本周记录")
                return False
            if context.tasker.stopping:
                return False
            if run["account"] and run["week"] == secret_task_week_key(now):
                try:
                    state.StateStore().mark_secret_task_completed(
                        run["account"], run["week"], run["settings"], now,
                    )
                    logger.debug("联盟机密任务：本周执行记录已保存")
                except (OSError, sqlite3.Error) as error:
                    logger.debug(f"机密任务周记录保存失败：{error}")
                    logger.warning("联盟机密任务：本周记录保存失败，下次会重新执行")
            run["pending"] = False
            return True
        logger.error("联盟机密任务：周记录操作配置无效")
        return False


@AgentServer.custom_action("union_secret_task_burst")
class UnionSecretTaskBurstAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            return self._run(context, argv)
        except Exception as error:
            logger.debug(f"union_secret_task_burst: {error}")
            if not context.tasker.stopping:
                logger.error(f"联盟机密任务：{error}" if isinstance(error, ValueError)
                             else "联盟机密任务：刷爆发异常，请检查游戏画面")
            return False

    def _run(self, context, argv):
        options = context.get_node_data(argv.node_name)["attach"]
        minutes = options["minutes"]
        if type(minutes) is not int or minutes < 1:
            raise ValueError("刷取上限须为正整数分钟")
        total = options["total"]
        if total not in (2, 3):
            raise ValueError("请选择双爆或三爆")
        required = {part for part in BURST_PARTS if options.get(part)}
        if len(required) > total:
            raise ValueError(f"部位勾多了，请只勾 {total} 个或更少")
        continue_on_timeout = options.get("continue_on_timeout", False)

        deadline = time.monotonic() + minutes * 60
        progress = self._wait_progress(context, 10)
        if progress is None:
            return False
        if progress[0] == 0:
            return context.override_next(argv.node_name, ["机密任务已完成"])

        combination = "＋".join(part for part in BURST_PARTS if part in required)
        extra = total - len(required)
        if extra:
            combination = "＋".join(filter(None, (combination, f"任意 {extra} 项")))
        logger.info(f"联盟机密任务：开始刷爆发（{combination}，最多 {minutes} 分钟）")
        attempts = 0
        while not context.tasker.stopping:
            if time.monotonic() >= deadline and not continue_on_timeout:
                break
            attempts += 1
            logger.info(f"联盟机密任务：第 {attempts} 场")
            task = context.clone().run_task("联盟机密任务入口分流", BURST_START_OVERRIDE)
            if not task or not task.status.succeeded:
                if context.tasker.stopping:
                    return False
                raise ValueError("未能进入机密比拼，请检查游戏画面")

            outcome = self._watch(context, required, total, deadline)
            if outcome == "stopped" or context.tasker.stopping:
                return False
            if outcome == "matched":
                logger.info(f"联盟机密任务：第 {attempts} 场达标，等待结算")
                return context.override_next(argv.node_name, ["等待机密任务比拼中"])
            if outcome == "limit" and continue_on_timeout:
                logger.info("联盟机密任务：已到刷取上限，打完本场后执行剩余次数")
                return context.override_next(argv.node_name, ["等待机密任务比拼中"])

            if not self._exit_battle(context, progress):
                return False
            if outcome == "unavailable":
                return context.override_next(argv.node_name, [])
            if outcome == "retry":
                logger.info(f"联盟机密任务：第 {attempts} 场未达标，重刷")

        if context.tasker.stopping:
            return False
        logger.info(f"联盟机密任务：已到刷取上限，尝试 {attempts} 场，请手动处理剩余次数")
        return context.override_next(argv.node_name, [])

    def _watch(self, context, required, total, deadline):
        battle_deadline = time.monotonic() + 120
        hits = set()
        seen = set()
        current_part = candidate = None
        quiet_frames = 2
        last_skill_at = None
        settlement_check_at = 0.0
        while not context.tasker.stopping:
            now = time.monotonic()
            if now >= deadline:
                return "limit"
            if now >= battle_deadline:
                raise ValueError("无法及时判定比拼结果，请检查游戏画面")
            image = self._capture(context)
            part = self._read_part(context, image)
            if part is None and now >= settlement_check_at:
                if self._settled(context, image):
                    raise ValueError("比拼已结算，未确认所选爆发组合，请检查本次机密分")
                settlement_check_at = now + 1

            effect = self._ocr(context, image, (110, .56, 500, .31), ["魅力爆发", "轻盈脚步"])
            texts = [entry.text for entry in effect.filtered_results] if effect.hit else []
            burst_visible = any("魅力爆发" in word for word in texts)
            new_burst = burst_visible and quiet_frames >= 2
            quiet_frames = 0 if burst_visible else quiet_frames + 1
            if part and part != current_part and (part == candidate or new_burst):
                if current_part and current_part not in hits:
                    logger.info(f"{current_part}：未爆发")
                if current_part in required - hits:
                    return "retry"
                current_part = part
                seen.add(part)
            candidate = part

            if new_burst and part and part not in hits:
                hits.add(part)
                logger.info(f"{part}：已爆发")
            if required <= hits and len(hits) >= total:
                return "matched"

            # 鞋袜是最后一个评分回合。等轻盈脚步的加分出现，给爆发提示留出显示时间。
            if current_part == "鞋袜":
                if last_skill_at is None and any("轻盈脚步" in word for word in texts):
                    last_skill_at = now
                if last_skill_at is not None and now - last_skill_at >= .6:
                    if current_part not in hits:
                        logger.info(f"{current_part}：未爆发")
                    missing = required - seen - hits
                    if missing:
                        parts = "、".join(part for part in BURST_PARTS if part in missing)
                        logger.info(f"联盟机密任务：本场未识别到必爆部位（{parts}），请检查搭配或更换组合")
                        return "unavailable"
                    if len(seen | hits) < total:
                        logger.info(f"联盟机密任务：本场评分部位不足 {total} 项，请检查搭配或更换组合")
                        return "unavailable"
                    return "retry"
            time.sleep(.1)
        return "stopped"

    def _exit_battle(self, context, previous_progress):
        deadline = time.monotonic() + 15
        clicked_confirm = False
        next_exit_click = 0.0
        while not context.tasker.stopping and time.monotonic() < deadline:
            image = self._capture(context)
            if self._ocr(context, image, (160, .05, 400, .105), ["^机密任务$"]).hit:
                progress = self._progress(context, image)
                if progress == previous_progress:
                    return True
                if progress is not None:
                    raise ValueError("退出后剩余次数发生变化，已停止刷取")
            elif not clicked_confirm and self._ocr(
                context, image, (70, .31, 580, .26),
                ["(?:确认|确定|是否|要).{0,4}退出", "退出(?:比拼|挑战|任务)", "放弃(?:比拼|挑战|任务)"],
            ).hit:
                button = self._ocr(context, image, (350, .43, 315, .22), ["^(确定|确认|退出|确认退出|确定退出)$"])
                if button.hit:
                    if not self._click(context, button.box):
                        return False
                    clicked_confirm = True
            elif not clicked_confirm and time.monotonic() >= next_exit_click:
                button = self._ocr(context, image, (0, .09, 135, .07), ["^退出$"])
                if button.hit:
                    if not self._click(context, button.box):
                        return False
                    next_exit_click = time.monotonic() + 1
            time.sleep(.2)
        if context.tasker.stopping:
            return False
        raise ValueError("未能确认退出及剩余次数，请检查游戏画面")

    def _wait_progress(self, context, seconds):
        deadline = time.monotonic() + seconds
        while not context.tasker.stopping and time.monotonic() < deadline:
            progress = self._progress(context, self._capture(context))
            if progress is not None:
                return progress
            time.sleep(.2)
        if context.tasker.stopping:
            return None
        raise ValueError("无法确认今日剩余机密次数")

    @staticmethod
    def _capture(context):
        controller = context.tasker.controller
        if not controller.post_screencap().wait().succeeded:
            raise ValueError("截图失败")
        image = controller.cached_image
        if image is None:
            raise ValueError("截图失败")
        return image

    @staticmethod
    def _click(context, box):
        if context.tasker.stopping:
            return False
        if not box or box[2] <= 0 or box[3] <= 0:
            raise ValueError("无法定位退出按钮")
        x, y, w, h = box
        result = context.run_action_direct("Click", JClick(target=(x + w // 2, y + h // 2)))
        if not result or not result.success:
            raise ValueError("点击退出失败")
        return True

    @staticmethod
    def _ocr(context, image, roi, expected):
        x, y, w, h = roi
        height = image.shape[0]
        result = context.run_recognition_direct(
            "OCR", JOCR(roi=(x, round(y * height), w, round(h * height)),
                        expected=expected, model=BURST_MODEL), image,
        )
        if result is None:
            raise ValueError("PP-OCRv6 识别失败")
        return result

    def _read_part(self, context, image):
        result = self._ocr(context, image, (10, .158, 210, .093), [BURST_PART_PATTERN])
        label = extract_ocr_text(result).removesuffix("评分")
        return next((part for part, labels in BURST_PARTS.items() if re.fullmatch(labels, label)), label or None)

    @staticmethod
    def _recognize_node(context, image, node):
        result = context.run_recognition(
            node, image, {node: {"recognition": {"param": {"model": BURST_MODEL}}}},
        )
        if result is None:
            raise ValueError("PP-OCRv6 识别失败")
        return result

    def _progress(self, context, image):
        result = self._recognize_node(context, image, PROGRESS_OCR_NODE)
        return parse_execution_progress(extract_ocr_text(result))

    def _settled(self, context, image):
        return self._recognize_node(context, image, "等待竞技场比拼结束").hit
