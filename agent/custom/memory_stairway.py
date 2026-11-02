"""心阶结算保护、弹窗确认及搭配页面推进。"""

import json
import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.pipeline import JClick
from utils import logger
from utils.arena_state import BEIJING, now_beijing
from utils.logger import debug_exception


def in_settlement(now=None):
    now = (now or now_beijing()).astimezone(BEIJING)
    return (now.weekday() == 6 and now.hour >= 23) or (now.weekday() == 0 and now.hour < 5)


def stop_for_settlement(context, node):
    logger.info("心阶结算中（北京时间周日23:00至周一05:00），跳过任务")
    return context.override_next(node, [])


@AgentServer.custom_action("memory_stairway_guard")
class MemoryStairwayGuardAction(CustomAction):
    def run(self, context, argv):
        if context.tasker.stopping:
            return False
        if in_settlement():
            return stop_for_settlement(context, argv.node_name)
        return context.override_next(argv.node_name, ["心阶任务链准备"])


@AgentServer.custom_action("memory_stairway_challenge")
class MemoryStairwayChallengeAction(CustomAction):
    NEXT_NODES = {
        "点击心阶-开始挑战": "点击境遇1",
        "点击境遇1-开始挑战": "点击境遇2",
        "点击境遇2-开始挑战": "stop",
    }

    def run(self, context, argv):
        if context.tasker.stopping:
            return False
        if in_settlement():
            return stop_for_settlement(context, argv.node_name)
        if not context.override_next(argv.node_name, [self.NEXT_NODES[argv.node_name]]):
            return False
        deadline = time.monotonic() + 30
        wait_calculation = argv.node_name == "点击心阶-开始挑战"
        toggled = False
        clear_frames = 0
        controller = context.tasker.controller
        try:
            while not context.tasker.stopping and time.monotonic() < deadline:
                if in_settlement():
                    return stop_for_settlement(context, argv.node_name)
                if not controller.post_screencap().wait().status.succeeded:
                    logger.error("心阶挑战：截图失败")
                    return False
                image = controller.cached_image
                results = {}
                for node in ("在加载页面", "心阶快速战斗选项", argv.node_name, "心阶快速战斗已开启"):
                    result = context.run_recognition(node, image)
                    if result is None:
                        logger.debug(f"心阶挑战：识别未返回结果：{node}")
                        logger.error("心阶：无法确认挑战页面")
                        return False
                    results[node] = result
                    if context.tasker.stopping:
                        return False
                    if time.monotonic() >= deadline:
                        logger.error("心阶：等待挑战页面超过30秒，任务已停止")
                        return False
                ready = (not results["在加载页面"].hit and results["心阶快速战斗选项"].hit
                         and results[argv.node_name].hit)
                if not ready or not results["心阶快速战斗已开启"].hit:
                    clear_frames = 0
                    if ready and not toggled:
                        if in_settlement():
                            return stop_for_settlement(context, argv.node_name)
                        # 两种挑战面板的圆圈均在这个位置；只点击一次，再截图确认红色。
                        action = context.run_action_direct("Click", JClick(target=(96, 990, 1, 1)))
                        if action is None or not action.success:
                            logger.error("心阶：开启快速战斗失败")
                            return False
                        toggled = True
                else:
                    if wait_calculation:
                        calculating = context.run_recognition("心阶搭配之力计算提示", image)
                        if calculating is None:
                            logger.error("心阶：无法识别搭配之力计算提示")
                            return False
                        clear_frames = 0 if calculating.hit else clear_frames + 1
                    if not wait_calculation or clear_frames >= 2:
                        if context.tasker.stopping:
                            return False
                        if time.monotonic() >= deadline:
                            logger.error("心阶：等待搭配计算超过30秒，任务已停止")
                            return False
                        if in_settlement():
                            return stop_for_settlement(context, argv.node_name)
                        box = results[argv.node_name].box
                        x, y, w, h = box if isinstance(box, (list, tuple)) else (box.x, box.y, box.w, box.h)
                        if w <= 0 or h <= 0:
                            logger.error("心阶：无法定位开始挑战按钮")
                            return False
                        # 结果等待期限从挑战点击开始累计；不因新弹窗或下一次点击而延长。
                        result_deadline = time.monotonic() + 30
                        action = context.run_action_direct(
                            "Click", JClick(target=(int(x + w / 2), int(y + h / 2), 1, 1)))
                        if action is None or not action.success:
                            logger.error("心阶：开始挑战失败")
                            return False
                        return MemoryStairwayPopupsAction().clear(
                            context, {"after_battle": True}, deadline=result_deadline)
                time.sleep(min(.5, max(0, deadline - time.monotonic())))
        except Exception as exc:
            debug_exception(f"心阶挑战失败：{exc}")
            logger.error("心阶：挑战失败")
            return False
        if not context.tasker.stopping:
            logger.error("心阶：30秒内未确认快速战斗开启或搭配计算完成，任务已停止")
        return False


@AgentServer.custom_action("memory_stairway_next_step")
class MemoryStairwayNextStepAction(CustomAction):
    def run(self, context, argv):
        deadline = time.monotonic() + 10
        next_click = 0.0
        clicks = 0
        try:
            params = json.loads(argv.custom_action_param or "{}")
            ready_node = params["ready_node"]
            controller = context.tasker.controller
            while not context.tasker.stopping and time.monotonic() < deadline:
                if not controller.post_screencap().wait().status.succeeded:
                    logger.error("心阶下一步：截图失败")
                    return False
                image = controller.cached_image
                ready = context.run_recognition(ready_node, image)
                if ready is None:
                    logger.debug(f"心阶下一步：识别未返回结果：{ready_node}")
                    logger.error("心阶：无法确认装备印象页面")
                    return False
                if context.tasker.stopping or time.monotonic() >= deadline:
                    break
                if ready.hit:
                    logger.debug(f"心阶下一步：已确认装备印象，点击{clicks}次")
                    return True

                loading = context.run_recognition("在加载页面", image)
                if loading is None:
                    logger.error("心阶下一步：加载状态识别失败")
                    return False
                if not loading.hit and time.monotonic() >= next_click:
                    button = context.run_recognition("心阶搭配下一步按钮", image)
                    if button is None:
                        logger.error("心阶下一步：按钮识别失败")
                        return False
                    if context.tasker.stopping or time.monotonic() >= deadline:
                        break
                    if button.hit:
                        box = button.box
                        if isinstance(box, (list, tuple)) and len(box) == 4:
                            x, y, w, h = box
                        else:
                            x, y, w, h = box.x, box.y, box.w, box.h
                        if w <= 0 or h <= 0:
                            logger.debug("心阶下一步：按钮识别框无效")
                            logger.error("心阶：无法定位下一步按钮")
                            return False
                        action = context.run_action_direct(
                            "Click", JClick(target=(int(x + w / 2), int(y + h / 2), 1, 1)))
                        if action is None or not action.success:
                            logger.error("心阶下一步：按钮点击失败")
                            return False
                        clicks += 1
                        next_click = time.monotonic() + 1
                time.sleep(min(.5, max(0, deadline - time.monotonic())))
        except Exception as exc:
            logger.debug(f"心阶下一步失败：{exc}")
            logger.error("心阶：前往装备印象页面失败")
            return False
        logger.debug(f"心阶下一步未完成：任务停止或10秒期限已到，未确认装备印象，点击{clicks}次")
        logger.error("心阶：未能及时进入装备印象页面，任务已停止")
        return False


@AgentServer.custom_action("memory_stairway_popups")
class MemoryStairwayPopupsAction(CustomAction):
    def run(self, context, argv):
        params = json.loads(argv.custom_action_param or "{}") or {}
        return self.clear(context, params)

    def clear(self, context, params, *, deadline=None):
        after_quick_challenge = params.get("after_quick_challenge", False)
        after_battle = params.get("after_battle", False)
        popup_nodes = ("心阶部件制作提示", "心阶快速挑战奖励弹窗") if after_quick_challenge else (
            "心阶新赛季开始弹窗", "心阶我知道了弹窗")
        if after_battle:
            popup_nodes = ("心阶挑战奖励弹窗", "心阶战斗评分弹窗", "心阶部件制作提示")
        nodes = ("导航在心阶页面", "在加载页面", *popup_nodes)
        if after_quick_challenge:
            nodes += ("点击真我之境",)
        timeout = 30 if after_battle else params.get("timeout", 15000) / 1000
        if deadline is None:
            deadline = time.monotonic() + timeout
        next_click = 0.0
        clear_frames = 0
        clicks = 0
        seen_result = False
        controller = context.tasker.controller
        last_state = "尚未确认心阶标题"
        try:
            while not context.tasker.stopping and time.monotonic() < deadline:
                if not controller.post_screencap().wait().status.succeeded:
                    logger.error("心阶弹窗处理：截图失败")
                    return False
                image = controller.cached_image
                hits = {}
                for node in nodes:
                    if context.tasker.stopping or time.monotonic() >= deadline:
                        break
                    result = context.run_recognition(node, image)
                    if result is None:
                        logger.debug(f"心阶弹窗处理：识别未返回结果：{node}")
                        logger.error("心阶：无法识别当前弹窗")
                        return False
                    hits[node] = result.hit
                if context.tasker.stopping or time.monotonic() >= deadline:
                    break
                if not hits["导航在心阶页面"] or hits["在加载页面"]:
                    clear_frames = 0
                    last_state = "心阶标题未确认或仍在加载"
                elif any(hits[node] for node in popup_nodes):
                    seen_result = True
                    clear_frames = 0
                    last_state = "仍有弹窗：" + "、".join(node for node in popup_nodes if hits[node])
                    if time.monotonic() >= next_click:
                        # User-verified blank region [603, 42, 21, 12]; never click background controls.
                        action = context.run_action_direct("Click", JClick(target=(613, 48, 1, 1)))
                        if action is None or not action.success:
                            logger.error("心阶弹窗处理：空白点击失败")
                            return False
                        clicks += 1
                        next_click = time.monotonic() + 1
                elif after_quick_challenge and not hits["点击真我之境"]:
                    clear_frames = 0
                    last_state = "尚未确认真我之境入口"
                elif after_battle and not seen_result:
                    clear_frames = 0
                    last_state = "尚未出现挑战结果弹窗"
                else:
                    clear_frames += 1
                    if clear_frames >= 2:
                        logger.debug(f"心阶弹窗检查完成，空白点击{clicks}次")
                        return True
                time.sleep(min(.5 if after_battle else .25, max(0, deadline - time.monotonic())))
        except Exception as exc:
            logger.debug(f"心阶弹窗处理失败：{exc}")
            logger.error("心阶：处理弹窗时发生异常")
            return False
        logger.debug(f"心阶弹窗处理未完成：任务停止或{timeout:g}秒期限已到，{last_state}，空白点击{clicks}次")
        logger.error("心阶：未能及时关闭弹窗，任务已停止")
        return False
