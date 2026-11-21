"""搭配评选赛时段分流与限时投稿，不保存跨账号的参赛状态。"""

import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from utils import debug_exception, logger
from utils.arena_state import BEIJING, now_beijing


def contest_period(now=None):
    now = (now or now_beijing()).astimezone(BEIJING)
    if (now.weekday() == 4 and now.hour >= 21) or (now.weekday() == 5 and now.hour < 5):
        return "settlement"
    if now.weekday() in (5, 6) or (now.weekday() == 0 and now.hour < 5):
        return "submission"
    return "judging"


@AgentServer.custom_action("outfit_contest_guard")
class OutfitContestGuard(CustomAction):
    def run(self, context, argv):
        if context.tasker.stopping:
            return False
        period = contest_period()
        enabled = context.get_node_data(argv.node_name)["attach"]["auto_submit"]
        if period == "settlement":
            logger.info("搭配评选赛：结算中，跳过任务")
            return context.override_next(argv.node_name, [])
        if period == "submission" and not enabled:
            logger.info("搭配评选赛：自动参赛未开启，跳过参赛期")
            return context.override_next(argv.node_name, [])
        # 每次进入均恢复入口和导航后继，避免上次跳过或其他账号的覆盖残留。
        prepare = "搭配评选赛任务链准备"
        next_nodes = ["搭配评选赛自动参赛"] if period == "submission" else ["点击我当评委"]
        action = context.get_node_data(prepare)["action"]
        action["param"]["custom_action_param"]["ignore_background_download"] = period == "judging"
        return (context.override_pipeline({prepare: {"action": action}})
                and context.override_next(prepare, next_nodes)
                and context.override_next(argv.node_name, [prepare]))


@AgentServer.custom_action("outfit_contest_submit")
class OutfitContestSubmit(CustomAction):
    def run(self, context, argv):
        try:
            name, result = self._wait(context, ["搭配评选赛已参赛", "搭配评选赛点击我要参赛"])
            if name == "搭配评选赛已参赛":
                logger.info("搭配评选赛：本期已参赛，跳过")
                return True
            logger.info("搭配评选赛：开始搭配参赛")
            _, result = self._transition(context, name, result.box, ["搭配评选赛换好了"])
            _, result = self._transition(context, "搭配评选赛上划分类", result.box, ["搭配评选赛点击套装"])
            _, result = self._transition(context, "搭配评选赛点击套装", result.box, ["搭配评选赛选择首套"])
            _, result = self._transition(context, "搭配评选赛选择首套", result.box, ["搭配评选赛换好了"])
            _, result = self._transition(context, "搭配评选赛换好了", result.box, ["搭配评选赛投稿参赛"])
            _, result = self._transition(context, "搭配评选赛投稿参赛", result.box, ["搭配评选赛拍摄确认上传"])
            self._transition(context, "搭配评选赛拍摄确认上传", result.box, ["搭配评选赛上传封面"])
            for checkbox in ("搭配评选赛保存模板已勾选", "搭配评选赛上传定格已勾选"):
                self._uncheck(context, checkbox)
            _, result = self._wait(context, ["搭配评选赛封面确认上传"])
            _, result = self._transition(context, "搭配评选赛封面确认上传", result.box, ["搭配评选赛不公开提示确定"])
            _, result = self._transition(context, "搭配评选赛不公开提示确定", result.box, ["搭配评选赛分享页面"])
            logger.info("搭配评选赛：投稿上传完成")
            self._transition(context, "搭配评选赛投稿后返回", result.box, ["搭配评选赛已参赛"])
            return True
        except RuntimeError as exc:
            debug_exception("搭配评选赛参赛失败")
            if not context.tasker.stopping:
                logger.error(f"搭配评选赛：{exc}")
            return False
        except Exception:
            debug_exception("搭配评选赛参赛异常")
            logger.error("搭配评选赛：参赛执行异常")
            return False

    @staticmethod
    def _check(context, deadline):
        if context.tasker.stopping:
            raise RuntimeError("任务已停止")
        if time.monotonic() >= deadline:
            raise RuntimeError("页面切换或上传超时，未能确认结果")

    def _capture(self, context, deadline):
        self._check(context, deadline)
        controller = context.tasker.controller
        if not controller.post_screencap().wait().succeeded or controller.cached_image is None:
            raise RuntimeError("截图失败")
        self._check(context, deadline)
        return controller.cached_image

    @staticmethod
    def _recognize(context, node, image):
        result = context.run_recognition(node, image)
        if result is None:
            raise RuntimeError("页面识别执行失败")
        return result

    def _wait(self, context, nodes, deadline=None, unchecked=None):
        deadline = deadline if deadline is not None else time.monotonic() + 30
        while True:
            image = self._capture(context, deadline)
            loading = self._recognize(context, "在加载页面", image)
            if not loading.hit:
                for node in nodes:
                    result = self._recognize(context, node, image)
                    self._check(context, deadline)
                    if result.hit and (unchecked is None or not self._recognize(context, unchecked, image).hit):
                        self._check(context, deadline)
                        return node, result
            time.sleep(min(.2, max(0, deadline - time.monotonic())))

    def _action(self, context, node, box, deadline):
        self._check(context, deadline)
        data = context.get_node_data(node)
        override = None
        action = data["action"]
        if action["type"] == "Custom":
            # 点击只执行一次；画面变化之后还必须确认真正的后继页面。
            params = dict(action["param"]["custom_action_param"])
            params["timeout"] = max(1, int((deadline - time.monotonic()) * 1000) - 300)
            override = {node: {"action": {"type": "Custom", "param": {
                "custom_action": "click_and_wait_change", "custom_action_param": params,
            }}}}
        if action["type"] != "Swipe" and (box[2] <= 0 or box[3] <= 0):
            raise RuntimeError("无法定位操作目标")
        if override is not None:
            # 本地 SDK 的 Custom 回调需要有效识别记录；叶节点子任务会先识别，且没有后继。
            override[node].update(pre_delay=0, post_delay=0)
            result = context.run_task(node, pipeline_override=override)
            succeeded = result is not None and result.status.succeeded
        else:
            result = context.run_action(node, box=box)
            succeeded = result is not None and result.success
        if not succeeded:
            raise RuntimeError("操作失败，任务已停止")
        self._check(context, deadline)

    def _transition(self, context, node, box, ready_nodes):
        deadline = time.monotonic() + 30
        self._action(context, node, box, deadline)
        return self._wait(context, ready_nodes, deadline)

    def _uncheck(self, context, node):
        deadline = time.monotonic() + 30
        self._wait(context, ["搭配评选赛上传封面"], deadline)
        image = self._capture(context, deadline)
        checked = self._recognize(context, node, image)
        if checked.hit:
            self._action(context, node, checked.box, deadline)
        self._wait(context, ["搭配评选赛上传封面"], deadline, unchecked=node)
