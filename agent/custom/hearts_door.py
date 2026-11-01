import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.pipeline import JActionType, JClick

from utils import debug_exception, logger


FREE_NODE = "心之门免费按钮"
RESULT_NODE = "心之门点击分享"
READY_NODE = "心之门感应按钮就绪"
CLOSE_NODE = "点击谜之海-感应第一次-空白"
DISPLAY_NODE = "心之门品质展示"


@AgentServer.custom_action("hearts_door")
class HeartsDoorAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            free = context.get_node_data(FREE_NODE)["enabled"]
            mystery = context.get_node_data("点击谜之海")["enabled"]
            share = context.get_node_data(RESULT_NODE)["enabled"]
            # 临时阁名、分享识别及开关隔离在本轮，不留下跨任务状态。
            context = context.clone()
            if not context.override_pipeline({RESULT_NODE: {"enabled": True}}):
                raise RuntimeError("无法准备结果识别")
            completed = 0
            if free:
                for sea in ("绮海", "幻海"):
                    self._select_sea(context, sea)
                    self._wait_page(context)
                    result = self._recognize(context, FREE_NODE, self._capture(context))
                    if not result.hit:
                        logger.info(f"心之门：{sea}无免费次数，跳过")
                        continue
                    self._click_center(context, result.box)
                    self._finish_draw(context, share and completed == 0)
                    completed += 1
                    logger.info(f"心之门：{sea}免费感应成功，累计 {completed}/2 次")
            if mystery and completed < 2:
                self._select_sea(context, "谜海")
                for index in range(2 - completed):
                    ordinal = "第一次" if index == 0 else "第二次"
                    self._wait_hit(context, f"识别谜之海-感应{ordinal}")
                    node = f"点击谜之海-感应{ordinal}"
                    result = self._wait_hit(context, node)
                    self._click_center(context, result.box)
                    self._finish_draw(context, share and completed == 0)
                    completed += 1
                    logger.info(f"心之门：谜海感应成功，累计 {completed}/2 次")
            logger.info(f"心之门：本次成功感应 {completed}/2 次")
            return True
        except RuntimeError as exc:
            debug_exception("心之门执行失败")
            if not context.tasker.stopping:
                logger.error(f"心之门：{exc}")
            return False
        except Exception:
            debug_exception("心之门执行异常")
            logger.error("心之门：执行异常，请检查当前页面")
            return False

    def _select_sea(self, context, sea):
        node = "点击谜之海" if sea == "谜海" else "心之门阁名"
        if node == "心之门阁名":
            if not context.override_pipeline({node: {"recognition": {
                "type": "OCR", "param": {"expected": [sea]},
            }}}):
                raise RuntimeError(f"无法准备{sea}识别")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            result = self._recognize(context, node, self._capture(context))
            if result.hit:
                self._action(context, node, result.box)
                time.sleep(1)
                return
            self._action(context, "左划心之门的阁")
            time.sleep(0.5)
        raise RuntimeError(f"查找{sea}超时")

    def _finish_draw(self, context, share):
        self._wait_hit(context, RESULT_NODE)
        if share:
            self._check_stop(context)
            result = context.run_task(RESULT_NODE)
            if not result or not result.status.succeeded:
                raise RuntimeError("分享失败，已停止感应")
        self._wait_page(context, close_result=True)

    def _wait_page(self, context, close_result=False):
        deadline = time.monotonic() + 30
        closed = False
        while time.monotonic() < deadline:
            image = self._capture(context)
            popup = self._recognize(context, RESULT_NODE, image)
            if popup.hit:
                if close_result and not closed:
                    self._close_popup(context)
                    closed = True
            elif not close_result or not self._close_display(context, image):
                if self._recognize(context, READY_NODE, image).hit:
                    return
            time.sleep(0.2)
        raise RuntimeError("未能确认返回感应页面，已停止感应")

    def _wait_hit(self, context, node):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            image = self._capture(context)
            result = self._recognize(context, node, image)
            if result.hit:
                return result
            if node == RESULT_NODE:
                insufficient = self._recognize(context, "点击体力用尽确定", image)
                if insufficient.hit:
                    self._action(context, "点击体力用尽确定", insufficient.box)
                    raise RuntimeError("资源不足，已停止感应")
            time.sleep(0.2)
        raise RuntimeError("未能确认感应画面，已停止感应")

    def _close_display(self, context, image):
        display = self._recognize(context, DISPLAY_NODE, image)
        if not display.hit:
            return False
        logger.debug("心之门：识别到品质展示，点击原关闭位置")
        self._close_popup(context)
        return True

    def _close_popup(self, context):
        target = context.get_node_data(CLOSE_NODE)["action"]["param"]["custom_action_param"]["target"]
        self._click_center(context, target)

    @staticmethod
    def _check_stop(context):
        if context.tasker.stopping:
            raise RuntimeError("任务已停止")

    def _capture(self, context):
        self._check_stop(context)
        controller = context.tasker.controller
        if not controller.post_screencap().wait().succeeded:
            raise RuntimeError("截图失败")
        self._check_stop(context)
        return controller.cached_image

    @staticmethod
    def _recognize(context, node, image):
        result = context.run_recognition(node, image)
        if result is None:
            raise RuntimeError("无法识别当前画面")
        return result

    def _action(self, context, node, box=(0, 0, 0, 0)):
        self._check_stop(context)
        if node != "左划心之门的阁" and (box[2] <= 0 or box[3] <= 0):
            raise RuntimeError("无法定位点击目标")
        result = context.run_action(node, box=box)
        if not result or not result.success:
            raise RuntimeError("操作失败，已停止感应")

    def _click_center(self, context, box):
        self._check_stop(context)
        x, y, w, h = box
        if w <= 0 or h <= 0:
            raise RuntimeError("无法定位点击目标")
        result = context.run_action_direct(
            JActionType.Click, JClick(target=(x + w // 2, y + h // 2)),
        )
        if not result or not result.success:
            raise RuntimeError("点击失败，已停止感应")
