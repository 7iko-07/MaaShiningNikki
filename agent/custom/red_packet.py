"""在任务识别、动作的间隙领取红包，并返回原页面。"""

from threading import Lock
import time

from maa.agent.agent_server import AgentServer
from maa.context import ContextEventSink
from maa.custom_action import CustomAction
from maa.pipeline import JClick
from utils import debug_exception, logger


ICON = "红包提示图标"
HEADER = "红包详情标题"
REWARD = "红包领取奖励层"
THANK = "红包感谢好友按钮"
DIALOG = "红包感谢对话框"
SEND = "红包发送感谢按钮"
EXIT = "红包退出按钮"
ENTRY = "红包处理入口"


class RedPacketRunner:
    def __init__(self, context):
        self.context = context
        self.controller = context.tasker.controller
        self.deadline = time.monotonic() + 30

    def frame(self):
        if self.context.tasker.stopping:
            raise InterruptedError
        if not self.controller.post_screencap().wait().succeeded:
            raise RuntimeError("截图失败")
        return self.controller.cached_image

    def hit(self, name, image):
        result = self.context.run_recognition(name, image)
        return result if result is not None and result.hit else None

    def wait(self, name, *, reward_closed=False):
        while not self.context.tasker.stopping and time.monotonic() < self.deadline:
            image = self.frame()
            if not reward_closed or self.hit(REWARD, image) is None:
                result = self.hit(name, image)
                if result is not None:
                    return result
            time.sleep(.1)
        if self.context.tasker.stopping:
            raise InterruptedError
        raise RuntimeError(f"等待{name}超时")

    def click(self, result):
        if self.context.tasker.stopping:
            raise InterruptedError
        box = result.box
        if box is None or box.w <= 0 or box.h <= 0:
            raise RuntimeError("未找到有效点击位置")
        action = self.context.run_action_direct(
            "Click", JClick(target=(box.x, box.y, box.w, box.h)),
        )
        if action is None or not action.success:
            raise RuntimeError("点击失败")

    def run(self):
        # 命中缓存帧后再次截图，避免点击已经消失或位置发生变化的提示。
        icon = self.hit(ICON, self.frame())
        if icon is None:
            return True
        logger.info("发现红包，正在领取")
        self.click(icon)
        self.wait(HEADER)
        self.wait(REWARD)
        # 用户确认：点击下方「感谢好友」的位置可以收起奖励层。
        self.click(self.wait(THANK))
        self.click(self.wait(THANK, reward_closed=True))
        self.wait(DIALOG)
        # 保留界面默认选中的感谢语。发送只点击一次，随后等待「退出」。
        self.click(self.wait(SEND))
        self.click(self.wait(EXIT))
        while not self.context.tasker.stopping and time.monotonic() < self.deadline:
            image = self.frame()
            if all(self.hit(name, image) is None for name in (HEADER, DIALOG)):
                logger.info("红包已领取并感谢好友，继续任务")
                return True
            time.sleep(.1)
        if self.context.tasker.stopping:
            raise InterruptedError
        raise RuntimeError("等待红包弹窗关闭超时")


@AgentServer.custom_action("handle_red_packet")
class RedPacketAction(CustomAction):
    def run(self, context, argv):
        try:
            return RedPacketRunner(context).run()
        except InterruptedError:
            return False
        except Exception as exc:
            debug_exception("红包处理异常")
            logger.error(f"红包处理失败：{exc}")
            return False


class RedPacketSink(ContextEventSink):
    def __init__(self):
        self._busy = set()
        self._checked_at = {}
        self._lock = Lock()

    def on_raw_notification(self, context, msg, details):
        if msg not in ("Node.NextList.Starting", "Node.RecognitionNode.Starting", "Node.Action.Starting"):
            return
        if details.get("name", "").startswith("红包"):
            return
        tasker = context.tasker
        key = getattr(tasker._handle, "value", tasker._handle)
        now = time.monotonic()
        with self._lock:
            if key in self._busy or now - self._checked_at.get(key, float("-inf")) < 1:
                return
            self._busy.add(key)
            self._checked_at[key] = now
        try:
            if tasker.stopping or context.get_node_data(ICON) is None:
                return
            image = tasker.controller.cached_image
            if image is None or not image.size:
                return
            result = context.run_recognition(ICON, image)
            if result is None or not result.hit:
                return
            handled = context.run_task(ENTRY)
            if handled is None or not handled.status.succeeded:
                if not tasker.stopping:
                    logger.error("红包未能处理完成，任务已停止，请检查当前画面")
                    tasker.post_stop()
        except Exception:
            debug_exception("红包监测异常")
            if not tasker.stopping:
                logger.error("红包监测异常，任务已停止，请查看日志")
                tasker.post_stop()
        finally:
            with self._lock:
                self._busy.remove(key)
                self._checked_at[key] = time.monotonic()


AgentServer.add_context_sink(RedPacketSink())
