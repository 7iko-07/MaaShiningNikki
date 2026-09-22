"""启动时选择已保存的官服账号，并按需切换服务器。"""

import json
import re
import time

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.pipeline import JClick, JOCR, JSwipe
from utils import logger
from utils.arena_state import set_login_server, set_login_suffix


@AgentServer.custom_action("switch_account")
class SwitchAccount(CustomAction):
    CURRENT = [200, 590, 290, 47]
    ACCOUNTS = [200, 669, 290, 168]
    LOGIN = [270, 726, 180, 57]

    def _image(self, context):
        controller = context.tasker.controller
        controller.post_screencap().wait()
        image = controller.cached_image
        if image is None:
            raise ValueError("截图失败")
        return image.copy()

    def _ocr(self, context, image, roi, expected=None):
        result = context.run_recognition_direct(
            "OCR", JOCR(roi=roi, expected=expected or []), image
        )
        if not result or not result.hit:
            return []
        return list(result.filtered_results or []) or (
            [result.best_result] if result.best_result else []
        )

    @staticmethod
    def _suffix(text):
        text = re.sub(r"\s+", "", text)
        # 只接受完整的掩码手机号，避免把“上次登录”等数字当成尾号。
        match = re.fullmatch(r"1[0-9]{2}[*＊•·xX×]{4}([0-9]{4})", text)
        return match[1] if match else None

    def _accounts(self, context, image, roi):
        return [
            (self._suffix(item.text), item.box)
            for item in self._ocr(context, image, roi)
            if self._suffix(item.text) is not None
        ]

    def _click(self, context, box):
        result = context.run_action_direct("Click", JClick(target=list(box)))
        if not result or not result.success:
            raise ValueError("点击失败")

    def _wait_login(self, context, suffix=None):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            image = self._image(context)
            login = self._ocr(context, image, self.LOGIN, [r"^登\s*录$"])
            current = self._accounts(context, image, self.CURRENT)
            if login and current and (suffix is None or current[0][0] == suffix):
                return image, login[0].box, current[0][0]
            time.sleep(0.3)
        raise ValueError("15 秒内未确认登录框或目标账号尾号")

    @staticmethod
    def _unchanged(before, after):
        # 只比较列表内手机号，排除背景动画和“上次登录”的时间变化。
        before = before[669:837, 200:490]
        after = after[669:837, 200:490]
        if before.shape != after.shape or before.size == 0:
            return False
        diff = np.abs(before.astype(np.int16) - after.astype(np.int16))
        return np.mean(np.any(diff > 25, axis=2)) < 0.01

    def _select(self, context, suffix):
        self._click(context, [575, 620, 11, 16])
        time.sleep(0.6)
        before = None
        for count in range(11):
            image = self._image(context)
            matches = [box for tail, box in self._accounts(context, image, self.ACCOUNTS)
                       if tail == suffix]
            if len(matches) > 1:
                raise ValueError(f"尾号 {suffix} 匹配多个账号，无法确定目标")
            if matches:
                self._click(context, matches[0])
                return
            if before is not None and self._unchanged(before, image):
                break
            if count == 10:
                break
            before = image
            result = context.run_action_direct(
                "Swipe", JSwipe(begin=[336, 834], end=[336, 669],
                                duration=300, end_hold=1500)
            )
            if not result or not result.success:
                raise ValueError("账号列表上滑失败")
            time.sleep(0.6)
        raise ValueError(f"未找到尾号 {suffix} 的账号（列表不再变化或已上滑 10 次）")

    def run(self, context, argv):
        set_login_suffix(context, "")
        try:
            params = json.loads(argv.custom_action_param or "{}")
            suffix = params.get("suffix", "")
            if not isinstance(suffix, str) or not re.fullmatch(r"[0-9]{4}", suffix):
                raise ValueError("请填写 4 位手机尾号")
            image = self._image(context)
            # 已在登录框时不点击背景里的登出。
            if not self._ocr(context, image, self.LOGIN, [r"^登\s*录$"]):
                logout = self._ocr(context, image, [0, 548, 92, 80], [r"登\s*出"])
                if not logout:
                    raise ValueError("未找到登出按钮")
                self._click(context, logout[0].box)
            _, login_box, current = self._wait_login(context)
            if current != suffix:
                self._select(context, suffix)
                _, login_box, _ = self._wait_login(context, suffix)
            self._click(context, login_box)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                image = self._image(context)
                # 登录弹窗后方同样有“点击开始”，必须确认弹窗已消失。
                modal = self._ocr(context, image, [190, 590, 350, 270],
                                  [r"其他账号登录", r"1[0-9]{2}.*[0-9]{4}", r"登\s*录"])
                start = context.run_recognition("点击开始", image)
                if not modal and start and start.hit:
                    set_login_suffix(context, suffix)
                    logger.info(f"已登录尾号 {suffix} 的账号，继续服务器选择流程")
                    return True
                time.sleep(0.3)
            raise ValueError("登录失败：15 秒内未返回点击开始页面，请检查登录失效或验证码")
        except Exception as exc:
            logger.error(f"切换账号失败：{exc}")
            return False


@AgentServer.custom_action("switch_server")
class SwitchServer(SwitchAccount):
    SERVERS = ("千羽森林", "抖落繁星", "镜中圆舞", "童话香氛")
    CHANGE_SERVER = [427, 976, 104, 43]
    SERVER_TITLE = [276, 253, 176, 47]
    # 只取“全部服务器”的名称列，避开最近登录、区号和等级。
    SERVER_NAMES = [309, 574, 116, 251]

    def _record_current_server(self, context, image, expected=""):
        # 读取开始页面底部当前区服栏，不把“保持当前服务器”当作区服名。
        texts = [re.sub(r"\s+", "", item.text)
                 for item in self._ocr(context, image, [0, 930, 720, 130])]
        matches = [name for name in self.SERVERS if any(name in text for text in texts)]
        server = matches[0] if len(matches) == 1 else ""
        if expected and server != expected:
            server = ""
        set_login_server(context, server)
        if not server:
            logger.warning("未能确认当前服务器，竞技场本次不读写每周搭配记录")

    def run(self, context, argv):
        set_login_server(context, "")
        try:
            server = json.loads(argv.custom_action_param or "{}").get("server", "")
            if not server:
                try:
                    self._record_current_server(context, self._image(context))
                except Exception as exc:
                    logger.warning(f"当前服务器识别失败，继续登录但不记录竞技场搭配：{exc}")
                return True
            if server not in self.SERVERS:
                raise ValueError(f"不支持的服务器：{server}")

            deadline = time.monotonic() + 15
            next_click = 0
            attempts = 0
            while time.monotonic() < deadline:
                image = self._image(context)
                if self._ocr(context, image, self.SERVER_TITLE, [r"更换服务器"]):
                    logger.info("已确认更换服务器窗口打开，开始查找服务器")
                    break
                change = self._ocr(context, image, self.CHANGE_SERVER, [r"点击换区"])
                if change and time.monotonic() >= next_click:
                    # 登录刚结束时页面可能尚不能响应点击；固定点击文字中心，
                    # 必须等窗口标题出现才进入下一步，否则间隔重试。
                    self._click(context, [477, 996, 2, 2])
                    attempts += 1
                    logger.info(f"点击换区：第 {attempts} 次，等待窗口打开")
                    next_click = time.monotonic() + 1.5
                time.sleep(0.3)
            else:
                raise ValueError(f"15 秒内未打开更换服务器窗口（已点击 {attempts} 次）")

            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                image = self._image(context)
                # 按用户要求，名称任意一个字命中即点击首个匹配结果。
                matches = self._ocr(context, image, self.SERVER_NAMES,
                                    [f"[{re.escape(server)}]"])
                if matches:
                    self._click(context, matches[0].box)
                    break
                time.sleep(0.3)
            else:
                raise ValueError(f"15 秒内未找到服务器：{server}")

            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                image = self._image(context)
                # 弹窗不遮挡背景“点击开始”，还需确认底部换区栏已恢复。
                change = self._ocr(context, image, self.CHANGE_SERVER, [r"点击换区"])
                dialog = self._ocr(context, image, self.SERVER_TITLE, [r"更换服务器"])
                start = context.run_recognition("点击开始", image)
                if change and not dialog and start and start.hit:
                    self._record_current_server(context, image, server)
                    logger.info(f"已选择服务器 {server}，继续点击开始")
                    return True
                time.sleep(0.3)
            raise ValueError("选择服务器后 15 秒内未返回开始页面")
        except Exception as exc:
            logger.error(f"切换服务器失败：{exc}")
            return False
