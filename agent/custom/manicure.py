"""美甲采购、投稿、评价与话题任务分流。"""

import json
import sqlite3
import time
from datetime import timedelta

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.pipeline import JWaitFreezes
from utils import logger
from utils import arena_state as state

from .click_and_wait_change import ClickAndWaitChangeAction


FREE = "美甲采购免费领取"
EXCHANGE = "美甲采购时尚礼券兑换"
READY = "美甲采购设计专区就绪"
SUBMISSION = "美甲投稿"
SUBMISSION_DISPATCH = "美甲投稿与点赞分流"
OPEN_SUBMISSION = "美甲投稿打开我的投稿"


@AgentServer.custom_action("manicure_review_click")
class ManicureReviewClickAction(ClickAndWaitChangeAction):
    def _additional_image_change(self, context, before, after):
        previous = self._name_masks(context, before)
        if previous is None:
            return False
        current = self._name_masks(context, after)
        if current is None:
            return False

        for label, old, new in zip(("昵称", "下排文字"), previous, current):
            if min(np.count_nonzero(old), np.count_nonzero(new)) < 12:
                continue
            difference = np.count_nonzero(old ^ new) / np.count_nonzero(old | new)
            if difference >= 0.20:
                logger.debug(f"美甲评价：{label}字形已变化，差异比例 {difference:.3f}")
                return True
        return False

    @staticmethod
    def _name_masks(context, image):
        report = context.run_recognition("美甲评价举报按钮", image)
        if not report or not report.hit or report.box is None:
            return None

        # 两行各自保留固定的右端；上排左边随着举报按钮位置变化。
        nickname = image[30:65, 110:560].astype(np.int16)
        lower = image[68:98, 250:591].astype(np.int16)
        nickname_ink = np.max(np.abs(nickname - (102, 98, 169)), axis=2) <= 12
        nickname_ink[:, :report.box.x + report.box.w + 4 - 110] = False
        lower_ink = np.max(np.abs(lower - (146, 146, 162)), axis=2) <= 12
        return nickname_ink, lower_ink


def next_manicure_refresh(item, now):
    if item == FREE:
        return now + timedelta(hours=30)
    monday = (now - timedelta(days=now.weekday())).replace(hour=5, minute=0, second=0, microsecond=0)
    return monday if now < monday else monday + timedelta(days=7)


@AgentServer.custom_action("manicure_purchase")
class ManicurePurchaseAction(CustomAction):
    def run(self, context, argv):
        account = state.take_account(context, "manicure")
        if context.get_node_data(OPEN_SUBMISSION).get("enabled", False):
            # 采购和投稿属于同一次美甲任务，沿用启动任务确认的账号。
            config = context.get_node_data(SUBMISSION_DISPATCH).get("attach", {})
            if not context.override_pipeline({SUBMISSION_DISPATCH: {"attach": {**config, "account": account}}}):
                return False
        if not context.get_node_data(argv.node_name).get("attach", {}).get("enabled", True):
            return True

        store = state.StateStore()
        items = []
        try:
            now = state.now_beijing()
            for item in (FREE, EXCHANGE):
                config = context.get_node_data(item).get("attach", {})
                if not config.get("selected"):
                    continue
                if config.get("periodic", True) and account and not store.manicure_available(account, item, now):
                    continue
                items.append(item)
        except (OSError, sqlite3.Error, ValueError) as exc:
            logger.debug(f"美甲采购记录读取失败：{exc}")
            logger.error("美甲：无法读取采购记录")
            return False

        if not items:
            return True
        if account is None:
            logger.warning("美甲：账号未确认，本次按游戏状态采购")
        if not self._task(context, "美甲采购前往采购页"):
            return False

        for item in items:
            state_found = self._wait_for_item(context, item)
            if state_found is None:
                return False
            if not state_found:
                continue
            # 采购子链只有在确认「恭喜获得」后才成功，先保存再关闭弹窗。
            if not self._task(context, item):
                return False
            now = state.now_beijing()
            if account:
                try:
                    store.mark_manicure_claimed(account, item, next_manicure_refresh(item, now))
                except (OSError, sqlite3.Error) as exc:
                    logger.debug(f"美甲采购记录保存失败：{exc}")
                    logger.warning("美甲：已获得缤纷星券，采购记录保存失败")
            logger.info("美甲：已免费领取缤纷星券" if item == FREE else "美甲：已兑换缤纷星券")
            if not self._task(context, "美甲采购关闭弹窗", {
                "美甲采购关闭弹窗": {"repeat": 3 if item == EXCHANGE else 1}
            }):
                return False

        return self._task(context, "美甲采购返回工作间")

    @staticmethod
    def _task(context, node, override=None):
        if context.tasker.stopping:
            return False
        result = context.run_task(node, pipeline_override=override)
        if result and result.status.succeeded:
            return True
        logger.debug(f"美甲采购子流程失败：{node}")
        if not context.tasker.stopping:
            logger.error("美甲：采购流程失败，请检查当前页面")
        return False

    @staticmethod
    def _wait_for_item(context, item):
        controller = context.tasker.controller
        deadline = time.monotonic() + 8
        absent = False
        unavailable = "美甲采购免费冷却" if item == FREE else "美甲采购本周限购"
        while not context.tasker.stopping and time.monotonic() < deadline:
            if not controller.post_screencap().wait().succeeded:
                logger.error("美甲：采购截图失败")
                return None
            image = controller.cached_image
            page = context.run_recognition(READY, image)
            if page is None:
                break
            if page.hit:
                offer = context.run_recognition(item, image)
                other = context.run_recognition(unavailable, image)
                if offer is None or other is None:
                    break
                if offer.hit:
                    return True
                unavailable_now = other.hit if item == FREE else not other.hit
                if unavailable_now and absent:
                    logger.info("美甲：免费领取冷却中" if item == FREE else "美甲：本周无可兑换的缤纷星券")
                    return False
                absent = unavailable_now
            else:
                absent = False
            time.sleep(0.25)
        if not context.tasker.stopping:
            logger.error("美甲：无法确认缤纷星券采购状态")
        return None


@AgentServer.custom_action("manicure_submission_dispatch")
class ManicureSubmissionDispatchAction(CustomAction):
    def run(self, context, argv):
        likes = context.get_node_data("点击浏览点赞").get("enabled", True)
        submission = context.get_node_data(OPEN_SUBMISSION).get("enabled", False)
        next_nodes = ["点击此刻投稿"] if likes or submission else []
        if submission:
            config = context.get_node_data(argv.node_name).get("attach", {})
            account = config.get("account")
            try:
                available = not config.get("periodic", True) or not account or state.StateStore().manicure_available(
                    account, SUBMISSION, state.now_beijing()
                )
            except (OSError, sqlite3.Error, ValueError) as exc:
                logger.debug(f"美甲投稿记录读取失败：{exc}")
                logger.error("美甲投稿：无法读取投稿记录")
                return False

            if available:
                result = context.run_task("点击此刻投稿", pipeline_override={
                    "点击此刻投稿": {
                        "next": ["美甲周期奖励确认", OPEN_SUBMISSION],
                        "timeout": 15000,
                    },
                    "美甲周期奖励确认": {"next": [OPEN_SUBMISSION], "timeout": 15000},
                })
                if not result or not result.status.succeeded or context.tasker.stopping:
                    if not context.tasker.stopping:
                        logger.error("美甲投稿：未能完成投稿流程，请检查当前页面")
                    return False
                next_nodes = ["点击浏览点赞", "此刻投稿-我的投稿"] if likes else ["此刻投稿-我的投稿"]
            else:
                logger.info("美甲投稿：本周已投稿")

        return context.override_next(argv.node_name, next_nodes)


@AgentServer.custom_action("manicure_submission_record")
class ManicureSubmissionRecordAction(CustomAction):
    def run(self, context, argv):
        logger.info("美甲投稿：上传成功")
        account = context.get_node_data(SUBMISSION_DISPATCH).get("attach", {}).get("account")
        if account:
            try:
                now = state.now_beijing()
                state.StateStore().mark_manicure_claimed(account, SUBMISSION, next_manicure_refresh(SUBMISSION, now))
            except (OSError, sqlite3.Error) as exc:
                logger.debug(f"美甲投稿记录保存失败：{exc}")
                logger.warning("美甲投稿：已上传，投稿记录保存失败")
        return True


@AgentServer.custom_action("manicure_topic_dispatch")
class ManicureTopicDispatchAction(CustomAction):
    def run(self, context, argv):
        params = json.loads(argv.custom_action_param or "{}") or {}
        deadline = time.monotonic() + params.get("timeout", 10000) / 1000
        controller = context.tasker.controller
        no_red = 0
        # Every invocation restores its own successor; a prior skip cannot leak.
        if not context.override_next(argv.node_name, ["点击话题任务"]):
            return False
        while not context.tasker.stopping and time.monotonic() < deadline:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms < 200:
                break
            if not context.wait_freezes(wait_freezes_param=JWaitFreezes(
                time=200, target=(0, 1133, 720, 147), rate_limit=100, timeout=remaining_ms,
            )):
                break
            if context.tasker.stopping or time.monotonic() >= deadline:
                break
            if not controller.post_screencap().wait().status.succeeded:
                return False
            image = controller.cached_image
            loading = context.run_recognition("在加载页面", image)
            popup = context.run_recognition("美甲周期奖励确认", image)
            page = context.run_recognition("美甲话题页面就绪", image)
            if (loading is not None and not loading.hit and popup is not None and not popup.hit
                    and page is not None and page.hit):
                red = context.run_recognition("点击话题任务", image)
                if context.tasker.stopping or time.monotonic() >= deadline:
                    break
                if red is None:
                    return False
                if red.hit:
                    return True
                no_red += 1
                if no_red >= 2:
                    logger.info("美甲话题任务无红点，跳过领奖")
                    return context.override_next(argv.node_name, [])
            else:
                no_red = 0
            time.sleep(min(.2, max(0, deadline - time.monotonic())))
        logger.error("美甲话题任务：页面未就绪或等待超时，停止任务")
        return False
