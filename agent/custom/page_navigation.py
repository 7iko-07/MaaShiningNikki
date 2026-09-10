import json
import time
from collections import deque

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JClick
from utils import logger


DEFAULT_PAGES = {
    "独自": ["导航在独自页面"],
    "主页面": ["是否是主页面"],
    "设计中心": ["导航在设计中心页面"],
    "美甲": ["导航在美甲页面"],
    "制衣": ["导航在制衣页面"],
    "情报屋": ["导航在情报屋页面"],
    "结伴": ["导航在结伴页面"],
    "联盟": ["导航在联盟页面"],
    "回家": ["导航在回家页面"],
    "邮件": ["导航在邮件页面"],
    "好友": ["导航在好友页面"],
    "福利": ["导航在福利页面"],
    "任务": ["导航在任务页面"],
    "竞技场": ["导航在竞技场页面"],
    "搭配评选赛": ["导航在搭配评选赛页面"],
    "美甲我的店铺": ["导航在美甲我的店铺页面"],
    "美甲评价": ["导航在美甲评价页面"],
    "此刻投稿": ["导航在美甲此刻投稿页面"],
    "商城": ["导航在商城页面"],
    "心阶": ["导航在心阶页面"],
    "心之门": ["导航在心之门页面"],
    "时空回廊": ["导航在时空回廊页面"],
    "主线故事": ["导航在主线故事页面"]
}

DEFAULT_ROUTES = [
    {"from": "主页面", "to": "设计中心", "tasks": ["导航确保底部菜单打开", "导航点击设计中心入口"]},
    {"from": "主页面", "to": "回家", "tasks": ["导航确保底部菜单打开", "导航点击回家入口"]},
    {"from": "设计中心", "to": "美甲", "tasks": ["导航点击美甲"]},
    {"from": "设计中心", "to": "制衣", "tasks": ["导航点击制衣引导"]},
    {"from": "设计中心", "to": "情报屋", "tasks": ["导航点击情报屋"]},
    {"from": "主页面", "to": "结伴", "tasks": ["导航确保底部菜单打开", "导航点击开始旅程入口", "导航点击结伴"]},
    {"from": "主页面", "to": "联盟", "tasks": ["导航确保侧边菜单打开", "导航点击联盟入口"]},
    {"from": "主页面", "to": "邮件", "tasks": ["导航确保侧边菜单打开", "导航点击邮件入口"]},
    {"from": "主页面", "to": "好友", "tasks": ["导航确保侧边菜单打开", "导航点击好友入口"]},
    {"from": "主页面", "to": "福利", "tasks": ["导航确保侧边菜单打开", "导航点击福利入口"]},
    {"from": "主页面", "to": "任务", "tasks": ["导航确保侧边菜单打开", "导航点击任务入口"]},
    {"from": "主页面", "to": "商城", "tasks": ["导航确保侧边菜单打开", "导航点击商城入口"]},
    {"from": "主页面", "to": "独自", "tasks": ["导航确保底部菜单打开", "导航点击开始旅程入口", "导航确保独自页面"]},
    {"from": "独自", "to": "结伴", "tasks": ["导航点击结伴"]},
    {"from": "结伴", "to": "独自", "tasks": ["导航点击独自"]},
    {"from": "独自", "to": "竞技场", "tasks": ["导航点击钻石竞技场"]},
    {"from": "独自", "to": "主线故事", "tasks": ["导航点击主线故事"]},
    {"from": "独自", "to": "时空回廊", "tasks": ["导航点击时空回廊"]},
    {"from": "独自", "to": "心阶", "tasks": ["导航点击心阶"]},
    {"from": "竞技场", "to": "独自", "tasks": ["导航点击返回"]},
    {"from": "主线故事", "to": "独自", "tasks": ["导航主线返回独自"]},
    {"from": "时空回廊", "to": "独自", "tasks": ["导航点击返回"]},
    {"from": "心阶", "to": "独自", "tasks": ["导航点击返回"]},
    {"from": "搭配评选赛", "to": "独自", "tasks": ["导航点击返回"]},
    {"from": "主页面", "to": "心之门", "tasks": ["导航点击心之门入口"]},
    {
        "from": "独自",
        "to": "搭配评选赛",
        "tasks": ["导航点击搭配评选赛"],
    },
    {"from": "美甲", "to": "美甲我的店铺", "tasks": ["导航点击我的店铺"]},
    {"from": "美甲我的店铺", "to": "美甲", "tasks": ["导航点击返回"]},
    {"from": "美甲", "to": "美甲评价", "tasks": ["导航点击此刻投稿", "导航点击浏览点赞"]},
    {"from": "美甲", "to": "此刻投稿", "tasks": ["导航点击此刻投稿"]},
    {"from": "美甲", "to": "设计中心", "tasks": ["导航点击返回"]},
    {"from": "制衣", "to": "设计中心", "tasks": ["导航点击返回"]},
    {"from": "情报屋", "to": "设计中心", "tasks": ["导航点击返回"]},
]

DEFAULT_ALIASES = {
    "首页": "主页面",
    "主页": "主页面",
    "主界面": "主页面",
    "设计": "设计中心",
    "制作目标服装": "制衣",
    "目标服装": "制衣",
    "情报": "情报屋",
    "家园": "回家",
    "一键领取": "任务",
    "每日任务": "任务",
    "商店": "商城",
    "联盟页面": "联盟",
    "结伴页面": "结伴",
    "分享": "时空回廊",
    "独自页面": "独自",
    "钻石竞技场": "竞技场",
    "忆海心阶": "心阶",
}

DEFAULT_WAIT_RECOGNITIONS = ["在加载页面", "在接受投稿页面", "导航在空白页面"]
DEFAULT_FALLBACK_STEPS = ["点击返回", "点击体力用尽确定", "主页点击esc", "点击OK-竞技场", "点击esc"]


@AgentServer.custom_action("page_navigate")
class PageNavigateAction(CustomAction):
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        target_page = self._normalize_page(
            self._get_first_param(params, "target", "target_page", "目标页面"),
            params,
        )
        if not target_page:
            logger.error("page_navigate: missing target page")
            return False

        pages = self._build_pages(params)
        routes = self._build_routes(params)
        if target_page not in pages:
            logger.error(f"page_navigate: unknown target page {target_page!r}")
            return False

        retry_detect = self._as_int(params.get("retry_detect"), 3)
        retry_delay = self._as_float(params.get("retry_delay"), 1)
        wait_nodes = self._build_wait_nodes(params)
        fallback_steps = self._build_fallback_steps(params)
        wait_timeout = self._as_float(params.get("wait_timeout"), 15.0)
        wait_interval = max(0.1, self._as_float(params.get("wait_interval"), 0.25))
        max_steps = self._as_int(params.get("max_steps"), 30)
        preferred_page = target_page

        for step_index in range(max(1, max_steps)):
            if not self._wait_while_intermediate(
                context,
                wait_nodes=wait_nodes,
                timeout=wait_timeout,
                interval=wait_interval,
            ):
                return False
            current_page = self._detect_current_page(
                context,
                pages,
                retry=retry_detect,
                retry_delay=retry_delay,
                preferred=preferred_page,
            )

            logger.info(
                f"page_navigate: step={step_index + 1}/{max_steps}, "
                f"current={current_page}, target={target_page}"
            )
            if current_page == target_page:
                return True

            path = self._shortest_path(routes, current_page, target_page) if current_page else []
            if path:
                logger.info(
                    "page_navigate: path="
                    + " -> ".join([current_page] + [edge["to"] for edge in path])
                )
                edge = path[0]
                route_context = context
                if "导航点击返回" in edge["tasks"]:
                    # Return buttons look alike on multiple pages. Keep the guard local.
                    route_context = context.clone()
                    if not route_context.override_pipeline({"导航点击返回": {"action": {
                        "type": "Custom", "param": {
                            "custom_action": "navigation_click",
                            "custom_action_param": {
                                "source_nodes": pages[current_page],
                                "target_nodes": pages[edge["to"]],
                            },
                        },
                    }}}):
                        return False
                for task_name in edge["tasks"]:
                    if not self._run_task(route_context, task_name):
                        logger.error(f"page_navigate: route task failed: {task_name!r}")
                        return False
                    if not self._wait_while_intermediate(
                        context,
                        wait_nodes=wait_nodes,
                        timeout=wait_timeout,
                        interval=wait_interval,
                    ):
                        return False
                if not self._wait_for_page(
                    context, pages[edge["to"]], edge["to"], wait_nodes,
                    wait_timeout, wait_interval,
                ):
                    return False
                preferred_page = edge["to"]
                continue

            if current_page:
                logger.warning(f"page_navigate: no route from {current_page} to {target_page}")
            else:
                logger.warning("page_navigate: failed to identify current page")

            if not self._run_fallback_step(context, fallback_steps):
                logger.error("page_navigate: no fallback step matched current screen")
                return False

        logger.error(f"page_navigate: exceeded maximum steps ({max_steps})")
        return False

    def _wait_for_page(self, context, nodes, page, wait_nodes, timeout, interval):
        """Require two fresh, non-loading observations before the next route edge."""
        started = time.monotonic()
        deadline = started + max(0, timeout)
        hits = 0
        while True:
            image = self._screencap(context)
            ready = not self._intermediate_reason(context, image, wait_nodes)
            ready = ready and any(self._recognize(context, image, node) for node in nodes)
            hits = hits + 1 if ready else 0
            if hits >= 2:
                logger.info(f"page_navigate: ready page={page}, elapsed={time.monotonic() - started:.3f}s")
                return True
            if time.monotonic() >= deadline:
                logger.error(f"page_navigate: destination timeout page={page}, nodes={nodes}, elapsed={time.monotonic() - started:.3f}s")
                return False
            time.sleep(min(max(0.1, interval), max(0, deadline - time.monotonic())))

    def _run_fallback_step(self, context, fallback_steps):
        image = self._screencap(context)
        for task_name in fallback_steps:
            if not self._recognize(context, image, task_name):
                continue
            logger.info(f"page_navigate: running fallback step {task_name!r}")
            if self._run_task(context, task_name):
                return True
            logger.warning(f"page_navigate: fallback step failed: {task_name!r}")
        return False

    def _wait_while_intermediate(
        self,
        context,
        wait_nodes,
        timeout,
        interval,
    ):
        if timeout <= 0:
            return True

        deadline = time.monotonic() + timeout
        waited = False
        while time.monotonic() < deadline:
            image = self._screencap(context)
            reason = self._intermediate_reason(
                context,
                image,
                wait_nodes=wait_nodes,
            )
            if not reason:
                if waited:
                    logger.info("page_navigate: intermediate page finished")
                return True

            waited = True
            logger.info(f"page_navigate: waiting intermediate page: {reason}")
            time.sleep(interval)

        logger.warning("page_navigate: intermediate page wait timeout")
        return False

    def _intermediate_reason(self, context, image, wait_nodes):
        for node_name in wait_nodes:
            if self._recognize(context, image, node_name):
                return node_name

        return ""

    def _detect_current_page(self, context: Context, pages, retry, retry_delay, preferred=None):
        ordered_pages = list(pages.keys())
        if preferred in pages:
            ordered_pages.remove(preferred)
            ordered_pages.insert(0, preferred)

        for attempt in range(max(1, retry)):
            image = self._screencap(context)
            for page_name in ordered_pages:
                for node_name in pages[page_name]:
                    if self._recognize(context, image, node_name):
                        logger.info(f"page_navigate: detected page {page_name} by {node_name}")
                        return page_name
            if attempt < retry - 1:
                time.sleep(retry_delay)

        return None

    def _recognize(self, context: Context, image, node_name):
        try:
            result = context.run_recognition(node_name, image)
            return bool(result and getattr(result, "hit", False))
        except Exception as e:
            logger.warning(f"page_navigate: recognition {node_name!r} failed: {e}")
            return False

    def _run_task(self, context: Context, task_name):
        started = time.monotonic()
        try:
            result = context.run_task(task_name)
            success = getattr(result, "success", None)
            if success is None:
                success = getattr(getattr(result, "status", None), "succeeded", False)
            succeeded = result is not None and bool(success)
            logger.info(f"page_navigate: task={task_name}, success={succeeded}, elapsed={time.monotonic() - started:.3f}s")
            return succeeded
        except Exception as e:
            logger.error(f"page_navigate: task {task_name!r} raised: {e}")
            return False

    def _screencap(self, context: Context):
        controller = context.tasker.controller
        controller.post_screencap().wait()
        return controller.cached_image

    def _shortest_path(self, routes, start, target):
        graph = {}
        for route in routes:
            graph.setdefault(route["from"], []).append(route)

        queue = deque([(start, [])])
        visited = {start}
        while queue:
            page, path = queue.popleft()
            if page == target:
                return path

            for edge in graph.get(page, []):
                next_page = edge["to"]
                if next_page in visited:
                    continue
                visited.add(next_page)
                queue.append((next_page, path + [edge]))

        return []

    def _build_pages(self, params):
        pages = {name: list(nodes) for name, nodes in DEFAULT_PAGES.items()}
        custom_pages = params.get("pages") or {}
        for page_name, nodes in custom_pages.items():
            normalized = self._normalize_page(page_name, params)
            if isinstance(nodes, str):
                pages[normalized] = [nodes]
            elif isinstance(nodes, list):
                pages[normalized] = [node for node in nodes if isinstance(node, str)]
        return pages

    def _build_wait_nodes(self, params):
        nodes = list(DEFAULT_WAIT_RECOGNITIONS)
        custom_nodes = self._get_first_param(params, "wait_nodes", "waiting_nodes", "intermediate_nodes") or []
        if isinstance(custom_nodes, str):
            custom_nodes = [custom_nodes]
        for node_name in custom_nodes:
            if isinstance(node_name, str) and node_name not in nodes:
                nodes.append(node_name)
        return nodes

    def _build_fallback_steps(self, params):
        steps = params.get("fallback_steps", DEFAULT_FALLBACK_STEPS)
        if isinstance(steps, str):
            steps = [steps]
        if not isinstance(steps, list):
            return list(DEFAULT_FALLBACK_STEPS)
        return [step for step in steps if isinstance(step, str) and step]

    def _build_routes(self, params):
        routes = [self._normalize_route(route, params) for route in DEFAULT_ROUTES]
        for route in params.get("routes") or []:
            normalized = self._normalize_route(route, params)
            if normalized:
                routes.append(normalized)
        return [route for route in routes if route]

    def _normalize_route(self, route, params):
        if not isinstance(route, dict):
            return None

        from_page = self._normalize_page(route.get("from") or route.get("source"), params)
        to_page = self._normalize_page(route.get("to") or route.get("target"), params)
        tasks = route.get("tasks") or route.get("task")
        if isinstance(tasks, str):
            tasks = [tasks]
        if not from_page or not to_page or not isinstance(tasks, list) or not tasks:
            return None

        return {
            "from": from_page,
            "to": to_page,
            "tasks": [task for task in tasks if isinstance(task, str)],
        }

    def _normalize_page(self, page_name, params):
        if not page_name:
            return ""

        page_name = str(page_name)
        aliases = dict(DEFAULT_ALIASES)
        aliases.update(params.get("aliases") or {})
        return aliases.get(page_name, page_name)

    def _get_first_param(self, params, *names):
        for name in names:
            value = params.get(name)
            if value is not None:
                return value
        return None

    def _as_int(self, value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _as_float(self, value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


@AgentServer.custom_action("main_story_return_to_solo")
class MainStoryReturnToSoloAction(CustomAction):
    """Return through the optional investigation page with page-guarded clicks."""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        navigation = PageNavigateAction()
        route_context = context.clone()
        solo = "导航在独自页面"
        investigation = "导航在调查行动页面"
        stages = [
            (["导航在主线故事页面"], [solo, investigation]),
            ([investigation], [solo]),
        ]
        for sources, targets in stages:
            if not route_context.override_pipeline({"导航点击返回": {"action": {
                "type": "Custom", "param": {
                    "custom_action": "navigation_click",
                    "custom_action_param": {
                        "source_nodes": sources,
                        "target_nodes": targets,
                    },
                },
            }}}):
                return False
            # navigation_click checks arrival before clicking, so the second stage
            # is a no-op when the first return already reached the solo page.
            if not navigation._run_task(route_context, "导航点击返回"):
                return False
        return True


@AgentServer.custom_action("navigation_click")
class NavigationClickAction(CustomAction):
    """Retry only a recognised button on the original, non-loading page."""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        params = json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        sources = params.get("source_nodes", [])
        targets = params.get("target_nodes", [])
        if not sources or not targets:
            logger.error(f"navigation_click: missing page guards node={argv.node_name}")
            return False
        navigation = PageNavigateAction()
        started = time.monotonic()
        deadline = started + 15.0
        next_click = started
        clicks = 0
        target_seen = False
        while True:
            image = navigation._screencap(context)
            # Latch arrival: even a later transient miss must never click the old entry.
            target_seen = target_seen or any(navigation._recognize(context, image, node) for node in targets)
            loading = navigation._intermediate_reason(context, image, DEFAULT_WAIT_RECOGNITIONS)
            if target_seen and not loading:
                logger.info(f"navigation_click: arrived node={argv.node_name}, clicks={clicks}, elapsed={time.monotonic() - started:.3f}s")
                return True
            if time.monotonic() >= deadline:
                logger.error(f"navigation_click: timeout node={argv.node_name}, clicks={clicks}, loading={loading}, target_seen={target_seen}")
                return False
            if not loading and not target_seen and clicks < 3 and time.monotonic() >= next_click:
                on_source = any(navigation._recognize(context, image, node) for node in sources)
                if on_source:
                    result = context.run_recognition(argv.node_name, image)
                    if result and result.hit and time.monotonic() < deadline:
                        box = result.box
                        target = params.get("target")
                        if target is None:
                            if box is None or box.w <= 0 or box.h <= 0:
                                logger.error(f"navigation_click: invalid button box node={argv.node_name}")
                                return False
                            target = [box.x, box.y, box.w, box.h]
                        action = context.run_action_direct("Click", JClick(target=target))
                        clicks += 1
                        next_click = time.monotonic() + 1.5
                        logger.info(f"navigation_click: node={argv.node_name}, click={clicks}/3, target={target}, success={bool(action and action.success)}")
                        if not action or not action.success:
                            return False
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))
