"""竞技场周期、完成记录，以及仅在当前进程有效的登录身份。"""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from utils import logger


BEIJING = timezone(timedelta(hours=8))
STATE_PATH = Path(__file__).resolve().parents[2] / "config" / "arena_weekly.sqlite3"


def now_beijing():
    return datetime.now(BEIJING)


def in_settlement(now=None):
    now = (now or now_beijing()).astimezone(BEIJING)
    return (now.weekday() == 5 and now.hour >= 21) or (
        now.weekday() == 6 and now.hour < 12
    )


def week_key(now=None):
    now = (now or now_beijing()).astimezone(BEIJING)
    sunday = (now - timedelta(days=(now.weekday() + 1) % 7)).replace(
        hour=12, minute=0, second=0, microsecond=0
    )
    if now < sunday:
        sunday -= timedelta(days=7)
    return sunday.isoformat()


class StateStore:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else STATE_PATH

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS account_modes "
                               "(device TEXT PRIMARY KEY, multi INTEGER NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS outfits "
                               "(account TEXT PRIMARY KEY, week TEXT NOT NULL, completed_at TEXT NOT NULL)")
            return connection
        except Exception:
            connection.close()
            raise

    def set_mode(self, device, multi):
        with closing(self._connect()) as connection, connection:
            connection.execute("INSERT OR REPLACE INTO account_modes VALUES (?, ?)",
                               (device, int(multi)))

    def mode(self, device):
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT multi FROM account_modes WHERE device = ?",
                                     (device,)).fetchone()
            # 没运行过 start 时不推断为单账号，先搭配一次但不记账。
            return bool(row[0]) if row else None

    def completed(self, account, week):
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT week FROM outfits WHERE account = ?",
                                     (account,)).fetchone()
            return row is not None and row[0] == week

    def mark_completed(self, account, week, now):
        with closing(self._connect()) as connection, connection:
            connection.execute("INSERT OR REPLACE INTO outfits VALUES (?, ?, ?)",
                               (account, week, now.isoformat()))


@dataclass
class LoginSession:
    task_id: int
    multi: bool
    suffix: str = ""
    server: str = ""


_sessions = {}


def begin_login(context, task_id, multi):
    device = context.tasker.controller.uuid
    _sessions[device] = LoginSession(task_id, multi)
    try:
        StateStore().set_mode(device, multi)
    except (OSError, sqlite3.Error) as exc:
        logger.warning(f"竞技场账号模式保存失败：{exc}")


def set_login_suffix(context, suffix):
    session = _sessions.get(context.tasker.controller.uuid)
    if session:
        session.suffix = suffix
        session.server = ""


def set_login_server(context, server):
    session = _sessions.get(context.tasker.controller.uuid)
    if session:
        session.server = server


def take_account(context):
    """登录身份仅供随后的一次竞技场任务使用，绝不从磁盘恢复旧身份。"""
    device = context.tasker.controller.uuid
    if not device:
        return None
    session = _sessions.pop(device, None)
    if session:
        detail = context.tasker.get_task_detail(session.task_id)
        if not detail or not detail.status.succeeded:
            return None
        if not session.multi:
            return "default"
        if session.suffix and session.server:
            return json.dumps([session.suffix, session.server], ensure_ascii=False)
        return None
    try:
        if StateStore().mode(device) is False:
            return "default"
    except (OSError, sqlite3.Error) as exc:
        logger.warning(f"竞技场账号模式读取失败：{exc}")
    return None
