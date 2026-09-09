"""使用包内 Python 和主程序原生库验证真实 Agent 握手，无需模拟器。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-dir", type=Path, default=Path("install"))
    parser.add_argument("--platform", required=True)
    args = parser.parse_args()
    root = args.install_dir.resolve()
    manifest = json.loads((root / "python-dependencies.json").read_text(encoding="utf-8"))
    # 必须在导入 maa 前设置，客户端使用主程序目录里的原生库。
    os.environ["MAAFW_BINARY_PATH"] = str(root / "runtimes" / args.platform / "native")
    from maa.agent_client import AgentClient
    from maa.library import Library
    from maa.resource import Resource
    from maa.toolkit import Toolkit

    actual = Library.version().removeprefix("v")
    if actual != manifest["maafw_version"]:
        raise RuntimeError(f"Native framework {actual} != Python {manifest['maafw_version']}")

    Toolkit.init_option(root.parent / "agent-check-logs")
    resource = Resource()
    client = AgentClient()
    if not client.bind(resource) or not client.set_timeout(15000):
        raise RuntimeError("Failed to initialize AgentClient")
    env = os.environ.copy()
    # Agent 必须使用 wheel 自带的原生库，避免掩盖混装问题。
    env.pop("MAAFW_BINARY_PATH", None)
    interface = json.loads((root / "interface.json").read_text(encoding="utf-8"))
    config = interface["agent"]
    command = [str(root / config["child_exec"]), *config["child_args"], client.identifier]
    process = subprocess.Popen(command, cwd=root, env=env)
    try:
        if not client.connect():
            raise RuntimeError("Packaged Agent handshake failed")
        actions = client.custom_action_list
        if "arena_compare" not in actions or "click_region" not in actions:
            raise RuntimeError(f"Missing custom actions: {actions}")
        if not client.disconnect():
            raise RuntimeError("Agent disconnect failed")
        if process.wait(timeout=15) != 0:
            raise RuntimeError("Agent exited with an error")
        print(f"Packaged Agent handshake OK: MaaFramework {actual}, {len(actions)} actions")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        # 在退出进程前销毁客户端。
        del client
        del resource


if __name__ == "__main__":
    main()
