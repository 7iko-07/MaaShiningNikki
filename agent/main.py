import os
import sys
import sysconfig
import json
import importlib.util
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# 获取当前main.py所在路径并设置上级目录为工作目录
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)
# print(f"设置工作目录为: {parent_dir}")

# 将当前目录添加到路径
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from utils import logger
except ImportError:
    # 如果logger不存在，创建一个简单的logger
    import logging

    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
    logger = logging


REQUIRED_MODULES = {
    "maafw": "maa",
    "loguru": "loguru",
}


def check_dependencies():
    """只检查随程序分发的依赖；运行时不联网、不调用 pip。"""
    missing = [
        package for package, module in REQUIRED_MODULES.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(
            f"Agent 缺少依赖: {', '.join(missing)}。请重新解压完整安装包；"
            "源码开发请手动安装 requirements.txt。"
        )

    manifest_path = Path(parent_dir) / "python-dependencies.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for package, expected in manifest["packages"].items():
            try:
                installed = version(package)
            except PackageNotFoundError:
                installed = "未安装"
            if installed != expected:
                raise RuntimeError(
                    f"依赖版本不匹配: {package} 需要 {expected}，实际 {installed}。"
                    "请重新解压完整安装包，不要混用不同版本的 python 和 runtimes 目录。"
                )
    logger.info(f"Agent 依赖检查通过，maafw={version('maafw')}；运行时不更新依赖")


def configure_native_library_path():
    """发行包与 UI 共用原生库；源码开发仍使用原有 wheel 或环境变量。"""
    release_root = Path(parent_dir)
    if not (release_root / "python-dependencies.json").is_file():
        return

    # 使用 Python 的目标平台，避免 Windows ARM 上的 x64 仿真选错架构。
    python_platform = sysconfig.get_platform().lower()
    os_tag = {"win": "win", "linux": "linux", "macosx": "osx"}.get(
        python_platform.split("-", 1)[0]
    )
    arch_tag = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(
        python_platform.rsplit("-", 1)[-1]
    )
    if os_tag is None or arch_tag is None:
        raise RuntimeError(f"发行包不支持当前 Python 平台: {python_platform}")

    native_dir = release_root / "runtimes" / f"{os_tag}-{arch_tag}" / "native"
    if not native_dir.is_dir():
        raise RuntimeError(f"发行包缺少原生库目录: {native_dir}。请重新解压对应平台的完整安装包。")
    os.environ["MAAFW_BINARY_PATH"] = str(native_dir.resolve())


def agent():
    try:
        configure_native_library_path()

        from maa.toolkit import Toolkit
        from maa.agent.agent_server import AgentServer
        import custom

        Toolkit.init_option("./")

        if len(sys.argv) != 2 or not sys.argv[1]:
            raise RuntimeError("缺少 Agent 连接标识符，请通过主程序启动")
        socket_id = sys.argv[1]

        if not AgentServer.start_up(socket_id):
            raise RuntimeError(f"AgentServer 启动失败: {socket_id}")
        logger.info("AgentServer 已启动，等待客户端连接")
        try:
            AgentServer.join()
        finally:
            AgentServer.shut_down()
        logger.info("AgentServer 关闭")
    except Exception as e:
        logger.exception("Agent 运行过程中发生异常")
        raise


def main():
    try:
        check_dependencies()
    except Exception:
        logger.exception("Agent 依赖检查失败")
        raise
    agent()


if __name__ == "__main__":
    main()
