"""在 CI 中用发行包自己的 Python 安装并精简依赖，禁止用于开发虚拟环境。"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def framework_version(tag: str) -> str:
    # GitHub Latest 为正式发行版；遇到未知版本格式时阻止打包。
    if not re.fullmatch(r"v?\d+\.\d+\.\d+", tag):
        raise ValueError(f"Unsupported MaaFramework release tag: {tag!r}")
    return tag.removeprefix("v")


def trim_python(python_root: Path):
    """依赖安装完成后，移除重复库和 Android 工具、pip、NumPy 测试和缓存。"""
    def remove(path):
        # 删除范围限定在包内 Python。
        path.resolve().relative_to(python_root.resolve())
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    # Windows 嵌入式 Python 和 Unix 独立 Python 的目录布局。
    for site in [python_root / "Lib/site-packages", *python_root.glob("lib/python*/site-packages")]:
        for path in [site / "maa/bin", site / "MaaAgentBinary", site / "pip", *site.glob("pip-*.dist-info")]:
            if path.exists():
                remove(path)
        for name in ("test", "tests"):
            for path in list((site / "numpy").rglob(name)):
                if path.is_dir():
                    remove(path)

    for scripts in (python_root / "Scripts", python_root / "bin"):
        for path in scripts.glob("pip*"):
            if re.fullmatch(r"pip(?:\d+(?:\.\d+)?)?(?:\.exe|-script\.py)?", path.name):
                remove(path)
    for name in ("setup_pip.py", "get-pip.py"):
        path = python_root / name
        if path.is_file():
            remove(path)
    for path in list(python_root.rglob("__pycache__")):
        if path.is_dir():
            remove(path)
    for path in python_root.rglob("*.pyc"):
        if path.with_suffix(".py").is_file():
            remove(path)
    print("Removed duplicate native libraries and Android tools, pip, NumPy tests and bytecode caches")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework-version", required=True)
    parser.add_argument("--install-dir", type=Path, default=Path("install"))
    args = parser.parse_args()
    root = args.install_dir.resolve()
    python_root = root / "python"
    if not Path(sys.executable).resolve().is_relative_to(python_root):
        raise RuntimeError("Run this script with the Python inside the release package")

    expected = framework_version(args.framework_version)
    subprocess.run(
        [sys.executable, "-I", "-m", "pip", "--isolated", "install",
         "--disable-pip-version-check", "--no-cache-dir", "--no-warn-script-location",
         "--only-binary=:all:",
         "--index-url", "https://pypi.org/simple", "--upgrade",
         "-r", str(root / "requirements.txt"), f"maafw=={expected}"],
        check=True,
    )
    subprocess.run([sys.executable, "-I", "-m", "pip", "check"], check=True)
    packages = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"].lower() not in {"pip", "setuptools", "wheel"}
    }
    actual = importlib.metadata.version("maafw")
    # 部分 wheel 的 Version 元数据保留 v 前缀；与发行标签使用相同规则校验。
    if framework_version(actual) != expected:
        raise RuntimeError(
            f"Installed maafw {actual!r} does not match the native release {expected!r}"
        )
    (root / "python-dependencies.json").write_text(
        json.dumps({"maafw_version": expected, "packages": packages},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    trim_python(python_root)
    print(f"Bundled Python dependencies: {packages}")


if __name__ == "__main__":
    main()
