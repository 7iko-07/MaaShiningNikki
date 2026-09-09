"""在 CI 中用发行包自己的 Python 安装依赖，禁止用于开发虚拟环境。"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys


def framework_version(tag: str) -> str:
    # GitHub Latest 为正式发行版；遇到未知版本格式时阻止打包。
    if not re.fullmatch(r"v?\d+\.\d+\.\d+", tag):
        raise ValueError(f"Unsupported MaaFramework release tag: {tag!r}")
    return tag.removeprefix("v")


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
    if importlib.metadata.version("maafw") != expected:
        raise RuntimeError("Installed maafw does not match the native release")
    (root / "python-dependencies.json").write_text(
        json.dumps({"maafw_version": expected, "packages": packages},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Bundled Python dependencies: {packages}")


if __name__ == "__main__":
    main()
