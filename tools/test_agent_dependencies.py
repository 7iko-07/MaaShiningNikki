"""离线启动与发行版本匹配回归测试；不安装或修改本机依赖。"""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools/ci"))
spec = importlib.util.spec_from_file_location("agent_main", ROOT / "agent/main.py")
agent_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_main)
from bundle_python_dependencies import framework_version
import bundle_python_dependencies


class DependenciesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch_root = patch.object(agent_main, "parent_dir", str(self.root))
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        log_patch = patch.object(agent_main, "logger")
        log_patch.start()
        self.addCleanup(log_patch.stop)

    def manifest(self, packages):
        (self.root / "python-dependencies.json").write_text(
            json.dumps({"packages": packages}), encoding="utf-8"
        )

    def test_matching_package_starts_without_pip_even_with_legacy_config(self):
        self.manifest({"maafw": "5.13.0", "loguru": "0.7.3"})
        (self.root / "config").mkdir()
        (self.root / "config/pip_config.json").write_text(
            '{"enable_pip_update":true,"enable_pip_install":true}', encoding="utf-8"
        )
        versions = {"maafw": "5.13.0", "loguru": "0.7.3"}
        with patch.object(agent_main.importlib.util, "find_spec", return_value=object()), \
             patch.object(agent_main, "version", side_effect=versions.__getitem__), \
             patch("subprocess.Popen", side_effect=AssertionError("must not run pip")), \
             patch.object(agent_main, "agent") as run_agent:
            agent_main.main()
            run_agent.assert_called_once()

    def test_wrong_version_fails_before_starting_agent(self):
        self.manifest({"maafw": "5.13.0"})
        with patch.object(agent_main.importlib.util, "find_spec", return_value=object()), \
             patch.object(agent_main, "version", return_value="5.12.3"), \
             patch.object(agent_main, "agent") as run_agent:
            with self.assertRaisesRegex(RuntimeError, "依赖版本不匹配"):
                agent_main.main()
            run_agent.assert_not_called()

    def test_missing_module_fails_without_installing(self):
        with patch.object(agent_main.importlib.util, "find_spec", return_value=None), \
             patch("subprocess.Popen", side_effect=AssertionError("must not run pip")):
            with self.assertRaisesRegex(RuntimeError, "缺少依赖"):
                agent_main.check_dependencies()

    def test_missing_transitive_dependency_fails(self):
        self.manifest({"numpy": "2.0.0"})
        with patch.object(agent_main.importlib.util, "find_spec", return_value=object()), \
             patch.object(agent_main, "version", side_effect=agent_main.PackageNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "实际 未安装"):
                agent_main.check_dependencies()

    def test_framework_release_tag_validation(self):
        self.assertEqual(framework_version("v5.13.0"), "5.13.0")
        for invalid in ("latest", "v5.13.0-beta.1", "5.13.0 --upgrade"):
            with self.assertRaises(ValueError):
                framework_version(invalid)

    def test_release_selects_native_libraries_for_python_platform(self):
        self.manifest({})
        platforms = {
            "win-amd64": "win-x64",
            "win-arm64": "win-arm64",
            "linux-x86_64": "linux-x64",
            "linux-aarch64": "linux-arm64",
            "macosx-10.13-x86_64": "osx-x64",
            "macosx-11.0-arm64": "osx-arm64",
        }
        # 同时存在多个运行时也必须按平台选择，不能取 glob 的第一项。
        for tag in platforms.values():
            (self.root / "runtimes" / tag / "native").mkdir(parents=True)
        for python_platform, tag in platforms.items():
            with self.subTest(platform=python_platform), \
                 patch.object(agent_main.sysconfig, "get_platform", return_value=python_platform), \
                 patch.dict(agent_main.os.environ, {"MAAFW_BINARY_PATH": "unrelated-native"}):
                agent_main.configure_native_library_path()
                self.assertEqual(agent_main.os.environ["MAAFW_BINARY_PATH"],
                                 str((self.root / "runtimes" / tag / "native").resolve()))

    def test_source_development_preserves_native_library_loading(self):
        for value in (None, "developer-native"):
            with self.subTest(path=value), patch.dict(agent_main.os.environ):
                agent_main.os.environ.pop("MAAFW_BINARY_PATH", None)
                if value is not None:
                    agent_main.os.environ["MAAFW_BINARY_PATH"] = value
                agent_main.configure_native_library_path()
                self.assertEqual(agent_main.os.environ.get("MAAFW_BINARY_PATH"), value)

    def test_release_missing_native_directory_fails_instead_of_using_other_platform(self):
        self.manifest({})
        (self.root / "runtimes/win-arm64/native").mkdir(parents=True)
        with patch.object(agent_main.sysconfig, "get_platform", return_value="win-amd64"), \
             patch.dict(agent_main.os.environ, {"MAAFW_BINARY_PATH": "unrelated-native"}):
            with self.assertRaisesRegex(RuntimeError, "缺少原生库目录.*win-x64"):
                agent_main.configure_native_library_path()
            self.assertEqual(agent_main.os.environ["MAAFW_BINARY_PATH"], "unrelated-native")

    def run_bundle(self, actual):
        package = Mock()
        package.metadata = {"Name": "MaaFw"}
        package.version = actual
        argv = ["bundle_python_dependencies.py", "--framework-version", "v5.14.2",
                "--install-dir", str(self.root)]
        with patch.object(sys, "argv", argv), \
             patch.object(sys, "executable", str(self.root / "python/python.exe")), \
             patch.object(bundle_python_dependencies.subprocess, "run") as run, \
             patch.object(bundle_python_dependencies.importlib.metadata, "distributions",
                          return_value=[package]), \
             patch.object(bundle_python_dependencies.importlib.metadata, "version",
                          return_value=actual):
            bundle_python_dependencies.main()
            self.assertEqual(run.call_count, 2)
            self.assertIn("maafw==5.14.2", run.call_args_list[0].args[0])

    def test_bundle_accepts_matching_wheel_version_with_or_without_v(self):
        for actual in ("5.14.2", "v5.14.2"):
            with self.subTest(actual=actual):
                self.run_bundle(actual)
                manifest = json.loads((self.root / "python-dependencies.json").read_text(
                    encoding="utf-8"
                ))
                self.assertEqual(manifest["maafw_version"], "5.14.2")
                # Agent 的依赖检查仍使用发行包元数据中的原始版本。
                self.assertEqual(manifest["packages"]["MaaFw"], actual)
                with patch.object(agent_main.importlib.util, "find_spec",
                                  return_value=object()), \
                     patch.object(agent_main, "version", return_value=actual), \
                     patch.object(agent_main, "agent") as run_agent:
                    agent_main.main()
                    run_agent.assert_called_once()

    def test_bundle_rejects_different_wheel_version_before_writing_manifest(self):
        with self.assertRaisesRegex(RuntimeError, "v5.13.0.*5.14.2"):
            self.run_bundle("v5.13.0")
        self.assertFalse((self.root / "python-dependencies.json").exists())

    def test_bundle_rejects_interpreter_outside_release_before_installing_or_trimming(self):
        argv = ["bundle_python_dependencies.py", "--framework-version", "v5.14.3",
                "--install-dir", str(self.root)]
        with patch.object(sys, "argv", argv), \
             patch.object(sys, "executable", str(self.root / ".venv/python.exe")), \
             patch.object(bundle_python_dependencies.subprocess, "run") as run, \
             patch.object(bundle_python_dependencies, "trim_python") as trim:
            with self.assertRaisesRegex(RuntimeError, "Python inside the release package"):
                bundle_python_dependencies.main()
            run.assert_not_called()
            trim.assert_not_called()

    def test_bundle_does_not_trim_or_write_manifest_when_pip_check_fails(self):
        argv = ["bundle_python_dependencies.py", "--framework-version", "v5.14.3",
                "--install-dir", str(self.root)]
        failure = bundle_python_dependencies.subprocess.CalledProcessError(1, "pip check")
        with patch.object(sys, "argv", argv), \
             patch.object(sys, "executable", str(self.root / "python/python.exe")), \
             patch.object(bundle_python_dependencies.subprocess, "run", side_effect=[Mock(), failure]), \
             patch.object(bundle_python_dependencies, "trim_python") as trim:
            with self.assertRaises(bundle_python_dependencies.subprocess.CalledProcessError):
                bundle_python_dependencies.main()
            trim.assert_not_called()
        self.assertFalse((self.root / "python-dependencies.json").exists())

    def test_trim_removes_extras_and_preserves_runtime_files(self):
        python = self.root / "python"
        removed = ["setup_pip.py", "get-pip.py", "Lib/helper.pyc", "Lib/__pycache__/os.pyc",
                   "Scripts/pip.exe", "Scripts/pip3.12-script.py", "bin/pip", "bin/pip3.12"]
        kept = ["Lib/helper.py", "python.exe", "python312.zip", "LICENSE.txt",
                "bin/python3", "bin/pip-tools", "Scripts/pipelines.exe"]
        for site in ("Lib/site-packages", "lib/python3.12/site-packages"):
            removed.extend(f"{site}/{name}" for name in (
                "maa/bin/native.dll", "MaaAgentBinary/maa-agent", "pip/__init__.py",
                "pip-26.0.dist-info/METADATA", "numpy/test/example.py",
                "numpy/_core/tests/nested/tests/example.py", "loguru/__pycache__/cached.pyc",
            ))
            kept.extend(f"{site}/{name}" for name in (
                "maa/__init__.py", "maafw-5.14.3.dist-info/METADATA",
                "maafw-5.14.3.dist-info/licenses/LICENSE.md",
                "MaaAgentBinary-1.0.1.dist-info/METADATA", "MaaAgentBinary-1.0.1.dist-info/LICENSE",
                "numpy/_core/native.pyd", "numpy/testing/_private/utils.py",
                "numpy-2.5.4.dist-info/licenses/LICENSE.txt", "numpy.libs/openblas.dll",
                "other/tests/example.py", "standalone.pyc",
            ))
        shared = [self.root / "runtimes/win-x64/native/MaaFramework.dll",
                  self.root / "libs/MaaAgentBinary/maa-agent"]
        for path in [python / name for name in removed + kept] + shared:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")

        bundle_python_dependencies.trim_python(python)
        for name in removed:
            self.assertFalse((python / name).exists(), name)
        for path in [python / name for name in kept] + shared:
            self.assertEqual(path.read_bytes(), b"fixture", str(path))

    def test_trim_rejects_symlink_outside_python(self):
        protected = self.root / "keep.txt"
        protected.write_bytes(b"fixture")
        python = self.root / "python"
        site = python / "Lib/site-packages"
        site.mkdir(parents=True)
        try:
            (site / "pip").symlink_to(self.root, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"此系统不允许创建测试符号链接: {error}")
        with self.assertRaises(ValueError):
            bundle_python_dependencies.trim_python(python)
        self.assertEqual(protected.read_bytes(), b"fixture")


if __name__ == "__main__":
    unittest.main()
