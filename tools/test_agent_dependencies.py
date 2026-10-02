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


if __name__ == "__main__":
    unittest.main()
