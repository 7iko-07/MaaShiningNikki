"""离线启动与发行版本匹配回归测试；不安装或修改本机依赖。"""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools/ci"))
spec = importlib.util.spec_from_file_location("agent_main", ROOT / "agent/main.py")
agent_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_main)
from bundle_python_dependencies import framework_version


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


if __name__ == "__main__":
    unittest.main()
