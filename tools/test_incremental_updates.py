import json
import urllib.error
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools.ci import build_incremental_updates as updates


class IncrementalUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def archive(self, name, files):
        path = self.root / name
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for filename, content in files.items():
                archive.writestr(filename, content)
        return path

    def test_exact_delta_and_deleted_dependency_metadata(self):
        old = self.archive("old.zip", {"same.txt": "same", "changed.txt": "old",
                                      "python/Lib/site-packages/maafw-5.13.1.dist-info/METADATA": "old"})
        new = self.archive("new.zip", {"same.txt": "same", "changed.txt": "new", "added.txt": "added",
                                      "python/Lib/site-packages/maafw-5.14.2.dist-info/METADATA": "new"})
        baseline = updates.archive_manifest(old, "v1", "win-x86_64")
        target = updates.archive_manifest(new, "v2", "win-x86_64")
        delta = updates.delta_manifest(baseline, target)
        self.assertEqual(delta["baseVersion"], "v1")
        self.assertNotIn("same.txt", delta["payload"])
        self.assertIn("changed.txt", delta["payload"])
        self.assertIn("added.txt", delta["payload"])
        self.assertEqual(delta["deleted"], ["python/Lib/site-packages/maafw-5.13.1.dist-info/METADATA"])
        path = self.root / "delta.zip"
        updates.write_delta(new, path, delta)
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(set(archive.namelist()), set(delta["payload"]) | {updates.MANIFEST})
            self.assertEqual(archive.read("changed.txt"), b"new")
            self.assertEqual(json.loads(archive.read(updates.MANIFEST)), delta)

    def test_protected_files_excluded(self):
        package = self.archive("package.zip", {"config/user.json": "private", "logs/log.txt": "log",
                                              "agent/user.sqlite3": "state", "agent/user.sqlite3-wal": "state",
                                              "appsettings.json": "settings", "agent/main.py": "script"})
        manifest = updates.archive_manifest(package, "v1", "win-x86_64")
        self.assertEqual(manifest["payload"], ["agent/main.py"])

    def test_timestamps_do_not_create_changes(self):
        a = self.archive("a.zip", {"file.txt": "same"})
        b = self.root / "b.zip"
        with zipfile.ZipFile(a) as source, zipfile.ZipFile(b, "w", zipfile.ZIP_STORED) as archive:
            info = zipfile.ZipInfo("file.txt", date_time=(2020, 1, 1, 0, 0, 0))
            info.external_attr = source.getinfo("file.txt").external_attr
            archive.writestr(info, "same")
        delta = updates.delta_manifest(updates.archive_manifest(a, "v1", "win-x86_64"),
                                       updates.archive_manifest(b, "v2", "win-x86_64"))
        self.assertEqual(delta["payload"], [])

    def test_equal_length_different_content_changes(self):
        a = self.archive("a.zip", {"f": "abc"})
        b = self.archive("b.zip", {"f": "xyz"})
        delta = updates.delta_manifest(updates.archive_manifest(a, "v1", "win-x86_64"),
                                       updates.archive_manifest(b, "v2", "win-x86_64"))
        self.assertEqual(delta["payload"], ["f"])

    def test_case_collisions_rejected(self):
        package = self.archive("package.zip", {"a.txt": "a", "A.txt": "b"})
        with self.assertRaises(ValueError):
            updates.archive_manifest(package, "v1", "win-x86_64")

    def test_symlink_contract_and_escape(self):
        package = self.root / "links.zip"
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr("python/bin/python3.12", "binary")
            info = zipfile.ZipInfo("python/bin/python3")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "python3.12")
        manifest = updates.archive_manifest(package, "v1", "linux-x86_64")
        link = next(f for f in manifest["files"] if f["path"] == "python/bin/python3")
        self.assertEqual(link["linkTarget"], "python3.12")
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr(info, "../../../outside")
        with self.assertRaises(ValueError):
            updates.archive_manifest(package, "v1", "linux-x86_64")

    def test_legacy_release_bootstrap_and_index(self):
        name = "MaaShiningNikki-win-x86_64-v2.zip"
        old_name = "MaaShiningNikki-win-x86_64-v1.zip"
        old = self.archive(old_name, {"unchanged.bin": bytes(range(256)) * 4096, "change": "old"})
        archive = self.archive(name, {"unchanged.bin": bytes(range(256)) * 4096, "change": "new"})
        # Keep the baseline outside the directory of new release assets.
        baseline = self.root / "baseline.zip"
        old.rename(baseline)
        releases = [{"draft": False, "tag_name": "v1", "assets": [{"name": old_name, "browser_download_url": "mock"}]}]

        def fake_download(asset, target, token):
            target.write_bytes(baseline.read_bytes())

        import io
        with patch.object(updates, "request", return_value=io.BytesIO(json.dumps(releases).encode())), \
                patch.object(updates, "download", side_effect=fake_download):
            # Avoid considering our baseline fixture as a new release package.
            moved = self.root.parent / (self.root.name + "-baseline.zip")
            baseline.rename(moved)
            baseline = moved
            try:
                index = updates.generate(self.root, "owner/repo", "v2", 3, "")
            finally:
                moved.unlink()
        self.assertEqual(len(index["deltas"]), 1)
        delta = index["deltas"][0]
        self.assertEqual(delta["fullAsset"], name)
        self.assertEqual(delta["baseVersion"], "v1")
        self.assertLess(delta["size"], archive.stat().st_size)
        self.assertNotIn("win", delta["asset"])
        with zipfile.ZipFile(archive) as package:
            self.assertIn(updates.MANIFEST, package.namelist())
        self.assertIn(updates.file_hash(archive), (self.root / "checksums.sha256").read_text())
        self.assertTrue((self.root / updates.manifest_asset_name(name)).exists())

    def test_platform_mismatch_rejected(self):
        old = {"schema": 1, "platform": "win-x86_64", "files": []}
        new = {"schema": 1, "platform": "win-aarch64", "files": []}
        with self.assertRaises(ValueError):
            updates.delta_manifest(old, new)

    def test_history_unavailable_still_publishes_full_package(self):
        archive = self.archive("MaaShiningNikki-win-x86_64-v2.zip", {"resource/test.json": "data"})
        with patch.object(updates, "request", side_effect=urllib.error.URLError("offline")):
            index = updates.generate(self.root, "owner/repo", "v2", 3, "")
        self.assertEqual(index["deltas"], [])
        with zipfile.ZipFile(archive) as package:
            self.assertIn(updates.MANIFEST, package.namelist())
        self.assertIn(updates.file_hash(archive), (self.root / "checksums.sha256").read_text())


if __name__ == "__main__":
    unittest.main()
