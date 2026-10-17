import fnmatch
import http.client
import io
import json
import urllib.error
from pathlib import Path
import stat
import tempfile
import urllib.request
import unittest
import warnings
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

    def test_unix_case_collisions_preserve_full_packages_and_other_platforms(self):
        names = [f"MaaShiningNikki-{platform}-v2.zip"
                 for platform in ("linux-x86_64", "macos-aarch64", "osx-x64")]
        archives = [self.archive(name, {"python/share/terminfo/2/2621A": "upper",
                                        "python/share/terminfo/2/2621a": "lower"})
                    for name in names]
        original = [archive.read_bytes() for archive in archives]
        windows = self.archive("MaaShiningNikki-win-x86_64-v2.zip", {"agent/main.py": "script"})
        with patch.object(updates, "request", return_value=io.BytesIO(b"[]")), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            index = updates.generate(self.root, "owner/repo", "v2", 3, "")
        self.assertEqual(index["deltas"], [])
        self.assertEqual(json.loads((self.root / "update-index.json").read_text()), index)
        checksums = (self.root / "checksums.sha256").read_text()
        for archive, content in zip(archives, original):
            self.assertEqual(archive.read_bytes(), content)
            self.assertFalse((self.root / updates.manifest_asset_name(archive.name)).exists())
            self.assertIn(f"{updates.file_hash(archive)}  {archive.name}\n", checksums)
            self.assertIn(f"::warning::Publishing full package only for {archive.name}", output.getvalue())
        with zipfile.ZipFile(windows) as package:
            self.assertIn(updates.MANIFEST, package.namelist())
        self.assertTrue((self.root / updates.manifest_asset_name(windows.name)).exists())
        self.assertIn(f"{updates.file_hash(windows)}  {windows.name}\n", checksums)

    def test_windows_case_collisions_still_fail_release(self):
        self.archive("MaaShiningNikki-win-x86_64-v2.zip", {"a.txt": "a", "A.txt": "b"})
        with patch.object(updates, "request", return_value=io.BytesIO(b"[]")):
            with self.assertRaises(updates.PackageCaseCollision):
                updates.generate(self.root, "owner/repo", "v2", 3, "")

    def test_exact_duplicates_after_case_collision_still_fail_release(self):
        archive = self.archive("MaaShiningNikki-linux-x86_64-v2.zip", {"a.txt": "a", "A.txt": "b"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(archive, "a") as package:
                package.writestr("a.txt", "duplicate")
        with patch.object(updates, "request", return_value=io.BytesIO(b"[]")):
            with self.assertRaisesRegex(ValueError, "Duplicate package path"):
                updates.generate(self.root, "owner/repo", "v2", 3, "")

    def test_unsafe_symlink_after_case_collision_still_fails_release(self):
        archive = self.archive("MaaShiningNikki-linux-x86_64-v2.zip", {"a.txt": "a", "A.txt": "b"})
        with zipfile.ZipFile(archive, "a") as package:
            info = zipfile.ZipInfo("python/bin/python3")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            package.writestr(info, "../../../outside")
        with patch.object(updates, "request", return_value=io.BytesIO(b"[]")):
            with self.assertRaisesRegex(ValueError, "Symlink escapes package"):
                updates.generate(self.root, "owner/repo", "v2", 3, "")

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

    def test_actions_layout_keeps_release_clean_and_three_baselines(self):
        name = "MaaShiningNikki-win-x86_64-v4.zip"
        archive = self.archive(name, {"unchanged.bin": bytes(range(256)) * 4096, "change": "new"})
        previous = self.root / "previous"
        previous.mkdir()
        releases = []
        fixtures = {}
        for tag in ("v3", "v2", "v1", "v0"):
            old = self.archive(f"MaaShiningNikki-win-x86_64-{tag}.zip",
                               {"unchanged.bin": bytes(range(256)) * 4096, "change": tag})
            target = previous / old.name
            old.rename(target)
            fixtures[old.name] = target
            releases.append({"draft": False, "tag_name": tag, "assets": [
                {"name": old.name, "browser_download_url": "mock"}]})
        artifacts = self.root / "actions"
        with patch.object(updates, "request", return_value=io.BytesIO(json.dumps(releases).encode())), \
                patch.object(updates, "actions_baseline", return_value={}), \
                patch.object(updates, "download", side_effect=lambda asset, target, token: target.write_bytes(fixtures[asset["name"]].read_bytes())), \
                patch.dict("os.environ", {"GITHUB_SHA": "abc123", "GITHUB_RUN_ID": "42"}):
            index = updates.generate(self.root, "owner/repo", "v4", 3, "token", artifacts)
        self.assertEqual({d["baseVersion"] for d in index["deltas"]}, {"v1", "v2", "v3"})
        self.assertEqual(index["source"], {"repository": "owner/repo", "headSha": "abc123", "runId": "42"})
        self.assertEqual(sorted(p.name for p in self.root.iterdir() if p.is_file()), [name, "checksums.sha256"])
        self.assertEqual((self.root / "checksums.sha256").read_text(), f"{updates.file_hash(archive)}  {name}\n")
        self.assertTrue((artifacts / "metadata" / updates.manifest_asset_name(name)).exists())
        self.assertEqual(json.loads((artifacts / "metadata" / "update-index.json").read_text()), index)
        self.assertEqual({p.name for p in (artifacts / "deltas").iterdir()}, {d["asset"] for d in index["deltas"]})

    def test_actions_baseline_avoids_full_historical_download(self):
        old = self.archive("old.zip", {"unchanged.bin": bytes(range(256)) * 4096, "change": "old"})
        manifest = updates.archive_manifest(old, "v1", "win-x86_64")
        old.unlink()
        name = "MaaShiningNikki-win-x86_64-v1.zip"
        self.archive("MaaShiningNikki-win-x86_64-v2.zip", {"unchanged.bin": bytes(range(256)) * 4096, "change": "new"})
        releases = [{"draft": False, "tag_name": "v1", "assets": [{"name": name}]}]
        with patch.object(updates, "request", return_value=io.BytesIO(json.dumps(releases).encode())), \
                patch.object(updates, "actions_baseline", return_value={updates.manifest_asset_name(name): manifest}), \
                patch.object(updates, "download") as download:
            index = updates.generate(self.root, "owner/repo", "v2", 3, "token", self.root / "actions")
        download.assert_not_called()
        self.assertEqual(len(index["deltas"]), 1)

    def test_actions_download_redirect_strips_token(self):
        req = urllib.request.Request("https://api.github.com/repos/owner/repo/actions/artifacts/1/zip",
                                     headers={"Authorization": "Bearer secret", "User-Agent": "test"})
        redirected = updates.StripAuthorizationRedirect().redirect_request(
            req, None, 302, "Found", {}, "https://storage.example/artifact.zip")
        self.assertFalse(redirected.has_header("Authorization"))
        self.assertEqual(redirected.get_header("User-agent"), "test")

    def test_reads_and_verifies_historical_actions_metadata(self):
        name = updates.manifest_asset_name("MaaShiningNikki-win-x86_64-v1.zip")
        index = {"schema": 1, "version": "v1", "source": {
            "repository": "owner/repo", "headSha": "commit", "runId": "42"}}
        manifests = {name: {"schema": 1, "version": "v1", "platform": "win-x86_64", "files": []}}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as package:
            package.writestr("update-index.json", json.dumps(index))
            package.writestr(name, json.dumps(manifests[name]))
        content = buffer.getvalue()
        import hashlib
        artifact = {"id": 1, "name": updates.metadata_artifact_name("v1"), "expired": False,
                    "digest": "sha256:" + hashlib.sha256(content).hexdigest(),
                    "workflow_run": {"id": 42, "head_sha": "commit"}}
        workflow = {"status": "completed", "conclusion": "success", "event": "push", "head_branch": "v1",
                    "head_sha": "commit", "path": ".github/workflows/install.yml", "head_repository": {"full_name": "owner/repo"}}

        def fake_request(url, token, **kwargs):
            if "/git/ref/tags/" in url:
                value = {"object": {"type": "commit", "sha": "commit"}}
            elif "/actions/artifacts?" in url:
                value = {"artifacts": [artifact]}
            else:
                value = workflow
            return io.BytesIO(json.dumps(value).encode())

        with patch.object(updates, "request", side_effect=fake_request), \
                patch.object(updates.urllib.request, "build_opener") as opener:
            opener.return_value.open.side_effect = lambda *args, **kwargs: io.BytesIO(content)
            self.assertEqual(updates.actions_baseline("owner/repo", "v1", "token", self.root), manifests)
            workflow["conclusion"] = "failure"
            self.assertEqual(updates.actions_baseline("owner/repo", "v1", "token", self.root), {})
            workflow["conclusion"] = "success"
            artifact["digest"] = "sha256:" + "0" * 64
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                updates.actions_baseline("owner/repo", "v1", "token", self.root)

    def test_history_unavailable_still_publishes_full_package(self):
        archive = self.archive("MaaShiningNikki-win-x86_64-v2.zip", {"resource/test.json": "data"})
        with patch.object(updates, "request", side_effect=urllib.error.URLError("offline")):
            index = updates.generate(self.root, "owner/repo", "v2", 3, "")
        self.assertEqual(index["deltas"], [])
        with zipfile.ZipFile(archive) as package:
            self.assertIn(updates.MANIFEST, package.namelist())
        self.assertIn(updates.file_hash(archive), (self.root / "checksums.sha256").read_text())

    def test_rerun_download_filter_excludes_previous_incremental_outputs(self):
        workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/install.yml").read_text(encoding="utf-8")
        download = workflow.split("      - uses: actions/download-artifact@v8\n", 1)[1].split("\n      - ", 1)[0]
        pattern = next(line.split("pattern:", 1)[1].strip().strip("\"'")
                       for line in download.splitlines() if "pattern:" in line)
        platforms = ["MaaShiningNikki-win-x86_64", "MaaShiningNikki-linux-x86_64", "MaaShiningNikki-macos-aarch64"]
        # A previous attempt's metadata/payloads coexist with the build inputs.
        available = [*platforms, updates.metadata_artifact_name("v2"), "delta-" + "a" * 24]
        selected = fnmatch.filter(available, pattern)
        self.assertEqual(selected, platforms)
        for name in selected:
            self.archive(name + "-v2.zip", {"agent/main.py": "script"})
        with patch.object(updates, "request", return_value=io.BytesIO(b"[]")):
            updates.generate(self.root, "owner/repo", "v2", 3, "", self.root / "actions")
        self.assertEqual({path.name for path in self.root.glob("*.zip")}, {name + "-v2.zip" for name in platforms})

    def test_history_list_body_timeout_still_publishes_full_package(self):
        archive = self.archive("MaaShiningNikki-win-x86_64-v2.zip", {"agent/main.py": "script"})
        with patch.object(updates, "request", side_effect=TimeoutError("history body stalled")):
            index = updates.generate(self.root, "owner/repo", "v2", 3, "token", self.root / "actions")
        self.assertEqual(index["deltas"], [])
        self.assertIn(updates.file_hash(archive), (self.root / "checksums.sha256").read_text())

    def test_actions_metadata_timeout_falls_back_to_historical_full_zip(self):
        old = self.archive("old.zip", {"unchanged.bin": bytes(range(256)) * 4096, "change": "old"})
        original = old.read_bytes()
        old.unlink()
        self.archive("MaaShiningNikki-win-x86_64-v2.zip", {"unchanged.bin": bytes(range(256)) * 4096, "change": "new"})
        releases = [{"draft": False, "tag_name": "v1", "assets": [{"name": "MaaShiningNikki-win-x86_64-v1.zip"}]}]
        with patch.object(updates, "request", return_value=io.BytesIO(json.dumps(releases).encode())), \
                patch.object(updates, "actions_baseline", side_effect=TimeoutError("metadata body stalled")), \
                patch.object(updates, "download", side_effect=lambda asset, target, token: target.write_bytes(original)):
            index = updates.generate(self.root, "owner/repo", "v2", 3, "token", self.root / "actions")
        self.assertEqual(len(index["deltas"]), 1)
        self.assertTrue((self.root / "checksums.sha256").exists())

    def test_historical_stream_errors_preserve_full_release(self):
        errors = [TimeoutError("stalled"), ConnectionResetError("reset"), http.client.IncompleteRead(b"", 10)]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                assets = self.root / type(error).__name__
                assets.mkdir()
                archive = assets / "MaaShiningNikki-win-x86_64-v2.zip"
                with zipfile.ZipFile(archive, "w") as package:
                    package.writestr("agent/main.py", "script")
                releases = [{"draft": False, "tag_name": "v1", "assets": [{"name": "MaaShiningNikki-win-x86_64-v1.zip"}]}]
                with patch.object(updates, "request", return_value=io.BytesIO(json.dumps(releases).encode())), \
                        patch.object(updates, "actions_baseline", side_effect=error), \
                        patch.object(updates, "download", side_effect=error):
                    index = updates.generate(assets, "owner/repo", "v2", 3, "token", assets / "actions")
                self.assertEqual(index["deltas"], [])
                self.assertTrue((assets / "actions/metadata/update-index.json").exists())
                self.assertIn(updates.file_hash(archive), (assets / "checksums.sha256").read_text())

    def test_history_budget_is_shared_and_preserves_completed_deltas(self):
        clock = [0.0]
        archives = [self.archive(f"MaaShiningNikki-win-{arch}-v4.zip", {
            "unchanged.bin": bytes(range(256)) * 4096, "change": "new"}) for arch in ("aarch64", "x86_64")]
        old = self.archive("old.zip", {"unchanged.bin": bytes(range(256)) * 4096, "change": "old"})
        content = old.read_bytes()
        old.unlink()
        releases = [{"draft": False, "tag_name": tag, "assets": [
            {"name": f"MaaShiningNikki-win-{arch}-{tag}.zip"} for arch in ("aarch64", "x86_64")
        ]} for tag in ("v3", "v2", "v1")]
        calls = []

        def download(asset, path, token):
            calls.append(asset["name"])
            clock[0] += 0.4 if len(calls) == 1 else 0.7
            path.write_bytes(content)

        with patch.object(updates.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(updates, "request", return_value=io.BytesIO(json.dumps(releases).encode())), \
                patch.object(updates, "actions_baseline", return_value={}), \
                patch.object(updates, "download", side_effect=download):
            index = updates.generate(self.root, "owner/repo", "v4", 3, "token", self.root / "actions", history_timeout=1)
        self.assertEqual(len(calls), 2)
        self.assertEqual([d["baseVersion"] for d in index["deltas"]], ["v3"])
        for archive in archives:
            with zipfile.ZipFile(archive) as package:
                self.assertIn(updates.MANIFEST, package.namelist())
            self.assertIn(updates.file_hash(archive), (self.root / "checksums.sha256").read_text())
        self.assertTrue((self.root / "actions/deltas" / index["deltas"][0]["asset"]).exists())

    def test_exhausted_budget_starts_no_network_requests(self):
        archive = self.archive("MaaShiningNikki-win-x86_64-v2.zip", {"agent/main.py": "script"})
        with patch.object(updates, "request") as request:
            result = updates.generate(self.root, "owner/repo", "v2", 3, "token", history_timeout=0)
        request.assert_not_called()
        self.assertEqual(result["deltas"], [])
        self.assertIn(updates.file_hash(archive), (self.root / "checksums.sha256").read_text())

    def test_budget_timer_is_restored_after_wrapped_network_timeout(self):
        handlers = []
        def install(signum, handler):
            handlers.append(handler)
            return "previous-handler"
        with patch.object(updates.signal, "SIGALRM", 14, create=True), \
                patch.object(updates.signal, "ITIMER_REAL", 0, create=True), \
                patch.object(updates.signal, "signal", side_effect=install), \
                patch.object(updates.signal, "setitimer", create=True) as timer:
            budget = updates.HistoryBudget(600)
            with self.assertRaises(urllib.error.URLError):
                with budget.operation():
                    try:
                        handlers[0]()
                    except TimeoutError as error:
                        raise urllib.error.URLError(error)
            self.assertEqual(budget.remaining, 0)
            self.assertEqual(timer.call_args_list[0].args, (0, 600))
            self.assertEqual(timer.call_args_list[-1].args, (0, 0))
            self.assertEqual(handlers[-1], "previous-handler")
            with self.assertRaises(updates.HistoryBudgetExceeded):
                with budget.operation():
                    self.fail("Exhausted budget entered another operation")

    def test_wrapped_metadata_deadline_never_starts_full_history_download(self):
        self.archive("MaaShiningNikki-win-x86_64-v2.zip", {"agent/main.py": "new"})
        releases = [{"draft": False, "tag_name": "v1", "assets": [{"name": "MaaShiningNikki-win-x86_64-v1.zip"}]}]
        handler = [None]
        def install(signum, callback):
            old = handler[0]
            handler[0] = callback
            return old
        def metadata(*args):
            try:
                handler[0]()
            except TimeoutError as error:
                raise urllib.error.URLError(error)
        with patch.object(updates.signal, "SIGALRM", 14, create=True), \
                patch.object(updates.signal, "ITIMER_REAL", 0, create=True), \
                patch.object(updates.signal, "signal", side_effect=install), \
                patch.object(updates.signal, "setitimer", create=True), \
                patch.object(updates, "request", return_value=io.BytesIO(json.dumps(releases).encode())), \
                patch.object(updates, "actions_baseline", side_effect=metadata), \
                patch.object(updates, "download") as download:
            result = updates.generate(self.root, "owner/repo", "v2", 3, "token", self.root / "actions")
        download.assert_not_called()
        self.assertEqual(result["deltas"], [])
        self.assertTrue((self.root / "checksums.sha256").exists())

    @unittest.skipUnless(hasattr(updates.signal, "setitimer"), "POSIX timer runs in Ubuntu release CI")
    def test_budget_interrupts_trickling_http_body(self):
        import socket
        import threading
        import time
        left, right = socket.socketpair()
        left.settimeout(1)
        def send():
            try:
                right.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n")
                for _ in range(1000):
                    right.sendall(b"x")
                    time.sleep(0.02)
            except OSError:
                pass
            finally:
                right.close()
        thread = threading.Thread(target=send, daemon=True)
        thread.start()
        started = time.monotonic()
        try:
            with self.assertRaises(updates.HistoryBudgetExceeded):
                with updates.HistoryBudget(0.1).operation():
                    response = http.client.HTTPResponse(left)
                    try:
                        response.begin()
                        response.read(1024 * 1024)
                    finally:
                        response.close()
            self.assertLess(time.monotonic() - started, 1)
        finally:
            left.close()
            thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
