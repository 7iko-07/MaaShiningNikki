"""Generate verified file manifests and delta ZIPs from final release archives.

No source-tree comparison: Python, native libraries and generated interface.json
are compared exactly as shipped. Existing clients still select the full ZIP.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import signal
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

MANIFEST = "update-manifest.json"
PROTECTED = {"config", "logs", "debug", "temp", "backup", "__pycache__"}
# History is optional. Stream reads can raise these directly, without urlopen
# wrapping them in URLError. Do not catch unrelated filesystem errors here.
HISTORY_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException,
                  ValueError, zipfile.BadZipFile, KeyError)


class HistoryBudgetExceeded(TimeoutError):
    pass


class HistoryBudget:
    """One cumulative budget for optional history work across all platforms.

    Release CI runs on Ubuntu: ITIMER_REAL interrupts even DNS/header waits and
    trickling HTTP bodies. On hosts without POSIX timers the boundary checks
    still account for elapsed work, alongside urllib's per-operation timeout.
    """

    def __init__(self, seconds: float = 600):
        self.remaining = seconds

    @contextmanager
    def operation(self):
        if self.remaining <= 0:
            raise HistoryBudgetExceeded("Historical update budget exhausted")
        started = time.monotonic()
        alarm = hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()
        previous = None

        def expired(*_):
            self.remaining = 0
            raise HistoryBudgetExceeded("Historical update budget exhausted")

        if alarm:
            previous = signal.signal(signal.SIGALRM, expired)
            signal.setitimer(signal.ITIMER_REAL, self.remaining)
        try:
            yield
            if time.monotonic() - started >= self.remaining:
                expired()
        except HistoryBudgetExceeded:
            self.remaining = 0
            raise
        finally:
            if alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
            self.remaining = max(0, self.remaining - (time.monotonic() - started))


class PackageCaseCollision(ValueError):
    """Distinct ZIP paths cannot be represented by the current update client."""


def managed(path: str) -> bool:
    parts = path.split("/")
    return (bool(path) and not path.startswith("/") and "\\" not in path and ":" not in path
            and all(p not in {"", ".", ".."} and not p.endswith((".", " ")) for p in parts)
            and not any(p.lower() in PROTECTED for p in parts)
            and path.lower() != MANIFEST
            and path.lower() != "appsettings.json"
            and not path.lower().startswith(".mfa-update")
            and not path.lower().endswith((".backupmfa", ".pyc", ".pending_update", ".sqlite3", ".sqlite3-wal", ".sqlite3-shm")))


def archive_manifest(archive: Path, version: str, platform: str) -> dict:
    files = []
    seen = {}
    exact_paths = set()
    case_collision = None
    with zipfile.ZipFile(archive) as package:
        for entry in package.infolist():
            if entry.is_dir() or not managed(entry.filename):
                continue
            key = entry.filename.lower()
            if entry.filename in exact_paths:
                raise ValueError(f"Duplicate package path: {entry.filename}")
            exact_paths.add(entry.filename)
            if key in seen and case_collision is None:
                case_collision = f"Package paths differ only by case: {seen[key]} and {entry.filename}"
            seen[key] = entry.filename
            mode = entry.external_attr >> 16
            if stat.S_ISLNK(mode):
                # Symlink support needs an explicit contract, rather than silently
                # hashing the destination or installing the link as a text file.
                link = package.read(entry).decode("utf-8")
                if link.startswith("/") or "\\" in link:
                    raise ValueError(f"Unsafe package symlink: {entry.filename}")
                normalized = os.path.normpath(str(PurePosixPath(entry.filename).parent / link)).replace("\\", "/")
                if not managed(normalized):
                    raise ValueError(f"Symlink escapes package: {entry.filename}")
            digest = hashlib.sha256()
            with package.open(entry) as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            record = {"path": entry.filename, "size": entry.file_size,
                      "sha256": digest.hexdigest(), "mode": mode & 0o777 or 0o644}
            if stat.S_ISLNK(mode):
                record["linkTarget"] = link
            files.append(record)
    # Validate every entry before allowing a full-package fallback; exact
    # duplicates and unsafe symlinks must still fail the release.
    if case_collision is not None:
        raise PackageCaseCollision(case_collision)
    files.sort(key=lambda f: f["path"])
    return {"schema": 1, "version": version, "platform": platform, "kind": "full",
            "files": files, "baseFiles": [], "payload": [f["path"] for f in files], "deleted": []}


def delta_manifest(old: dict, new: dict) -> dict:
    if old["schema"] != 1 or old["platform"] != new["platform"]:
        raise ValueError("Incompatible baseline manifest")
    previous = {f["path"]: f for f in old["files"]}
    target = {f["path"]: f for f in new["files"]}
    payload = [path for path, f in target.items()
               if path not in previous or any(f.get(k) != previous[path].get(k)
                                              for k in ("sha256", "mode", "linkTarget"))]
    return {**new, "kind": "delta", "baseVersion": old["version"], "baseFiles": old["files"],
            "payload": payload, "deleted": sorted(set(previous) - set(target))}


def write_delta(archive: Path, target: Path, manifest: dict) -> None:
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
        for path in manifest["payload"]:
            info = source.getinfo(path)
            with source.open(info) as reader, output.open(info, "w", force_zip64=True) as writer:
                while chunk := reader.read(1024 * 1024):
                    writer.write(chunk)


def request(url: str, token: str, *, api: bool = False):
    headers = {"User-Agent": "MFA-incremental-release", "Accept": "application/vnd.github+json"}
    # Never forward the repository token to arbitrary redirected asset URLs.
    if api and token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)


def download(asset: dict, target: Path, token: str) -> None:
    with request(asset["browser_download_url"], token) as response, target.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    digest = asset.get("digest") or ""
    if digest.startswith("sha256:") and file_hash(target) != digest[7:]:
        raise ValueError(f"Release asset checksum mismatch: {asset['name']}")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest_asset_name(archive_name: str) -> str:
    return "manifest-" + hashlib.sha256(archive_name.encode()).hexdigest()[:24] + ".json"


def metadata_artifact_name(version: str) -> str:
    return "update-metadata-" + hashlib.sha256(version.encode()).hexdigest()[:24]


class StripAuthorizationRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
        return redirected


def actions_baseline(repository: str, version: str, token: str, cache: Path) -> dict:
    """Read small historical manifests from Actions; callers can use the full ZIP if unavailable."""
    if not token:
        return {}
    api = f"https://api.github.com/repos/{repository}"
    with request(f"{api}/git/ref/tags/{urllib.parse.quote(version, safe='')}", token, api=True) as response:
        target = json.load(response)["object"]
    for _ in range(8):
        if target["type"] == "commit":
            break
        if target["type"] != "tag":
            raise ValueError("Unsupported release tag target")
        with request(f"{api}/git/tags/{target['sha']}", token, api=True) as response:
            target = json.load(response)["object"]
    if target["type"] != "commit":
        raise ValueError("Release tag nesting exceeded")
    name = metadata_artifact_name(version)
    page = 1
    while True:
        with request(f"{api}/actions/artifacts?name={name}&per_page=100&page={page}", token, api=True) as response:
            artifacts = json.load(response)["artifacts"]
        for artifact in artifacts:
            run = artifact.get("workflow_run") or {}
            if artifact["name"] != name or artifact["expired"] or run.get("head_sha") != target["sha"]:
                continue
            with request(f"{api}/actions/runs/{run['id']}", token, api=True) as response:
                workflow = json.load(response)
            if (workflow.get("status") != "completed" or workflow.get("conclusion") != "success"
                    or workflow.get("event") not in {"push", "workflow_dispatch"}
                    or workflow.get("head_branch") != version or workflow.get("head_sha") != target["sha"]
                    or workflow.get("path", "").split("@")[0] != ".github/workflows/install.yml"
                    or workflow.get("head_repository", {}).get("full_name", "").lower() != repository.lower()):
                continue
            path = cache / f"{name}.zip"
            # urllib normally retains Authorization across redirects. Strip it
            # before following the temporary storage URL returned by GitHub.
            opener = urllib.request.build_opener(StripAuthorizationRedirect())
            req = urllib.request.Request(f"{api}/actions/artifacts/{artifact['id']}/zip", headers={
                "User-Agent": "MFA-incremental-release", "Authorization": f"Bearer {token}"})
            with opener.open(req, timeout=120) as response, path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            digest = artifact.get("digest") or ""
            if digest.startswith("sha256:") and file_hash(path) != digest[7:]:
                raise ValueError("Actions metadata checksum mismatch")
            with zipfile.ZipFile(path) as package:
                index = json.loads(package.read("update-index.json"))
                source = index.get("source", {})
                if (index.get("schema") != 1 or index.get("version") != version
                        or source.get("repository", "").lower() != repository.lower()
                        or source.get("headSha") != target["sha"] or source.get("runId") != str(run["id"])):
                    raise ValueError("Actions metadata does not match release")
                return {entry.filename: json.loads(package.read(entry)) for entry in package.infolist()
                        if re.fullmatch(r"manifest-[0-9a-f]{24}\.json", entry.filename)}
        if len(artifacts) < 100:
            return {}
        page += 1


def package_identity(name: str) -> tuple[str, str]:
    match = re.search(r"-(win|linux|macos|osx)-(x86_64|x64|arm64|aarch64)(?:-|\.)", name)
    if not match:
        raise ValueError(f"Package platform missing: {name}")
    return name.split("-", 1)[0], f"{match[1]}-{match[2]}"


def generate(assets: Path, repository: str, version: str, history: int, token: str,
             artifacts: Path | None = None, history_timeout: float = 600) -> dict:
    metadata = artifacts / "metadata" if artifacts is not None else assets
    deltas = artifacts / "deltas" if artifacts is not None else assets
    metadata.mkdir(parents=True, exist_ok=True)
    deltas.mkdir(parents=True, exist_ok=True)
    budget = HistoryBudget(history_timeout)
    try:
        with budget.operation(), request(f"https://api.github.com/repos/{repository}/releases?per_page=100", token, api=True) as response:
            releases = [r for r in json.load(response) if not r["draft"] and r["tag_name"] != version][:history]
    except HISTORY_ERRORS as error:
        print(f"::warning::Cannot read historical releases; publishing full packages only: {error}")
        releases = []
    cache = assets / ".incremental-cache"
    cache.mkdir(exist_ok=True)
    index = {"schema": 1, "version": version, "deltas": []}
    historical_manifests = {}
    if artifacts is not None:
        index["source"] = {"repository": repository, "headSha": os.getenv("GITHUB_SHA", ""),
                           "runId": os.getenv("GITHUB_RUN_ID", "")}
    for archive in sorted(assets.glob("*.zip")):
        if archive.name.startswith("delta-"):
            continue
        product, platform = package_identity(archive.name)
        try:
            new = archive_manifest(archive, version, platform)
        except PackageCaseCollision as error:
            if not platform.startswith(("linux-", "macos-", "osx-")):
                raise
            print(f"::warning::Publishing full package only for {archive.name}; "
                  f"incremental updates skipped: {error}")
            continue
        with zipfile.ZipFile(archive, "a", zipfile.ZIP_DEFLATED) as package:
            if MANIFEST in package.namelist():
                raise ValueError(f"Manifest already present: {archive.name}")
            package.writestr(MANIFEST, json.dumps(new, ensure_ascii=False, indent=2))
        (metadata / manifest_asset_name(archive.name)).write_text(
            json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
        full_hash = file_hash(archive)
        for release in releases:
            if budget.remaining <= 0:
                break
            old_assets = release.get("assets", [])
            candidates = []
            for asset in old_assets:
                if asset["name"].endswith(".zip") and not asset["name"].startswith("delta-"):
                    try:
                        if package_identity(asset["name"]) == (product, platform):
                            candidates.append(asset)
                    except ValueError:
                        pass
            if len(candidates) != 1:
                continue
            old_asset = candidates[0]
            sidecar = next((a for a in old_assets if a["name"] == manifest_asset_name(old_asset["name"])), None)
            try:
                with budget.operation():
                    if sidecar:
                        old_path = cache / sidecar["name"]
                        download(sidecar, old_path, token)
                        old = json.loads(old_path.read_text(encoding="utf-8"))
                    else:
                        tag = release["tag_name"]
                        if artifacts is not None and tag not in historical_manifests:
                            try:
                                historical_manifests[tag] = actions_baseline(repository, tag, token, cache)
                            except HistoryBudgetExceeded:
                                raise
                            except HISTORY_ERRORS as error:
                                if budget.remaining <= 0:
                                    raise HistoryBudgetExceeded("Historical update budget exhausted") from error
                                print(f"::warning::Cannot read Actions baseline {tag}; using full package: {error}")
                                historical_manifests[tag] = {}
                        old = historical_manifests.get(tag, {}).get(manifest_asset_name(old_asset["name"]))
                        if old is None:
                            old_path = cache / old_asset["name"]
                            download(old_asset, old_path, token)
                            old = archive_manifest(old_path, tag, platform)
                    if old["version"] != release["tag_name"]:
                        raise ValueError("Baseline manifest version mismatch")
                delta = delta_manifest(old, new)
                # No OS/architecture in delta asset names: old clients prefer full packages.
                identifier = hashlib.sha256(f"{archive.name}\0{old['version']}".encode()).hexdigest()[:24]
                delta_path = deltas / f"delta-{identifier}.zip"
                write_delta(archive, delta_path, delta)
                if delta_path.stat().st_size >= archive.stat().st_size:
                    delta_path.unlink()
                    continue
                index["deltas"].append({"fullAsset": archive.name, "asset": delta_path.name,
                                         "baseVersion": old["version"], "version": version, "platform": platform,
                                         "sha256": file_hash(delta_path), "size": delta_path.stat().st_size,
                                         "fullSha256": full_hash})
            except HISTORY_ERRORS as error:
                # Historical packages are optional; a failed baseline never blocks the full release.
                print(f"::warning::Skipping baseline {release['tag_name']} {platform}: {error}")
    (metadata / "update-index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    with (assets / "checksums.sha256").open("w", encoding="utf-8") as output:
        for path in sorted(assets.iterdir()):
            if path.is_file() and path.name != "checksums.sha256":
                output.write(f"{file_hash(path)}  {path.name}\n")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--artifacts", type=Path, help="Store incremental packages and metadata outside Release assets")
    args = parser.parse_args()
    generate(args.assets.resolve(), args.repository, args.version, args.history, os.getenv("GITHUB_TOKEN", ""),
             args.artifacts.resolve() if args.artifacts else None)
