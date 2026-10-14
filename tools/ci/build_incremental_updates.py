"""Generate verified file manifests and delta ZIPs from final release archives.

No source-tree comparison: Python, native libraries and generated interface.json
are compared exactly as shipped. Existing clients still select the full ZIP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import urllib.error
import urllib.request
import zipfile

MANIFEST = "update-manifest.json"
PROTECTED = {"config", "logs", "debug", "temp", "backup", "__pycache__"}


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
    seen = set()
    with zipfile.ZipFile(archive) as package:
        for entry in package.infolist():
            if entry.is_dir() or not managed(entry.filename):
                continue
            key = entry.filename.lower()
            if key in seen:
                raise ValueError(f"Duplicate package path: {entry.filename}")
            seen.add(key)
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


def package_identity(name: str) -> tuple[str, str]:
    match = re.search(r"-(win|linux|macos|osx)-(x86_64|x64|arm64|aarch64)(?:-|\.)", name)
    if not match:
        raise ValueError(f"Package platform missing: {name}")
    return name.split("-", 1)[0], f"{match[1]}-{match[2]}"


def generate(assets: Path, repository: str, version: str, history: int, token: str) -> dict:
    try:
        with request(f"https://api.github.com/repos/{repository}/releases?per_page=100", token, api=True) as response:
            releases = [r for r in json.load(response) if not r["draft"] and r["tag_name"] != version][:history]
    except urllib.error.URLError as error:
        print(f"::warning::Cannot read historical releases; publishing full packages only: {error}")
        releases = []
    cache = assets / ".incremental-cache"
    cache.mkdir(exist_ok=True)
    index = {"schema": 1, "version": version, "deltas": []}
    for archive in sorted(assets.glob("*.zip")):
        if archive.name.startswith("delta-"):
            continue
        product, platform = package_identity(archive.name)
        new = archive_manifest(archive, version, platform)
        with zipfile.ZipFile(archive, "a", zipfile.ZIP_DEFLATED) as package:
            if MANIFEST in package.namelist():
                raise ValueError(f"Manifest already present: {archive.name}")
            package.writestr(MANIFEST, json.dumps(new, ensure_ascii=False, indent=2))
        (assets / manifest_asset_name(archive.name)).write_text(
            json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
        for release in releases:
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
                if sidecar:
                    old_path = cache / sidecar["name"]
                    download(sidecar, old_path, token)
                    old = json.loads(old_path.read_text(encoding="utf-8"))
                else:
                    old_path = cache / old_asset["name"]
                    download(old_asset, old_path, token)
                    old = archive_manifest(old_path, release["tag_name"], platform)
                if old["version"] != release["tag_name"]:
                    raise ValueError("Baseline manifest version mismatch")
                delta = delta_manifest(old, new)
                # No OS/architecture in delta asset names: old clients prefer full packages.
                identifier = hashlib.sha256(f"{archive.name}\0{old['version']}".encode()).hexdigest()[:24]
                delta_path = assets / f"delta-{identifier}.zip"
                write_delta(archive, delta_path, delta)
                if delta_path.stat().st_size >= archive.stat().st_size:
                    delta_path.unlink()
                    continue
                index["deltas"].append({"fullAsset": archive.name, "asset": delta_path.name,
                                         "baseVersion": old["version"], "version": version, "platform": platform,
                                         "sha256": file_hash(delta_path), "size": delta_path.stat().st_size})
            except (urllib.error.URLError, ValueError, zipfile.BadZipFile, KeyError) as error:
                # Historical packages are optional; a failed baseline never blocks the full release.
                print(f"::warning::Skipping baseline {release['tag_name']} {platform}: {error}")
    (assets / "update-index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
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
    args = parser.parse_args()
    generate(args.assets.resolve(), args.repository, args.version, args.history, os.getenv("GITHUB_TOKEN", ""))
