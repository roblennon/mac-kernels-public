#!/usr/bin/env python3
"""Download pinned Hugging Face files and verify them against the Hub's own hashes.

Writes a `.mk-verified.json` manifest beside the files.
"""
import hashlib
import json
import urllib.request


def prepare(repo, revision, directory, selected, verify_only):
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true"
    with urllib.request.urlopen(url, timeout=30) as response:
        metadata = json.load(response)
    expected = {f["path"]: f for f in metadata if f["type"] == "file"}
    names = selected or tuple(expected)
    if not verify_only:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, revision=revision, local_dir=str(directory),
                          allow_patterns=list(names), max_workers=2)
    verified = []
    for name in names:
        path = directory / name
        entry = expected[name]
        before = path.stat()
        if before.st_size != entry["size"]:
            raise RuntimeError(f"Wrong downloaded size: {path}")
        sha256 = hashlib.sha256()
        blob_sha1 = hashlib.sha1(f"blob {before.st_size}\0".encode())
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024**2):
                sha256.update(chunk)
                blob_sha1.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"File changed during verification: {path}")
        if entry.get("lfs"):
            if sha256.hexdigest() != entry["lfs"]["oid"]:
                raise RuntimeError(f"Wrong LFS SHA256: {path}")
        elif blob_sha1.hexdigest() != entry["oid"]:
            raise RuntimeError(f"Wrong Git blob hash: {path}")
        verified.append({"path": name, "size": after.st_size, "mtime_ns": after.st_mtime_ns,
                         "sha256": sha256.hexdigest()})
        print(f"VERIFIED {repo}/{name} {sha256.hexdigest()}", flush=True)
    manifest = {"repo": repo, "revision": revision, "metadata_url": url, "files": verified}
    target = directory / ".mk-verified.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(target)
