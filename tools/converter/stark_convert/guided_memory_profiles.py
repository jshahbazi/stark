"""Exact-artifact resident planning shared with the native request gate."""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class ResidentProfile:
    choice: str
    policy: str
    repo: str
    revision: str
    weight_bytes: int
    paged_weight_bytes: int
    maximum_context_tokens: int
    maximum_mtp_drafts: int
    request_headroom_divisor: int
    paged_working_set_bytes: int
    host_reserve_bytes: int


@lru_cache(maxsize=1)
def profiles():
    raw = json.loads(Path(__file__).with_suffix(".json").read_text())
    if raw.get("schema_version") != 1 or raw.get("policy") != "qwen38-paged-text-v1":
        raise ValueError("Unknown resident profile policy")
    if (
        raw.get("maximum_context_tokens") != 32768
        or raw.get("maximum_mtp_drafts") != 2
        or raw.get("request_headroom_divisor") != 1
        or raw.get("paged_working_set_bytes") != 1024**3
        or raw.get("host_reserve_bytes") != 8 * 1024**3
    ):
        raise ValueError("Changed resident profile bounds")
    return raw


def qualified_profile(model, choice):
    """Unknown, changed, linked or incomplete artifacts retain full accounting.

    Memory layout is bound by metadata hashes and complete payload sizes. This
    does not freshly rehash payload bytes or transfer numerical qualification.
    """
    try:
        raw = profiles()
        row = raw["profiles"][choice]
        if (
            sum(row["payloads"].values()) != row["weight_bytes"]
            or not 0 <= row["paged_weight_bytes"] <= row["weight_bytes"]
        ):
            return None
        root = Path(model)
        if root.is_symlink() or not root.is_dir():
            return None
        for name, digest in row["files_sha256"].items():
            path = root / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > 2 * 1024**2:
                return None
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                return None
        payloads = {path.name: path for path in root.iterdir() if path.suffix in {".safetensors", ".bin"}}
        if payloads.keys() != row["payloads"].keys():
            return None
        for name, size in row["payloads"].items():
            info = payloads[name].lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size != size:
                return None
        return ResidentProfile(
            choice,
            raw["policy"],
            row["repo"],
            row["revision"],
            row["weight_bytes"],
            row["paged_weight_bytes"],
            raw["maximum_context_tokens"],
            raw["maximum_mtp_drafts"],
            raw["request_headroom_divisor"],
            raw["paged_working_set_bytes"],
            raw["host_reserve_bytes"],
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
