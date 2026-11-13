"""Pinned, bounded Safetensors header reads without weight payload requests."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .streaming import (
    MAX_SOURCE_TENSORS,
    ConversionError,
    TensorInfo,
    parse_safetensors_header,
    safetensors_header_records,
)

MAX_HEADER_BYTES = 4 * 1024 * 1024
MAX_API_BYTES = 16 * 1024 * 1024
INDEX = "model.safetensors.index.json"
_RANGE = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)")


class HeaderAccessError(Exception):
    """A complete, pinned header could not be fetched within the read budget."""


class HeaderBudgetError(HeaderAccessError):
    """The tensor header exceeds the catalogue's inspection budget."""


def _storage_location(response):
    """Return the storage URL a pinned Hub request was redirected to, if any."""
    if not getattr(response, "history", None):
        return None
    location = str(getattr(response, "url", "") or "")
    return location if location.startswith("https://") else None


def _fetch_range(hub, url, start, end, check_deadline, expected_total=None, *, storage=False):
    check_deadline()
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    if not storage:
        # Hub credentials are for the Hub endpoint only. A storage location is
        # already authorized by its own signature and never receives them.
        headers = {**hub.utils.build_hf_headers(), **headers}
    with hub.utils.get_session().stream(
        "GET", url, headers=headers, timeout=5, follow_redirects=not storage
    ) as response:
        content_range = _RANGE.fullmatch(response.headers.get("content-range", ""))
        if (
            response.status_code != 206
            or content_range is None
            or response.headers.get("content-encoding", "identity") != "identity"
        ):
            raise HeaderAccessError()
        first, last, total = map(int, content_range.groups())
        if (first, last) != (start, end) or not end < total <= 2**63 - 1 or expected_total not in (None, total):
            raise HeaderAccessError()
        data = bytearray()
        for chunk in response.iter_raw(chunk_size=8192):
            check_deadline()
            if len(data) + len(chunk) > end - start + 1:
                raise HeaderAccessError()
            data.extend(chunk)
        if len(data) != end - start + 1:
            raise HeaderAccessError()
        location = None if storage else _storage_location(response)
    return bytes(data), total, location


def read_range(hub, url, start, end, check_deadline, expected_total=None):
    data, total, _ = _fetch_range(hub, url, start, end, check_deadline, expected_total)
    return data, total


def read_header(hub, repo, revision, name, remaining, size=None, *, maximum=MAX_HEADER_BYTES):
    url = hub.hf_hub_url(repo_id=repo, filename=name, revision=revision)
    prefix, total, location = _fetch_range(hub, url, 0, 7, remaining, size)
    length = int.from_bytes(prefix, "little")
    if length > maximum:
        raise HeaderBudgetError()
    if length < 2 or 8 + length >= total:
        raise ConversionError("Invalid safetensors header size")
    data = None
    if location is not None:
        # The length prefix already resolved this pinned file to its storage
        # location. Reading the header there saves the second Hub redirect; the
        # exact range and the file size seen through the Hub are still enforced.
        try:
            data, _, _ = _fetch_range(hub, location, 8, 7 + length, remaining, total, storage=True)
        except Exception:
            data = None
    if data is None:
        data, _, _ = _fetch_range(hub, url, 8, 7 + length, remaining, total)
    _, headers = parse_safetensors_header(data, total)
    return {
        key: TensorInfo(Path(name), dtype, shape, offset, nbytes, (0, 0, total, 0))
        for key, (dtype, shape, offset, nbytes) in headers.items()
    }


def indexed_headers(hub, repo, revision, shards, sizes, remaining, *, maximum=MAX_HEADER_BYTES):
    """Return every indexed shard's header from one Hub API call, or ``None``.

    The Hub parses an indexed checkpoint's shard headers itself and labels the
    reply with the commit it describes. The reply is used only when that commit
    is the pinned revision and each header passes the same record checks as a
    ranged read, against the pinned file sizes. Anything else returns ``None``
    so the caller reads exact header ranges and reports from those bytes.
    """
    endpoint = getattr(getattr(hub, "constants", None), "ENDPOINT", None)
    if not isinstance(endpoint, str) or not shards or any(type(sizes.get(name)) is not int for name in shards):
        return None
    try:
        remaining()
        url = f"{endpoint}/api/models/{repo}/safetensors-metadata/{INDEX}"
        with hub.utils.get_session().stream(
            "GET", url, headers=hub.utils.build_hf_headers(), timeout=5, follow_redirects=True
        ) as response:
            described = response.headers.get("etag", "").removeprefix("W/").strip('"')
            if response.status_code != 200 or described != revision:
                return None
            body = bytearray()
            for chunk in response.iter_bytes():
                remaining()
                body.extend(chunk)
                if len(body) > MAX_API_BYTES:
                    return None
        parsed = json.loads(body)["headers"]
        headers = {}
        for name in shards:
            header = parsed[name]
            data_bytes = max((info["data_offsets"][1] for key, info in header.items() if key != "__metadata__"))
            # A valid file is its length prefix, header and contiguous tensor
            # data, so the pinned size fixes where the data begins.
            data_start = sizes[name] - data_bytes
            if not 10 <= data_start <= 8 + maximum:
                return None
            _, records = safetensors_header_records(header, data_start, sizes[name])
            headers[name] = {
                key: TensorInfo(Path(name), dtype, shape, offset, nbytes, (0, 0, sizes[name], 0))
                for key, (dtype, shape, offset, nbytes) in records.items()
            }
        return headers
    except Exception:
        return None


def checkpoint_headers(hub, repo, revision, shards, index, sizes, remaining):
    def read(name):
        return read_header(hub, repo, revision, name, remaining, sizes.get(name))

    headers = indexed_headers(hub, repo, revision, shards, sizes, remaining) if index is not None else None
    if headers is None:
        with ThreadPoolExecutor(max_workers=4) as pool:
            headers = dict(zip(shards, pool.map(read, shards), strict=True))
    tensors = {}
    for name in shards:
        remaining()
        header = headers[name]
        if tensors.keys() & header.keys():
            raise ConversionError("Tensor occurs in more than one shard")
        if index is not None and any(index.get(key) != name for key in header):
            raise ConversionError("Checkpoint shard index disagrees with its tensor headers")
        tensors.update(header)
        if len(tensors) > MAX_SOURCE_TENSORS:
            raise ConversionError("Tensor inventory exceeds the metadata planning budget")
    if index is not None and tensors.keys() != index.keys():
        raise ConversionError("Checkpoint shard index references missing tensors")
    return tensors
