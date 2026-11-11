"""Pinned, bounded Safetensors header reads without weight payload requests."""

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .streaming import MAX_SOURCE_TENSORS, ConversionError, TensorInfo, parse_safetensors_header

MAX_HEADER_BYTES = 4 * 1024 * 1024
_RANGE = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)")


class HeaderAccessError(Exception):
    """A complete, pinned header could not be fetched within the read budget."""


class HeaderBudgetError(HeaderAccessError):
    """The tensor header exceeds the catalogue's inspection budget."""


def read_range(hub, url, start, end, check_deadline, expected_total=None):
    check_deadline()
    headers = {
        **hub.utils.build_hf_headers(),
        "Range": f"bytes={start}-{end}",
        "Accept-Encoding": "identity",
    }
    with hub.utils.get_session().stream("GET", url, headers=headers, timeout=5, follow_redirects=True) as response:
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
    return bytes(data), total


def read_header(hub, repo, revision, name, remaining, size=None, *, maximum=MAX_HEADER_BYTES):
    url = hub.hf_hub_url(repo_id=repo, filename=name, revision=revision)
    prefix, total = read_range(hub, url, 0, 7, remaining, size)
    length = int.from_bytes(prefix, "little")
    if length > maximum:
        raise HeaderBudgetError()
    if length < 2 or 8 + length >= total:
        raise ConversionError("Invalid safetensors header size")
    data, _ = read_range(hub, url, 8, 7 + length, remaining, total)
    _, headers = parse_safetensors_header(data, total)
    return {
        key: TensorInfo(Path(name), dtype, shape, offset, nbytes, (0, 0, total, 0))
        for key, (dtype, shape, offset, nbytes) in headers.items()
    }


def checkpoint_headers(hub, repo, revision, shards, index, sizes, remaining):
    def read(name):
        return read_header(hub, repo, revision, name, remaining, sizes.get(name))

    tensors = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, header in zip(shards, pool.map(read, shards), strict=True):
            remaining()
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
