"""Parse .gputrace capture bundles.

Reads MTLBuffer/MTLTexture files from .gputrace bundles, extracts resource
labels from device-resources binary files, and reads buffer data with
configurable struct layouts.
"""

from __future__ import annotations

import json
import os
import re
import stat
import struct
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_RESOURCE_NAME = re.compile(r"MTL(Buffer|Texture)-\d+-\d+")
_MAX_METADATA_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_METADATA_BYTES = 128 * 1024 * 1024
_MAX_STRINGS = 262_144
_MAX_METADATA_FILES = 1024


@contextmanager
def _capture_file(gputrace: Path, filename: str):
    """Read only a regular basename from the opened capture directory."""
    if not filename or Path(filename).name != filename or filename in {".", ".."} or "\\" in filename:
        raise ValueError("Invalid capture filename")
    directory = os.open(gputrace, os.O_RDONLY | os.O_DIRECTORY)
    try:
        descriptor = os.open(filename, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("Capture resources must be regular files")
        yield handle


def _resource_names(gputrace: Path) -> set[str]:
    result = set()
    for name in os.listdir(gputrace):
        if _RESOURCE_NAME.fullmatch(name):
            try:
                with _capture_file(gputrace, name):
                    result.add(name)
            except (OSError, ValueError):
                continue
    return result


def _resource_path(gputrace: Path, filename: str) -> None:
    if not _RESOURCE_NAME.fullmatch(filename):
        raise ValueError("Invalid capture resource filename")


def _metadata_files(gputrace: Path):
    total = 0
    names = [name for name in os.listdir(gputrace) if name.startswith("device-resources")]
    if len(names) > _MAX_METADATA_FILES:
        raise ValueError("Capture metadata file budget exceeded")
    for name in names:
        with _capture_file(gputrace, name) as handle:
            size = os.fstat(handle.fileno()).st_size
            total += size
            if size > _MAX_METADATA_BYTES or total > _MAX_TOTAL_METADATA_BYTES:
                raise ValueError("Capture metadata byte budget exceeded")
            data = handle.read(size + 1)
            if len(data) != size:
                raise ValueError("Capture metadata changed while reading")
            yield data


@dataclass(frozen=True)
class CaptureMetadata:
    uuid: str
    captured_frames: int
    graphics_api: str


@dataclass(frozen=True)
class CaptureResource:
    filename: str
    size_bytes: int
    label: str
    kind: str  # "buffer" or "texture"


@dataclass(frozen=True)
class ShaderLibrary:
    path: str


@dataclass(frozen=True)
class CaptureInfo:
    metadata: CaptureMetadata
    resources: tuple[CaptureResource, ...]
    shader_functions: tuple[str, ...]
    shader_libraries: tuple[ShaderLibrary, ...]


@dataclass(frozen=True)
class BufferElement:
    index: int
    fields: tuple[tuple[str, object], ...]  # (field_name, value)


# ── metadata ──────────────────────────────────────────────────────────────


def parse_metadata(gputrace: Path) -> CaptureMetadata:
    try:
        with _capture_file(gputrace, "metadata") as handle:
            if os.fstat(handle.fileno()).st_size > _MAX_METADATA_BYTES:
                raise ValueError("Capture metadata byte budget exceeded")
            metadata_bytes = handle.read(_MAX_METADATA_BYTES + 1)
            if len(metadata_bytes) > _MAX_METADATA_BYTES:
                raise ValueError("Capture metadata byte budget exceeded")
    except FileNotFoundError:
        return CaptureMetadata(uuid="", captured_frames=0, graphics_api="")

    result = subprocess.run(
        ["plutil", "-convert", "json", "-o", "-", "-"],
        input=metadata_bytes,
        capture_output=True,
    )
    if result.returncode != 0:
        return CaptureMetadata(uuid="", captured_frames=0, graphics_api="")

    data = json.loads(result.stdout)
    api_code = data.get("DYCaptureSession.graphics_api")
    api = {1: "Metal"}.get(api_code, f"Unknown({api_code})" if api_code else "Unknown")

    return CaptureMetadata(
        uuid=data.get("(uuid)", ""),
        captured_frames=data.get("DYCaptureEngine.captured_frames_count", 0),
        graphics_api=api,
    )


# ── binary string extraction ─────────────────────────────────────────────


def _extract_strings(data: bytes, min_length: int = 3) -> list[tuple[int, str]]:
    """Extract printable ASCII strings with their byte offsets."""
    if len(data) > _MAX_METADATA_BYTES:
        raise ValueError("Capture metadata byte budget exceeded")
    strings: list[tuple[int, str]] = []
    for match in re.finditer(rb"[ -~]+", data):
        if len(match[0]) >= min_length:
            if len(strings) >= _MAX_STRINGS:
                raise ValueError("Capture metadata string budget exceeded")
            strings.append((match.start(), match[0].decode("ascii")))
    return strings


# ── label extraction ──────────────────────────────────────────────────────


def extract_buffer_labels(gputrace: Path) -> dict[str, str]:
    """Extract MTLBuffer/MTLTexture filename → label mapping.

    Scans device-resources binary files for printable strings, matching
    resource filenames to their nearest following label string.
    """
    label_map: dict[str, str] = {}
    inventory = _resource_names(gputrace)

    for data in _metadata_files(gputrace):
        strings = _extract_strings(data)

        resource_refs = [
            (off, s)
            for off, s in strings
            if _RESOURCE_NAME.fullmatch(s) and s in inventory
        ]

        label_strings = [
            (off, s)
            for off, s in strings
            if " " in s
            and len(s) >= 5
            and not s.startswith("MTL")
            and not s.startswith("/")
        ]

        label_index = 0
        for roff, rname in resource_refs:
            while label_index < len(label_strings) and label_strings[label_index][0] <= roff:
                label_index += 1
            if label_index < len(label_strings):
                label_map[rname] = label_strings[label_index][1]

    return label_map


# ── shader info ───────────────────────────────────────────────────────────


def extract_shader_info(
    gputrace: Path,
) -> tuple[list[str], list[str]]:
    """Extract shader function names and library paths.

    Returns (functions, libraries).
    """
    functions: list[str] = []
    libraries: list[str] = []

    seen_functions = set()
    for data in _metadata_files(gputrace):
        strings = _extract_strings(data)

        for _, s in strings:
            if s.endswith(".metallib"):
                libraries.append(s)

        in_functions = False
        for _, s in strings:
            if s.endswith(".metallib"):
                in_functions = True
                continue
            if s == "pipeline-libraries":
                in_functions = False
                continue
            if (
                in_functions
                and re.match(r"^[a-z_][a-z0-9_]*$", s)
                and len(s) > 3
                and s not in ("function", "functions", "buffer", "buffers")
            ):
                if s not in seen_functions:
                    seen_functions.add(s)
                    functions.append(s)

    return functions, libraries


# ── buffer reading ────────────────────────────────────────────────────────

_LAYOUT_FORMATS: dict[str, tuple[str, int, int]] = {
    "float": ("f", 1, 1),
    "float2": ("ff", 2, 2),
    "float3": ("ffff", 4, 3),        # Metal simd_float3 is 16-byte aligned; 4th float is padding
    "packed_float3": ("fff", 3, 3),  # Metal packed_float3, no alignment padding
    "float4": ("ffff", 4, 4),
    "uint32": ("I", 1, 1),
    "int32": ("i", 1, 1),
    "half": ("e", 1, 1),
    "half2": ("ee", 2, 2),
    "half4": ("eeee", 4, 4),
}


def parse_layout(layout: str) -> tuple[str, list[tuple[str, int, int]]]:
    """Parse a layout string into a struct format and field descriptors.

    Returns (struct_format, [(component_name, n_read, n_report), ...]).
    """
    components = [c.strip() for c in layout.split(",")]
    fmt_parts: list[str] = []
    fields: list[tuple[str, int, int]] = []

    for comp in components:
        if comp not in _LAYOUT_FORMATS:
            raise ValueError(f"Unknown layout component: {comp}")
        fmt, n_read, n_report = _LAYOUT_FORMATS[comp]
        fmt_parts.append(fmt)
        fields.append((comp, n_read, n_report))

    return "<" + "".join(fmt_parts), fields


def read_buffer(
    gputrace: Path,
    filename: str,
    layout: str,
    start: int = 0,
    count: int = 10,
) -> list[BufferElement]:
    """Read a buffer file with the given layout.

    Layout formats:
        "float"                  - single float per element
        "float4"                 - 4 floats (RGBA, position, etc.)
        "float4,float4,float4"  - compound struct (e.g., Particle)
        "uint32"                 - single uint32 per element
        "half4"                  - 4 half-precision floats
    """
    _resource_path(gputrace, filename)

    fmt, fields = parse_layout(layout)
    stride = struct.calcsize(fmt)

    with _capture_file(gputrace, filename) as handle:
        data = handle.read()
    total_elements = len(data) // stride

    results: list[BufferElement] = []
    for idx in range(start, min(start + count, total_elements)):
        offset = idx * stride
        values = struct.unpack_from(fmt, data, offset)

        field_entries: list[tuple[str, object]] = []
        vi = 0
        for ci, (comp_name, n_read, n_report) in enumerate(fields):
            if n_report == 1:
                field_entries.append((f"field{ci}", values[vi]))
            else:
                field_entries.append(
                    (f"field{ci}", list(values[vi : vi + n_report]))
                )
            vi += n_read
        results.append(BufferElement(index=idx, fields=tuple(field_entries)))

    return results


def buffer_element_count(gputrace: Path, filename: str, layout: str) -> int:
    """Return total element count for a buffer given a layout."""
    _resource_path(gputrace, filename)
    fmt, _ = parse_layout(layout)
    stride = struct.calcsize(fmt)
    with _capture_file(gputrace, filename) as handle:
        return os.fstat(handle.fileno()).st_size // stride


# ── resource listing ──────────────────────────────────────────────────────


def list_resources(gputrace: Path) -> CaptureInfo:
    """List all resources in a .gputrace bundle."""
    labels = extract_buffer_labels(gputrace)
    metadata = parse_metadata(gputrace)
    functions, libraries = extract_shader_info(gputrace)

    resources: list[CaptureResource] = []
    for fname in sorted(_resource_names(gputrace)):
        m = _RESOURCE_NAME.fullmatch(fname)
        with _capture_file(gputrace, fname) as handle:
            size = os.fstat(handle.fileno()).st_size
        kind = "buffer" if m.group(1) == "Buffer" else "texture"
        label = labels.get(fname, "")
        resources.append(
            CaptureResource(
                filename=fname,
                size_bytes=size,
                label=label,
                kind=kind,
            )
        )

    return CaptureInfo(
        metadata=metadata,
        resources=tuple(resources),
        shader_functions=tuple(functions),
        shader_libraries=tuple(ShaderLibrary(path=p) for p in libraries),
    )


# ── buffer stats (no numpy dependency) ───────────────────────────────────


@dataclass(frozen=True)
class BufferStats:
    filename: str
    label: str
    size_bytes: int
    float_count: int
    min_val: float
    max_val: float
    mean_val: float
    first_four: tuple[float, ...]


def buffer_summary(gputrace: Path, filename: str, label: str) -> BufferStats | None:
    """Compute summary statistics for a buffer interpreted as float32."""
    _resource_path(gputrace, filename)
    with _capture_file(gputrace, filename) as handle:
        data = handle.read()
    n_floats = len(data) // 4
    if n_floats == 0:
        return None

    values = struct.unpack_from(f"<{n_floats}f", data)

    return BufferStats(
        filename=filename,
        label=label,
        size_bytes=len(data),
        float_count=n_floats,
        min_val=min(values),
        max_val=max(values),
        mean_val=sum(values) / n_floats,
        first_four=tuple(values[:4]),
    )


def resolve_buffer_filename(
    gputrace: Path,
    buffer_query: str,
    labels: dict[str, str] | None = None,
) -> str | None:
    """Resolve a buffer label or filename to an actual filename.

    Tries exact filename match first, then exact label match,
    then partial label match (case-insensitive).
    """
    inventory = _resource_names(gputrace)
    if buffer_query in inventory:
        return buffer_query

    if labels is None:
        labels = extract_buffer_labels(gputrace)

    # Exact label match (case-insensitive)
    for fname, label in labels.items():
        if fname in inventory and label.lower() == buffer_query.lower():
            return fname

    # Partial match
    for fname, label in labels.items():
        if fname in inventory and buffer_query.lower() in label.lower():
            return fname

    return None
