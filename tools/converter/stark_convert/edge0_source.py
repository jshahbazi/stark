"""Pinned Edge0 Q3_K source inventory and bounded admission checks.

Independent implementation of the formats documented by Edge0-AI/Edge0 at
b7900cbb322926ccebdcd0c0c5a1280497ecff22, notably edge0-strata/tools/iq_pack.py,
tools/mtp_pack.py and include/strata/core/latent_dense.hpp. No upstream source
is copied. Strata's MIT copyright/license and provenance are preserved in
Stark's provenance inventory. This module reads containers; it decodes no blocks.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import struct
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .gguf_import import _count, _HeaderReader, _number, _read, _string, _value
from .streaming import ConversionError, UnsupportedConversion

EDGE0_REPO_ID = "Edge0/Edge0-145B-q3k"
EDGE0_REVISION = "d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74"
EDGE0_LAYOUT = "edge0-strata"

# File identities are source facts, not evidence that Stark can execute them.
EDGE0_FILES: dict[str, dict] = {
    "LICENSE": {"size": 13059, "sha256": "b1f5016a9f27a770e0fdd98f37c588477154d07a96f50e203de8ac53cdbefa79"},
    "NOTICE": {"size": 2560, "sha256": "f2f4feabdd5b75e3ed5b41870a022bfc1dfdbcd798b8009f648cc870edbef777"},
    "dense.bin": {"size": 1538625024, "sha256": "f9bcc7f36eac4e59a7b1cb1962e038ba12dc831a41fb193761f1d1e72657fb62"},
    "experts.bin": {"size": 40076574720, "sha256": "947d15f3d69ecdb45d602a62bc84803451da779a22294e1db5b3568290b42879"},
    "index.txt": {"size": 117464, "sha256": "980366930f95ae1b8138652256c3b9be8925b02395d4f093b72410797675a349"},
    "labels.bin": {"size": 98304, "sha256": "ce9b4a20b92c5c866e65d84a3884c104986522471cf11cc5f797a153affbcc06"},
    "latent_dense_q8.bin": {
        "size": 3743416320,
        "sha256": "fbb25caaa0dcf45ef62f1a4945dcf98a53a166934a3b461b166fbafa831d87ca",
    },
    "model-skeleton.gguf": {
        "size": 5991569248,
        "sha256": "ab49fbe7590cdcda45d7a2bcae4be0bd42c6a70b951d594a00eb158d7df0b72f",
    },
    "model.json": {"size": 1531, "sha256": "95b8280bf5151e11cf7b0622688172c9572dd8fd13558bb64d7e8edfc8c32ed4"},
    "mtp/dense.bin": {"size": 116099072, "sha256": "c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a"},
    "mtp/dense.txt": {"size": 1851, "sha256": "991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd"},
    "mtp/experts.bin": {
        "size": 707788800,
        "sha256": "09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f",
    },
    "native_experts.txt": {"size": 1463, "sha256": "0de7b81744e89f722374842a73b8b26a8a6145c645493471a9c10a4f5f566e70"},
    "ple-fp8.gguf": {
        "size": 51200246112,
        "sha256": "c7615c59b060277ddf6f84729fa8f7469cfc82c7fb983d6e7eab5a2f7446f170",
    },
    "release-scope.json": {"size": 2534, "sha256": "9ab2d5f7bb8fef4caf664502a73a6b4371c56ad0244e81571df48587a69a11f4"},
    "scales/l0.f32": {"size": 7168, "sha256": "ccfb45da4cb91bf0cc19d867472ddfb5c2175640046f6bfc754aa4e344855972"},
    "scales/l1.f32": {"size": 7168, "sha256": "73a4a4964c59fe2ae6eebd994f099fde0389ddf7c49fe5994ea372d7f826a5c6"},
    "scales/l10.f32": {"size": 7168, "sha256": "30d4046e0efc59b0ff81d74d1da05f41661d12f3cdc3cade317d646b15aef135"},
    "scales/l11.f32": {"size": 7168, "sha256": "d41bc1a5fc130916cea24416da58d6c3edac68c6a27f24719c885083aef13d8f"},
    "scales/l12.f32": {"size": 7168, "sha256": "877b3eaac39446fe23023a06a643a0783f6604e4a722dcd5ed40e6976cb68408"},
    "scales/l13.f32": {"size": 7168, "sha256": "b62631e02ddf5ae3be123d4385ae1d147a13253e250a4d645a4e5f77673aeaa7"},
    "scales/l14.f32": {"size": 7168, "sha256": "547f5bd37f52a0e913532c90fb7ad11658ac4fabc6b2a28be159c4e07b1e286b"},
    "scales/l15.f32": {"size": 7168, "sha256": "374222a8750c7aa5e72f5a2b05229521931be61f99778b841d571ca1b19ae4d4"},
    "scales/l16.f32": {"size": 7168, "sha256": "de4aaf5ddc41a6b961db1e786929aad539da114495867cc3ee2283f77cb10a0a"},
    "scales/l17.f32": {"size": 7168, "sha256": "c29c3976ce49f137147213a84dece24c4221dae5326f1a0cf45717b03624368b"},
    "scales/l18.f32": {"size": 7168, "sha256": "aaf7450624f01bb074bae633f17432841a3794c8efaa76202f3d259046d2f679"},
    "scales/l19.f32": {"size": 7168, "sha256": "81625f559d0218685104a168d32ac8cabfb271531a8e8e9c8282df25f2754884"},
    "scales/l2.f32": {"size": 7168, "sha256": "1e2ad898903d58779bddef0708f6b9b605376f451da7d4a3e092c75921abee0e"},
    "scales/l20.f32": {"size": 7168, "sha256": "0effc90ca93708fc835a5fee3a51a9e4e8e28139e7d289e618bb160fbacc6de4"},
    "scales/l21.f32": {"size": 7168, "sha256": "4643ac0076f9df916418a8ea58ff70dfb5cd3ce8f8220686698dd1011ed70c48"},
    "scales/l22.f32": {"size": 7168, "sha256": "8c15e401e96b52144fab43ca05e312d9cd12a6ced369aff85b55bb4e8233e45d"},
    "scales/l23.f32": {"size": 7168, "sha256": "c737877b64bb32eb444eb497db2b1661c5f92bd35716a2b5ed57fa27c9c5bedc"},
    "scales/l24.f32": {"size": 7168, "sha256": "e708f5923e499b338c1e95b633eef61e75034ec68d8ea1a9579c159cdc372b5b"},
    "scales/l25.f32": {"size": 7168, "sha256": "f0ac87f861b8c85065e09fdef886f957f73e6ef7d42ef86d6bc6e4abe2daafd3"},
    "scales/l26.f32": {"size": 7168, "sha256": "14adbeb9d6dcb10dafe0912d897eb02e2aa6abdb9e0a56f377945697bee02630"},
    "scales/l27.f32": {"size": 7168, "sha256": "1003f7dbacce30b7d2e1d758cfb7057a742bdf648472fcbd63333234f076e515"},
    "scales/l28.f32": {"size": 7168, "sha256": "e912c6d85335fc9ec83635064e523b0efe19fd5ef4de4f299bddfd1203c33693"},
    "scales/l29.f32": {"size": 7168, "sha256": "8c186910f9afcecd23abdbaf2eee52f48360b114f4e47543a41922414db1fd5b"},
    "scales/l3.f32": {"size": 7168, "sha256": "5b4b1994bcb00a0e78150152ed4d3eba1c90203f3772fc9d3b01bb133661949a"},
    "scales/l30.f32": {"size": 7168, "sha256": "b5cd2e55d0baeed12a8479feb5df830817821a1d00ff24050c451ecdf4913404"},
    "scales/l31.f32": {"size": 7168, "sha256": "b65fc3a4d56ca8ec2bd59b41ca63e0946ce30f27e4380477a79e7bcb868bef34"},
    "scales/l32.f32": {"size": 7168, "sha256": "4590579308a3973129592ced56ff24ed38f3bfe3ef4a8412c9dc2dbd40d170a7"},
    "scales/l33.f32": {"size": 7168, "sha256": "b8e28a6a3a4e68b935d8779bc43f0cca8a7c87e43cfe0cdcb9ce757631c5bd79"},
    "scales/l34.f32": {"size": 7168, "sha256": "bad3b4908fecfd92429cdc1b16c0b00314a8f843f9510b8a87afdd297a348c65"},
    "scales/l35.f32": {"size": 7168, "sha256": "2bc0b3d52dbc8ff6079cf978307698aede8a7ebb0084ba746f6164ce020c4fb1"},
    "scales/l36.f32": {"size": 7168, "sha256": "d6627bdbf25218a3dd429fa752a1ef882fd610eb23b66871b42b078713059356"},
    "scales/l37.f32": {"size": 7168, "sha256": "8afc17ed3b68756be8e5590ecbebb3e36179fabafb71901faf63e67ffe02f244"},
    "scales/l38.f32": {"size": 7168, "sha256": "445fbb2f2365079f06569460e6a905dad3e7f475d23af5ac2b967ab548e27c27"},
    "scales/l39.f32": {"size": 7168, "sha256": "020697f9289610c49dea1b47b72dd7f0f3746352c52b02f7d9d05db9011ddb07"},
    "scales/l4.f32": {"size": 7168, "sha256": "223b5a8a9fef3fb289915b985f0492db776be6ee4bc794af6bd6b40bc73f5951"},
    "scales/l40.f32": {"size": 7168, "sha256": "64b0d2d16992197dc7e2f0986f3d71ab0b744b4df0bdbab15bf08450c0919b00"},
    "scales/l41.f32": {"size": 7168, "sha256": "a75175bdc03d571f557d59aede6de9a0e734c61b03cfacc4ceb74c19b3787b7d"},
    "scales/l42.f32": {"size": 7168, "sha256": "e6596c611be63531fbf46fc73c1d1fcf86d38b09cfb27f97ee63571eea4be9cd"},
    "scales/l43.f32": {"size": 7168, "sha256": "e6e2227612938fac5320a81970215dac65bbee6e555573a5a3da7632adfe6087"},
    "scales/l44.f32": {"size": 7168, "sha256": "4187577ae6c484c700dfbf003a698aef48ce27fff9035ba40781abec05b58bf7"},
    "scales/l45.f32": {"size": 7168, "sha256": "270938e5d001b88e23a01d7c0f967ca165f1ac35adeff23f6401365d4d381430"},
    "scales/l46.f32": {"size": 7168, "sha256": "e3c18301cbfdf48eaa11cbb657f0696d3582aad5f68e2c9c27bafa3481b5a4e2"},
    "scales/l47.f32": {"size": 7168, "sha256": "552e22ee2bf8dbce16b68ac95a1146a141443a42e3326f7115d47dbe0ec9b845"},
    "scales/l5.f32": {"size": 7168, "sha256": "2b54008e9681de98a3590ab2d87eccda2f0ce4cfb6a99c91ddea2bfaf7ee9103"},
    "scales/l6.f32": {"size": 7168, "sha256": "7c0475b18ff380f4fc900ef308226b0a98ef855e793add4c1d149a3620a18f30"},
    "scales/l7.f32": {"size": 7168, "sha256": "ffb91ee9d1abf5ce63a07c1cc6f8182849700c9701705c3a386398fde797771b"},
    "scales/l8.f32": {"size": 7168, "sha256": "2ef5f3fdea5a3bc49e8b7cd93a41a793460da3685ed9d0ea10e9aae053c85477"},
    "scales/l9.f32": {"size": 7168, "sha256": "788cb1c638605ac71728913ce4fcb14c008dbbf46982c4c9c9d1f7af90e03ba5"},
    "third-party/Qwen-LICENSE.txt": {
        "size": 3235,
        "sha256": "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08",
    },
    "tokenizer/chat_template.jinja": {
        "size": 8952,
        "sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    },
    "tokenizer/merges.txt": {
        "size": 3353258,
        "sha256": "4ba698529766c7a2f5052083ea10cdf2485577f5237a332ee291054eb11e3332",
    },
    "tokenizer/token_type.json": {
        "size": 744960,
        "sha256": "5088c8c298fc06af8382ddb3b76c888703ac2634e82263b97eedcc2ad202738b",
    },
    "tokenizer/tokenizer.json": {
        "size": 541,
        "sha256": "c0bd435efbc6c3b28964b47999afe16c4ad268c7c7c4e8cb3af410796a6f30bf",
    },
    "tokenizer/vocab.json": {
        "size": 5737005,
        "sha256": "4ba64f0332abcfb0b600b7df1537d1e836e68009d5fb7cdb77339738dc6365c4",
    },
}
PAYLOAD_FILES: tuple[str, ...] = (
    "experts.bin",
    "ple-fp8.gguf",
    "model-skeleton.gguf",
    "latent_dense_q8.bin",
    "dense.bin",
    "mtp/experts.bin",
    "mtp/dense.bin",
    "labels.bin",
)

_TYPES = {0: ("f32", 1, 4), 1: ("f16", 1, 2), 8: ("q8_0", 32, 34), 24: ("i8", 1, 1), 30: ("bf16", 1, 2)}
_MAX_FILE_SIZE = 1 << 40
_NAME = re.compile(r"[A-Za-z0-9_.]+\Z")
_CHUNK = 1024 * 1024
SKELETON_HEADER_BYTES = 11_012_448
SKELETON_HEADER_SHA256 = "1cd961c7675aae8dcb9efc219ff58813f37441c33a567073bd521c72673b6685"
# Physical rows from the pinned ple-fp8.gguf tensor header. The logical head
# vocabularies occupy fewer rows; no general padding or rounding rule is assumed.
EDGE0_TABLE_ROWS = 320_001_536
EDGE0_TABLE_WIDTH = 160


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]  # GGUF order: contiguous dimension first.
    kind: str
    offset: int  # Absolute byte offset.
    nbytes: int


@dataclass(frozen=True)
class GGUFSource:
    metadata: dict
    tensors: dict[str, Tensor]
    data_start: int


def source_path(directory: Path, name: str) -> Path:
    """Reject traversal and symlinks, including parent-directory symlinks."""
    rel = PurePosixPath(name)
    if not name or rel.is_absolute() or str(rel) != name or any(p in {".", ".."} for p in rel.parts):
        raise ConversionError("Unsafe Edge0 source path")
    root = directory.resolve()
    path = root
    for part in rel.parts:
        path /= part
        if path.is_symlink():
            raise ConversionError("Edge0 source symlinks are unsupported")
    if not path.is_file():
        raise ConversionError(f"Missing Edge0 source file: {name}")
    return path


def read_gguf(path: Path, *, expected_size: int | None = None) -> GGUFSource:
    """Read a bounded GGUF v3 header and check all indexed spans.

    expected_size permits inspection of a bounded remote-header capture against
    the complete server-reported file length. Admission always uses local size.
    """
    size = path.stat().st_size if expected_size is None else expected_size
    if type(size) is not int or not 0 < size <= _MAX_FILE_SIZE:
        raise ConversionError("Invalid Edge0 GGUF size")
    with path.open("rb") as raw:
        stream = _HeaderReader(raw)
        if _read(stream, 4) != b"GGUF" or _number(stream, "I") != 3:
            raise UnsupportedConversion("Edge0 requires little-endian GGUF v3")
        nt, nm = _count(stream), _count(stream)
        if not 0 < nt <= 2000 or nm > 256:
            raise UnsupportedConversion("Edge0 GGUF inventory exceeds source limits")
        metadata = {}
        for _ in range(nm):
            key = _string(stream)
            if not key or key in metadata:
                raise ConversionError("Duplicate or empty Edge0 GGUF metadata key")
            metadata[key] = _value(stream, _number(stream, "I"))
        alignment = metadata.get("general.alignment", 32)
        if type(alignment) is not int or alignment != 32:
            raise UnsupportedConversion("Edge0 requires GGUF alignment 32")
        tensors = {}
        for _ in range(nt):
            name, rank = _string(stream), _number(stream, "I")
            if not _NAME.fullmatch(name) or name in tensors or rank not in {1, 2, 3, 4}:
                raise ConversionError("Invalid Edge0 GGUF tensor name or rank")
            shape = tuple(_number(stream, "Q") for _ in range(rank))
            if any(not 0 < d <= 400_000_000 for d in shape):
                raise ConversionError("Invalid Edge0 GGUF dimensions")
            kind_id, offset = _number(stream, "I"), _number(stream, "Q")
            if kind_id not in _TYPES:
                raise UnsupportedConversion("Unsupported Edge0 GGUF tensor encoding")
            kind, block, block_bytes = _TYPES[kind_id]
            if shape[0] % block or offset % alignment:
                raise ConversionError("Invalid Edge0 GGUF block alignment")
            nbytes = math.prod(shape) // block * block_bytes
            if nbytes > _MAX_FILE_SIZE:
                raise ConversionError("Edge0 GGUF tensor exceeds source limits")
            tensors[name] = Tensor(name, shape, kind, offset, nbytes)
        start = stream.tell() + (-stream.tell() % alignment)
        end = 0
        for tensor in sorted(tensors.values(), key=lambda t: t.offset):
            if tensor.offset != end + (-end % alignment) or start + tensor.offset + tensor.nbytes > size:
                raise ConversionError("Invalid Edge0 GGUF tensor spans")
            end = tensor.offset + tensor.nbytes
        if start + end != size:
            raise ConversionError("Edge0 GGUF has unindexed trailing data")
    return GGUFSource(
        metadata,
        {name: Tensor(t.name, t.shape, t.kind, start + t.offset, t.nbytes) for name, t in tensors.items()},
        start,
    )


def fetch_skeleton_metadata(directory: Path, *, offline: bool = False) -> dict:
    """Fetch only the pinned GGUF header, hash it, then inspect metadata.

    This preflight deliberately does not attest to the full GGUF payload hash.
    A server ignoring Range or returning another byte span fails closed.
    """
    cache = directory / ".cache"
    if cache.is_symlink():
        raise ConversionError("Unsafe Edge0 header cache")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "edge0-skeleton-header.gguf"
    if path.is_symlink():
        raise ConversionError("Unsafe Edge0 header cache")

    def verified() -> bool:
        return (
            path.is_file()
            and path.stat().st_size == SKELETON_HEADER_BYTES
            and hashlib.sha256(path.read_bytes()).hexdigest() == SKELETON_HEADER_SHA256
        )

    if not verified():
        if offline:
            raise ConversionError("Verified Edge0 skeleton header is unavailable offline")
        url = f"https://huggingface.co/{EDGE0_REPO_ID}/resolve/{EDGE0_REVISION}/model-skeleton.gguf"
        request = urllib.request.Request(url, headers={"Range": f"bytes=0-{SKELETON_HEADER_BYTES - 1}"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                expected_range = f"bytes 0-{SKELETON_HEADER_BYTES - 1}/{EDGE0_FILES['model-skeleton.gguf']['size']}"
                if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                    raise ConversionError("Edge0 header server did not honor the bounded byte range")
                data = response.read(SKELETON_HEADER_BYTES + 1)
        except (OSError, urllib.error.URLError):
            raise ConversionError("Could not fetch the pinned Edge0 skeleton header") from None
        if len(data) != SKELETON_HEADER_BYTES or hashlib.sha256(data).hexdigest() != SKELETON_HEADER_SHA256:
            raise ConversionError("Edge0 skeleton header identity mismatch")
        fd, temporary = tempfile.mkstemp(prefix="edge0-header-", dir=cache)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(data)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    source = read_gguf(path, expected_size=EDGE0_FILES["model-skeleton.gguf"]["size"])
    validate_skeleton(source)
    return source.metadata


def _text(path: Path, limit: int = 256 * 1024) -> list[str]:
    if path.stat().st_size > limit:
        raise ConversionError("Edge0 manifest exceeds source limits")
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeError:
        raise ConversionError("Invalid Edge0 manifest encoding") from None


def _uints(fields: list[str]) -> list[int]:
    if any(not s.isascii() or not s.isdecimal() or len(s) > 18 for s in fields):
        raise ConversionError("Invalid Edge0 manifest integer")
    return [int(s) for s in fields]


def read_pack_index(path: Path, dense_size: int) -> dict[str, Tensor]:
    """Read the pinned unsharded v3 subset; refuse all other pack layouts."""
    lines = _text(path)
    if len(lines) < 3 or not lines[0].startswith("# strata pack index v3 -- "):
        raise UnsupportedConversion("Edge0 requires Strata pack index v3")
    header = re.fullmatch(r"# align 64 pool ([0-9]+) tensors ([0-9]+)", lines[1])
    if header is None:
        raise ConversionError("Invalid Edge0 pack index header")
    pool, count = _uints(list(header.groups()))
    if pool != dense_size or count != len(lines) - 2 or not 0 < count <= 2000:
        raise ConversionError("Edge0 pack pool size or tensor count mismatch")
    tensors, pool_end, file_end = {}, 0, 0
    for line in lines[2:]:
        fields = line.split()
        if len(fields) != 19 or not _NAME.fullmatch(fields[0]) or fields[0] in tensors:
            raise ConversionError("Invalid or duplicate Edge0 pack tensor")
        values = _uints(fields[1:])
        file_id, kind, offset, nbytes, pool_offset, expanded, ne0, ne1, bits, group, block = values[:11]
        if file_id != 0 or any(values[11:]) or ne0 < 1 or ne0 > 400_000_000 or ne1 > 400_000_000:
            raise UnsupportedConversion("Unsupported Edge0 pack tensor layout")
        shape = (ne0, ne1) if ne1 else (ne0,)
        if pool_offset != pool_end:
            raise ConversionError("Invalid Edge0 pack pool offset")
        if kind == 0:
            if (offset, nbytes, expanded, bits, group, block) != (0, 0, 0, 8, 0, 32) or ne0 % 32:
                raise UnsupportedConversion("Unsupported Edge0 native tensor record")
            tensor = Tensor(fields[0], shape, "q8_0", 0, math.prod(shape) // 32 * 34)
        else:
            kinds = {2: ("f32", 4), 4: ("bf16", 2), 5: ("f16", 2)}
            if kind not in kinds or (bits, group, block) != (0, 0, 1):
                raise UnsupportedConversion("Unsupported Edge0 dense tensor encoding")
            encoding, width = kinds[kind]
            if nbytes != math.prod(shape) * width or expanded != nbytes or offset != file_end:
                raise ConversionError("Edge0 dense tensor span or size mismatch")
            tensor = Tensor(fields[0], shape, encoding, offset, nbytes)
            file_end = offset + nbytes + (-nbytes % 64)
        pool_end += expanded + (-expanded % 64)
        tensors[fields[0]] = tensor
    if file_end != dense_size or pool_end != pool:
        raise ConversionError("Edge0 dense file contains unindexed data")
    return tensors


def read_native_experts(path: Path, experts_size: int) -> dict:
    lines = _text(path)
    if (
        not lines
        or lines[0]
        != "# strata native experts v6: latent A-only, Q3_K gu + shared colscale (expert_embd 1792, expert_ff 640)"
    ):
        raise UnsupportedConversion("Edge0 requires the pinned v6 latent expert layout")
    # Q3_K: 110 bytes/256 values; IQ4_NL: 18 bytes/32 values.
    gate_bytes = 640 * (1792 // 256) * 110
    down_bytes = 1792 * (640 // 32) * 18
    record = 2 * gate_bytes + down_bytes
    if len(lines) != 49 or experts_size != 48 * 512 * record:
        raise ConversionError("Edge0 expert bank size or layer count mismatch")
    for layer, line in enumerate(lines[1:]):
        fields = line.split()
        if len(fields) != 5 or _uints(fields) != [layer, 11, 20, layer * 512 * record, record]:
            raise ConversionError("Edge0 expert layer type, order or span mismatch")
    return {
        "version": 6,
        "layers": 48,
        "experts_per_layer": 512,
        "record_bytes": record,
        "gate_bytes": gate_bytes,
        "up_bytes": gate_bytes,
        "down_bytes": down_bytes,
    }


def read_mtp_index(path: Path, dense_size: int) -> dict[str, Tensor]:
    tensors, end = {}, 0
    for line in _text(path):
        fields = line.split()
        if len(fields) != 6 or not _NAME.fullmatch(fields[0]) or fields[0] in tensors:
            raise ConversionError("Invalid or duplicate Edge0 MTP dense record")
        name, kind = fields[:2]
        rows, cols, offset, size = _uints(fields[2:])
        if kind not in {"q8_0", "f32", "bf16"} or not 0 < rows <= 400_000_000 or not 0 < cols <= 400_000_000:
            raise UnsupportedConversion("Unsupported Edge0 MTP dense layout")
        _, block, width = next(v for v in _TYPES.values() if v[0] == kind)
        if cols % block or size != rows * (cols // block) * width or offset != end:
            raise ConversionError("Edge0 MTP dense span or size mismatch")
        tensors[name] = Tensor(name, (cols, rows), kind, offset, size)
        end += size
    if len(tensors) != 29 or end != dense_size:
        raise ConversionError("Edge0 MTP dense count or file size mismatch")
    return tensors


def compare_duplicate_tensors(
    skeleton: GGUFSource, pack: dict[str, Tensor], skeleton_path: Path, dense_path: Path
) -> dict:
    """Compare shared raw tensors using at most two 1 MiB payload buffers."""
    compared, total = 0, 0
    with skeleton_path.open("rb") as left, dense_path.open("rb") as right:
        for name, source in skeleton.tensors.items():
            target = pack.get(name)
            if (
                target is None
                or target.kind != source.kind
                or target.shape != source.shape
                or target.nbytes != source.nbytes
            ):
                raise ConversionError("Edge0 skeleton and pack tensor geometry mismatch")
            if target.kind == "q8_0":
                continue  # Index explicitly delegates this tensor to the GGUF.
            left.seek(source.offset)
            right.seek(target.offset)
            remaining = source.nbytes
            while remaining:
                n = min(_CHUNK, remaining)
                a, b = left.read(n), right.read(n)
                if len(a) != n or len(b) != n or a != b:
                    raise ConversionError("Edge0 skeleton and pack payload mismatch")
                remaining -= n
            compared += 1
            total += source.nbytes
    return {"tensors": compared, "bytes": total, "method": "bounded-byte-comparison"}


def validate_labels(path: Path) -> None:
    if path.stat().st_size != 48 * 512 * 4:
        raise ConversionError("Edge0 projection label count mismatch")
    if any(v[0] > 7 for v in struct.iter_unpack("<I", path.read_bytes())):
        raise ConversionError("Edge0 expert projection labels must be in 0..7")


def validate_column_scales(path: Path) -> None:
    if path.stat().st_size != 1792 * 4:
        raise ConversionError("Edge0 column scale count mismatch")
    if any(not math.isfinite(v[0]) or v[0] <= 0 for v in struct.iter_unpack("<f", path.read_bytes())):
        raise ConversionError("Edge0 column scales must be finite and positive")


EXPECTED_GEOMETRY: dict = {
    "qwen4exp.block_count": 48,
    "qwen4exp.context_length": 262144,
    "qwen4exp.embedding_length": 2560,
    "qwen4exp.attention.head_count": 24,
    "qwen4exp.attention.head_count_kv": 2,
    "qwen4exp.rope.dimension_sections": [11, 11, 10, 0],
    "qwen4exp.rope.freq_base": 10000000.0,
    "qwen4exp.attention.layer_norm_rms_epsilon": 9.999999974752427e-07,
    "qwen4exp.expert_count": 512,
    "qwen4exp.expert_used_count": 10,
    "qwen4exp.attention.key_length": 256,
    "qwen4exp.attention.value_length": 256,
    "qwen4exp.expert_feed_forward_length": 640,
    "qwen4exp.expert_shared_feed_forward_length": 640,
    "qwen4exp.ssm.conv_kernel": 4,
    "qwen4exp.ssm.state_size": 128,
    "qwen4exp.ssm.group_count": 16,
    "qwen4exp.ssm.time_step_rank": 48,
    "qwen4exp.ssm.inner_size": 6144,
    "qwen4exp.attention.recurrent_layers": [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
    ],
    "qwen4exp.full_attention_interval": 4,
    "qwen4exp.rope.dimension_count": 64,
    "qwen4exp.hyper_connection.count": 4,
    "qwen4exp.hyper_connection.low_rank": 320,
    "qwen4exp.attention.indexer.head_count": 4,
    "qwen4exp.attention.indexer.key_length": 128,
    "qwen4exp.attention.indexer.top_k": 2048,
    "qwen4exp.attention.compress_ratios": [
        0,
        0,
        0,
        4,
        0,
        0,
        0,
        4,
        0,
        0,
        0,
        4,
        0,
        0,
        0,
        4,
        0,
        0,
        0,
        4,
        0,
        0,
        0,
        4,
        0,
        0,
        0,
        4,
        0,
        0,
        0,
        4,
        0,
        0,
        0,
        4,
        0,
        0,
        0,
        4,
        0,
        0,
        0,
        4,
        0,
        0,
        0,
        4,
    ],
    "qwen4exp.ple.layers": [1],
    "qwen4exp.ple.ngram_size": 3,
    "qwen4exp.ple.heads_per_ngram": 8,
    "qwen4exp.ple.conv_kernel": 4,
    "qwen4exp.ple.eos_token_id": 248044,
    "qwen4exp.ple.image_token_id": 248056,
    "qwen4exp.embedding_length_per_layer_input": 160,
    "qwen4exp.ple.layer_multipliers": [23703573157769, 20109073645365, 8052911324071],
    "qwen4exp.ple.head_offsets": [
        0,
        20000003,
        40000026,
        60000059,
        80000106,
        100000165,
        120000228,
        140000297,
        160000374,
        180000455,
        200000548,
        220000655,
        240000802,
        260000955,
        280001114,
        300001275,
    ],
    "qwen4exp.ple.head_vocab_sizes": [
        20000003,
        20000023,
        20000033,
        20000047,
        20000059,
        20000063,
        20000069,
        20000077,
        20000081,
        20000093,
        20000107,
        20000147,
        20000153,
        20000159,
        20000161,
        20000171,
    ],
}


def validate_skeleton(source: GGUFSource) -> None:
    metadata = source.metadata
    if metadata.get("general.architecture") != "qwen4exp":
        raise UnsupportedConversion("Edge0 skeleton architecture mismatch")
    actual = {k: v for k, v in metadata.items() if k.startswith("qwen4exp.")}

    def equal(a, b):
        if type(a) is not type(b):
            return False
        if isinstance(a, list):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
        return a == b

    if set(actual) != set(EXPECTED_GEOMETRY) or any(not equal(actual[k], v) for k, v in EXPECTED_GEOMETRY.items()):
        raise ConversionError("Edge0 skeleton geometry differs from the pinned artifact")
    if Counter(t.kind for t in source.tensors.values()) != {"q8_0": 302, "bf16": 484, "f32": 292, "f16": 1}:
        raise ConversionError("Edge0 skeleton tensor inventory mismatch")
    for name in ("token_embd.weight", "output.weight"):
        t = source.tensors.get(name)
        if t is None or t.shape != (2560, 248320) or t.kind != "q8_0":
            raise ConversionError("Edge0 vocabulary projection geometry mismatch")


def validate_ngram_table(table: GGUFSource, geometry: dict) -> dict:
    """Keep stored table geometry separate from addressable logical head rows."""
    tensor = table.tensors.get("per_layer_token_embd.weight")
    payload_bytes = EDGE0_TABLE_WIDTH * EDGE0_TABLE_ROWS
    if (
        len(table.tensors) != 1
        or tensor is None
        or tensor.shape != (EDGE0_TABLE_WIDTH, EDGE0_TABLE_ROWS)
        or tensor.kind != "i8"
    ):
        raise ConversionError("Edge0 n-gram table physical shape or encoding mismatch")
    if tensor.nbytes != payload_bytes:
        raise ConversionError("Edge0 n-gram table physical payload size mismatch")
    if (
        table.metadata.get("strata.ple.format") != "f8_e4m3"
        or table.metadata.get("strata.ple.scale") != 0.00019931793212890625
    ):
        raise ConversionError("Edge0 n-gram table format or scale mismatch")
    offsets = geometry.get("qwen4exp.ple.head_offsets")
    sizes = geometry.get("qwen4exp.ple.head_vocab_sizes")
    if (
        not isinstance(offsets, list)
        or not isinstance(sizes, list)
        or len(offsets) != 16
        or len(sizes) != 16
        or any(type(value) is not int or value < 0 for value in offsets)
        or any(type(value) is not int or value <= 0 for value in sizes)
    ):
        raise ConversionError("Invalid Edge0 n-gram logical head ranges")
    logical_rows = 0
    for offset, size in zip(offsets, sizes, strict=True):
        if offset != logical_rows or offset + size > EDGE0_TABLE_ROWS:
            raise ConversionError("Edge0 n-gram logical head ranges exceed or disagree with physical storage")
        logical_rows += size
    return {
        "table_rows": EDGE0_TABLE_ROWS,
        "logical_table_rows": logical_rows,
        "table_padding_rows": EDGE0_TABLE_ROWS - logical_rows,
        "table_payload_bytes": payload_bytes,
    }


def validate_source(directory: Path) -> dict:
    """Validate the complete pinned source after per-file hash verification."""
    paths = {name: source_path(directory, name) for name in EDGE0_FILES}
    for name, spec in EDGE0_FILES.items():
        if paths[name].stat().st_size != spec["size"]:
            raise ConversionError(f"Edge0 source size mismatch: {name}")
    skeleton = read_gguf(paths["model-skeleton.gguf"])
    validate_skeleton(skeleton)
    pack = read_pack_index(paths["index.txt"], paths["dense.bin"].stat().st_size)
    if len(pack) != 1367 or set(pack) != set(skeleton.tensors) | {
        f"blk.{layer}.ffn_{role}_exps.{suffix}"
        for layer in range(48)
        for role in ("gate", "up", "down")
        for suffix in ("scale", "input_scale")
    }:
        raise ConversionError("Edge0 pack tensor inventory differs from the pinned layout")
    for name, tensor in pack.items():
        if name not in skeleton.tensors and (tensor.kind, tensor.shape, tensor.nbytes) != ("f32", (512,), 2048):
            raise ConversionError("Edge0 expert scale vector geometry mismatch")
    experts = read_native_experts(paths["native_experts.txt"], paths["experts.bin"].stat().st_size)
    validate_labels(paths["labels.bin"])
    for layer in range(48):
        validate_column_scales(paths[f"scales/l{layer}.f32"])
    projection_per_layer = 8 * 1792 * (2560 // 32) * 34 + 8 * 2560 * (1792 // 32) * 34
    if paths["latent_dense_q8.bin"].stat().st_size != 48 * projection_per_layer:
        raise ConversionError("Edge0 Q8 projection formula mismatch")
    table = read_gguf(paths["ple-fp8.gguf"])
    table_layout = validate_ngram_table(table, skeleton.metadata)
    mtp = read_mtp_index(paths["mtp/dense.txt"], paths["mtp/dense.bin"].stat().st_size)
    # Strata Q2_0: 64 values in 18 bytes; fused gate/up then down.
    mtp_record = (1280 * (2560 // 64) + 2560 * (640 // 64)) * 18
    if paths["mtp/experts.bin"].stat().st_size != 512 * mtp_record:
        raise ConversionError("Edge0 MTP expert formula mismatch")
    duplicates = compare_duplicate_tensors(skeleton, pack, paths["model-skeleton.gguf"], paths["dense.bin"])
    return {
        "gguf_metadata": skeleton.metadata,
        "source_layout": {
            "layout": EDGE0_LAYOUT,
            "pack_index_version": 3,
            "experts": experts,
            "skeleton_tensors": len(skeleton.tensors),
            "pack_tensors": len(pack),
            "mtp_dense_tensors": len(mtp),
            "mtp_expert_record_bytes": mtp_record,
            "projection_layer_bytes": projection_per_layer,
            **table_layout,
            "duplicate_payload_validation": duplicates,
            "native_loading": "unverified",
        },
    }
