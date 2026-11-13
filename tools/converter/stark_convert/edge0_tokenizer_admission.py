"""Publication gate for a separately qualified source-native Edge0 tokenizer.

The proof candidate remains in edge0_native_tokenizer.py. A profile may be
accepted only after its immutable proof is reviewed; caller-supplied receipts
and a bare ``passed`` flag cannot enable preparation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import tokenizers

from . import edge0_native_tokenizer as candidate
from ._build_metadata import open_regular
from .edge0_source import EDGE0_FILES, EDGE0_REPO_ID, EDGE0_REVISION, source_path

POLICY = "edge0-native-v2"
DERIVED_FILES = ("tokenizer.json", "tokenizer_config.json")
# Owner-accepted for local, non-commercial source preparation only. This binds
# the Python tokenizer/oracle proof; Stark's Rust parser and model are unverified.
ACCEPTED_PROFILE: dict | None = {
    "policy": POLICY,
    "profile": "edge0-native-v2-unicode-20261112-edge0-03b1-python0233",
    "candidate_revision": "edge0-native-tokenizer-v2-unicode",
    "source": {"repo": "Edge0/Edge0-145B-q3k", "revision": "d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74"},
    "tokenizers_version": "0.23.3",
    "candidate_module_sha256": "29097cf0ae44d7b2858b709fda6a484a75e0ccfd4eabd61ff0c42ad7bc3ec5db",
    "unicode_asset_sha256": "a8d7b7bea4f53942453a535e71b52a91098328eed16dcd466af69ed7a8225820",
    "tokenizer_json_sha256": "f1d23198102cc49d8cb199600be2f5ec97d35b7540830b284010a425147066c4",
    "tokenizer_config_sha256": "5a095bafcd6923c00e6d7041dbe493f7a7995c1331c75a3c3e6625ec91cce6ae",
    "proof": {
        "attempt": "20261112-EDGE0-03B.1",
        "sha256": "d74501849a329954290a58bb376bc3d59b513cc5af01a04b1da40e3e94052ca0",
    },
    "primary_eos_token_id": 248046,
    "eog_token_ids": [248044, 248046, 248063, 248064, 248065],
    "scope": "local_noncommercial_evaluation_preparation_only",
    "native_rust_tokenizer_parser_validated": False,
    "inference_validated": False,
    "generation_stream_validated": False,
    "upstream_frontend_equivalence_claimed": False,
}


class NativeTokenizerUnqualified(ValueError):
    """No immutable profile qualifies this derivation for local preparation."""


def _encoded(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _package_bytes(name: str) -> bytes:
    with open_regular(Path(__file__).with_name(name)) as stream:
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise NativeTokenizerUnqualified("Native tokenizer qualification input exceeds limit")
    return data


def _source_bytes(directory: Path, name: str) -> bytes:
    expected = EDGE0_FILES[name]
    with open_regular(source_path(directory, name)) as stream:
        data = stream.read(expected["size"] + 1)
    if len(data) != expected["size"] or _sha(data) != expected["sha256"]:
        raise NativeTokenizerUnqualified("Native tokenizer source identity mismatch")
    return data


def qualification_identity() -> dict:
    profile = copy.deepcopy(ACCEPTED_PROFILE)
    if not isinstance(profile, dict):
        raise NativeTokenizerUnqualified("Native tokenizer policy is not qualified")
    digests = ("candidate_module_sha256", "unicode_asset_sha256", "tokenizer_json_sha256", "tokenizer_config_sha256")
    proof = profile.get("proof")
    if (
        profile.get("policy") != POLICY
        or not isinstance(profile.get("profile"), str)
        or not profile["profile"]
        or profile.get("candidate_revision") != candidate.UNICODE_CANDIDATE_REVISION
        or profile.get("source") != {"repo": EDGE0_REPO_ID, "revision": EDGE0_REVISION}
        or profile.get("tokenizers_version") != tokenizers.__version__
        or not isinstance(proof, dict)
        or proof.get("attempt") != "20261112-EDGE0-03B.1"
        or type(profile.get("primary_eos_token_id")) is not int
        or not isinstance(profile.get("eog_token_ids"), list)
        or len(profile["eog_token_ids"]) != 5
        or any(type(value) is not int for value in profile["eog_token_ids"])
        or profile["eog_token_ids"] != sorted(set(profile["eog_token_ids"]))
        or any(
            not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in [*(profile.get(key) for key in digests), proof.get("sha256")]
        )
        or _sha(_package_bytes("edge0_native_tokenizer.py")) != profile.get("candidate_module_sha256")
        or _sha(_package_bytes("edge0_unicode.json")) != profile.get("unicode_asset_sha256")
    ):
        raise NativeTokenizerUnqualified("Native tokenizer qualification identity mismatch")
    return profile


def _derive(directory: Path, gguf_metadata: dict) -> tuple[dict[str, bytes], dict]:
    metadata = candidate.sealed_metadata(directory, gguf_metadata)
    asset = json.loads(_package_bytes("edge0_unicode.json"))
    program = candidate.candidate_program(metadata, unicode_asset=asset)
    special = metadata["descriptor"]["special_ids"]
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "add_bos_token": False,
        "clean_up_tokenization_spaces": False,
        "bos_token": metadata["tokens"][special["tokenizer.ggml.bos_token_id"]],
        "eos_token": metadata["tokens"][special["tokenizer.ggml.eos_token_id"]],
        "eos_token_id": special["tokenizer.ggml.eos_token_id"],
        "eog_token_ids": sorted(
            metadata["vocab"][name]
            for name in (
                "<|endoftext|>",
                "<|im_end|>",
                "<|fim_pad|>",
                "<|repo_name|>",
                "<|file_sep|>",
            )
        ),
        "pad_token": metadata["tokens"][special["tokenizer.ggml.padding_token_id"]],
        "added_tokens_decoder": {
            str(token["id"]): {key: value for key, value in token.items() if key != "id"}
            for token in program["added_tokens"]
        },
        "chat_template": _source_bytes(directory, "tokenizer/chat_template.jinja").decode("utf-8"),
    }
    return {"tokenizer.json": _encoded(program), "tokenizer_config.json": _encoded(config)}, metadata


def inspect_derived_metadata(directory: Path, gguf_metadata: dict) -> dict:
    """Return hashes for review; this does not qualify or write any artifact."""
    derived, _ = _derive(Path(directory), gguf_metadata)
    return {name: {"size": len(data), "sha256": _sha(data)} for name, data in derived.items()}


def _validated(directory: Path, gguf_metadata: dict) -> tuple[dict[str, bytes], dict]:
    profile = qualification_identity()
    derived, _ = _derive(directory, gguf_metadata)
    keys = {"tokenizer.json": "tokenizer_json_sha256", "tokenizer_config.json": "tokenizer_config_sha256"}
    if any(_sha(derived[name]) != profile[key] for name, key in keys.items()):
        raise NativeTokenizerUnqualified("Native tokenizer derived identity mismatch")
    config = json.loads(derived["tokenizer_config.json"])
    if (
        config["eos_token_id"] != profile["primary_eos_token_id"]
        or config["eog_token_ids"] != profile["eog_token_ids"]
    ):
        raise NativeTokenizerUnqualified("Native tokenizer termination identity mismatch")
    return derived, _validation(profile)


def _validation(profile: dict) -> dict:
    return {
        "schema_version": 1,
        "policy": POLICY,
        "profile": profile["profile"],
        "qualification": profile,
        "source_metadata_validated": True,
        "borrowed_qwen_equivalence": "rejected",
        "native_loading": "unverified",
        "execution": "unverified",
    }


def validation_identity() -> dict:
    return _validation(qualification_identity())


def preflight_tokenizer(directory: Path, gguf_metadata: dict, *, offline: bool = False) -> dict:
    """Validate a registered local derivation without fetching or writing files."""
    _, validation = _validated(Path(directory), gguf_metadata)
    return validation


def prepare_tokenizer(directory: Path, gguf_metadata: dict, *, offline: bool = False) -> dict:
    directory = Path(directory)
    derived, validation = _validated(directory, gguf_metadata)
    for name, data in derived.items():
        path = directory / name
        if path.is_symlink():
            raise NativeTokenizerUnqualified("Native tokenizer derived target is linked")
        if path.exists():
            with open_regular(path) as stream:
                if stream.read(len(data) + 1) != data:
                    raise NativeTokenizerUnqualified("Native tokenizer derived target identity mismatch")
        else:
            with path.open("xb") as stream:
                stream.write(data)
    return {
        "native_tokenizer_validation": validation,
        "derived_metadata": {
            name: {
                "size": len(data),
                "sha256": _sha(data),
                "derivation": "pinned-edge0-source-and-qualified-native-unicode-classes",
                "profile": validation["profile"],
                "qualification_proof_sha256": validation["qualification"]["proof"]["sha256"],
            }
            for name, data in derived.items()
        },
    }
