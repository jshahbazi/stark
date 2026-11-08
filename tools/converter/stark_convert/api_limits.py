"""Immutable byte admission policies; production defaults remain independent."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True, slots=True)
class APIByteLimits:
    text_body_bytes: int = 1024 * 1024
    image_body_bytes: int = 6 * 1024 * 1024
    pending_body_bytes: int = 24 * 1024 * 1024
    message_text_bytes: int = 1024 * 1024
    tool_result_bytes: int = 256 * 1024
    argument_bytes: int = 64 * 1024
    schema_bytes: int = 256 * 1024
    tool_output_bytes: int = 512 * 1024
    isolated: bool = False

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name != "isolated" and (type(value) is not int or not 0 < value <= 2**63 - 1):
                raise ValueError("invalid_api_byte_limits")
        if type(self.isolated) is not bool or self.pending_body_bytes < max(
            self.text_body_bytes, self.image_body_bytes
        ):
            raise ValueError("invalid_api_byte_limits")

    def metadata(self):
        return asdict(self)


PRODUCTION_BYTE_LIMITS = APIByteLimits()
BOUND_VERSION = "bpe-bytelevel-nfc-history-v2"
POLICY_VERSION = "maxxbench-context-bytes-v2"
SERIALIZER_VERSION = "json-utf8-compact-v1"
PINNED_BASH_SCHEMA_SHA256 = "4a08480a7cfeabc6389b905c978b74318be10ec73ba651ba6436fbac380f67f4"
SHAPE_PROOF_VERSION = "pinned-bash-parser-v1"
# Exact reviewed source registrations in api_tools.tool_parser_for; deliberately
# exclude its broader syntax-equivalent program fallback from this proof.
_TOOL_PROFILES = {
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8": ("qwen3-hermes-v1", 16384),
    "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5": ("qwen3-hermes-v1", 16384),
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041": ("qwen38-xml-v1", 3),
    "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715": ("qwen35-xml-v1", 3),
    "34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891": ("glm53-xml-v1", 3),
    "7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466": ("glm53-xml-v1", 3),
}


def tool_profile(source):
    """Resolve only hash-pinned programs, without executing a template."""
    if not isinstance(source, str):
        raise ValueError("benchmark_tool_profile_unsupported")
    return tool_profile_digest(hashlib.sha256(source.encode("utf-8")).hexdigest())


def tool_profile_digest(digest):
    """Bind the parser's argument-tree proof to the pinned bash schema.

    XML/GLM parsers consult _xml_tool_schemas and accept only command:string;
    _xml_parameter_value returns that raw string. Successful arguments contain
    at most object/key/string (three nodes). Hermes accepts a generic dict and
    retains its existing 16,384-node guard. The owning parser is checked again
    against Desktop's actual loaded codec before readiness.
    """
    if not isinstance(digest, str) or digest not in _TOOL_PROFILES:
        raise ValueError("benchmark_tool_profile_unsupported")
    parser, nodes = _TOOL_PROFILES[digest]
    return {
        "template_sha256": digest,
        "parser": parser,
        "argument_nodes_per_call": nodes,
        "shape_proof_version": SHAPE_PROOF_VERSION,
        "schema_sha256": PINNED_BASH_SCHEMA_SHA256,
    }


@dataclass(frozen=True, slots=True)
class TokenizerByteBound:
    sha256: str
    normalizer: str
    max_token_bytes: int
    max_json_token_bytes: int
    max_decoded_token_bytes: int
    max_argument_json_token_bytes: int
    bound_version: str = BOUND_VERSION
    canonical_decomposition_max: int = 4

    def __post_init__(self):
        if (
            re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None
            or self.normalizer not in {"none", "NFC"}
            or self.bound_version != BOUND_VERSION
            or self.canonical_decomposition_max != 4
            or any(
                type(value) is not int or value < 1
                for value in (
                    self.max_token_bytes,
                    self.max_json_token_bytes,
                    self.max_decoded_token_bytes,
                    self.max_argument_json_token_bytes,
                )
            )
        ):
            raise ValueError("tokenizer_byte_bound_unsupported")

    def metadata(self):
        return asdict(self)


def _byte_alphabet():
    initial = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    values = {chr(value): value for value in initial}
    for index, value in enumerate(value for value in range(256) if value not in initial):
        values[chr(256 + index)] = value
    return values


def _preserving_pretokenizer(value):
    if not isinstance(value, dict):
        return False
    if value.get("type") == "Sequence":
        parts = value.get("pretokenizers")
        return (
            isinstance(parts, list)
            and bool(parts)
            and all(_preserving_pretokenizer(part) for part in parts)
            and parts[-1].get("type") == "ByteLevel"
            and sum(part.get("type") == "ByteLevel" for part in parts) == 1
        )
    if value.get("type") == "Split":
        return (
            value.get("behavior") == "Isolated"
            and value.get("invert") is False
            and isinstance(value.get("pattern"), dict)
            and set(value["pattern"]) == {"Regex"}
            and isinstance(value["pattern"]["Regex"], str)
        )
    return value.get("type") == "ByteLevel" and value.get("add_prefix_space") is False


def tokenizer_byte_bound(path: Path) -> TokenizerByteBound:
    """Inspect tokenizer metadata only, retaining maxima and a source digest.

    NFC preimages use the canonical decomposition scalar-count proof (D≤4),
    not a byte-ratio shortcut. Added-token stripping and dropped source spans
    would invalidate a finite history bound and are refused.
    """
    raw = Path(path).read_bytes()
    try:
        data = json.loads(raw)
        model = data["model"]
        vocab = model["vocab"]
        decoder = data["decoder"]
        normalizer = data.get("normalizer")
        normalization = "none" if normalizer is None else "NFC" if normalizer == {"type": "NFC"} else None
        processor = data.get("post_processor")
        if (
            model.get("type") != "BPE"
            or model.get("unk_token") is not None
            or model.get("fuse_unk") is not False
            or model.get("byte_fallback") is not False
            or model.get("continuing_subword_prefix") not in (None, "")
            or model.get("end_of_word_suffix") not in (None, "")
            or not isinstance(vocab, dict)
            or not vocab
            or decoder.get("type") != "ByteLevel"
            or (processor is not None and processor.get("type") != "ByteLevel")
            or normalization is None
            or not _preserving_pretokenizer(data.get("pre_tokenizer"))
            or data.get("truncation") is not None
            or data.get("padding") is not None
        ):
            raise ValueError
        alphabet = _byte_alphabet()
        if not set(alphabet) <= set(vocab):
            raise ValueError
        added = data.get("added_tokens") or []
        if not isinstance(added, list) or any(
            not isinstance(token, dict)
            or token.get("lstrip") is not False
            or token.get("rstrip") is not False
            or not isinstance(token.get("content"), str)
            or not token["content"]
            or type(token.get("id")) is not int
            or token["id"] < 0
            or (token["content"] in vocab and vocab[token["content"]] != token["id"])
            for token in added
        ):
            raise ValueError
        maximum = wire = decoded = argument_wire = 0
        for token, identifier in vocab.items():
            if not isinstance(token, str) or not token or type(identifier) is not int or identifier < 0:
                raise ValueError
            value = bytes(alphabet[character] for character in token)
            maximum = max(maximum, len(value))
            wire = max(wire, sum(6 if byte < 32 else 2 if byte in (34, 92) else 1 for byte in value))
            argument_wire = max(argument_wire, sum(7 if byte < 32 else 4 if byte in (34, 92) else 1 for byte in value))
            # Concatenating byte fragments cannot increase UTF-8 replacement
            # expansion beyond decoding each fragment separately.
            decoded = max(decoded, len(value.decode("utf-8", errors="replace").encode("utf-8")))
        for token in added:
            value = token["content"].encode("utf-8")
            maximum = max(maximum, len(value))
            wire = max(wire, sum(6 if byte < 32 else 2 if byte in (34, 92) else 1 for byte in value))
            decoded = max(decoded, len(value))
            argument_wire = max(argument_wire, sum(7 if byte < 32 else 4 if byte in (34, 92) else 1 for byte in value))
        return TokenizerByteBound(
            hashlib.sha256(raw).hexdigest(), normalization, maximum, wire, decoded, argument_wire
        )
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError):
        raise ValueError("tokenizer_byte_bound_unsupported") from None


def benchmark_byte_policy(
    context_tokens,
    output_tokens,
    step_limit,
    tokenizer,
    envelope,
    *,
    template_metadata_bytes=0,
    profile,
):
    """Derive budgets numerically without allocating a maximum-sized payload."""
    for value in (context_tokens, output_tokens, step_limit):
        if type(value) is not int or value < 1:
            raise ValueError("invalid_benchmark_byte_policy")
    if type(template_metadata_bytes) is not int or template_metadata_bytes < 0:
        raise ValueError("benchmark_protocol_capacity_unavailable")
    if not isinstance(tokenizer, TokenizerByteBound) or output_tokens >= context_tokens:
        raise ValueError("invalid_benchmark_byte_policy")
    if not isinstance(profile, dict) or profile != tool_profile_digest(profile.get("template_sha256")):
        raise ValueError("benchmark_tool_profile_unsupported")
    if (
        not isinstance(envelope, dict)
        or envelope.get("serializer_version") != SERIALIZER_VERSION
        or set(envelope)
        != {
            "serializer_version",
            "max_calls",
            "max_messages",
            "tool_schema_sha256",
            "request_envelope_bytes",
            "response_envelope_bytes",
        }
        or not isinstance(envelope.get("tool_schema_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", envelope["tool_schema_sha256"]) is None
        or envelope["tool_schema_sha256"] != PINNED_BASH_SCHEMA_SHA256
        or envelope.get("max_calls") != 16
        or envelope.get("max_messages") != 2 + (step_limit - 1) * 17
        or any(
            type(envelope.get(name)) is not int or envelope[name] < 1
            for name in ("request_envelope_bytes", "response_envelope_bytes")
        )
    ):
        raise ValueError("invalid_benchmark_byte_policy")
    raw = context_tokens * tokenizer.max_token_bytes
    wire = context_tokens * max(tokenizer.max_json_token_bytes, tokenizer.max_argument_json_token_bytes)
    if tokenizer.normalizer == "NFC":
        raw *= 4 * tokenizer.canonical_decomposition_max
        # Seven covers a raw control scalar after canonical argument JSON then
        # request JSON; the same D<=4 scalar preimage proof covers both layers.
        wire = context_tokens * tokenizer.max_token_bytes * 7 * tokenizer.canonical_decomposition_max
    generated = output_tokens * tokenizer.max_decoded_token_bytes
    calls = max(0, step_limit - 1) * 16
    discarded_raw = (step_limit - 1) * generated
    discarded_wire = 6 * discarded_raw
    # Canonical XML arguments introduce the fixed command key and root/string
    # delimiters. Measure their outer JSON framing separately from token text.
    canonical = json.dumps({"command": ""}, ensure_ascii=False, separators=(",", ":"))
    argument_frame = calls * (len(json.dumps(canonical, ensure_ascii=False).encode("utf-8")) - 2)
    incoming_raw = raw + discarded_raw
    incoming_wire = wire + discarded_wire + argument_frame
    body = max(incoming_wire + envelope["request_envelope_bytes"], PRODUCTION_BYTE_LIMITS.text_body_bytes)
    image_body = max(body, PRODUCTION_BYTE_LIMITS.image_body_bytes)
    limits = APIByteLimits(
        text_body_bytes=body,
        image_body_bytes=image_body,
        pending_body_bytes=image_body,
        message_text_bytes=incoming_raw,
        tool_result_bytes=max(incoming_raw, PRODUCTION_BYTE_LIMITS.tool_result_bytes),
        argument_bytes=max(raw, 6 * generated + 1024, PRODUCTION_BYTE_LIMITS.argument_bytes),
        schema_bytes=max(raw, PRODUCTION_BYTE_LIMITS.schema_bytes),
        tool_output_bytes=max(generated, PRODUCTION_BYTE_LIMITS.tool_output_bytes),
        isolated=True,
    )
    # Framing bytes upper-bound skeleton nodes; argument trees have the bound
    # of the exact owning parser. These are counters, never allocation requests.
    nodes = max(65536, envelope["request_envelope_bytes"] + calls * profile["argument_nodes_per_call"])
    header = envelope["request_envelope_bytes"] + envelope["response_envelope_bytes"]
    template_request = incoming_wire + header + 6 * template_metadata_bytes
    # Both helper implementations count string bytes and container/scalar
    # costs. A parent contributes at most four units per child, native scalar
    # values at most 32, and Python integer accounting at most twice its
    # decimal source bytes. Keys/schema/codec strings are framing, not tokens.
    value_bytes = 2 * (max(incoming_raw, 6 * generated + 1024) + header + 6 * template_metadata_bytes) + 36 * nodes
    output_bytes = raw
    working = template_request + value_bytes + output_bytes + 64 * nodes
    live = max(256 * 1024**2, 8 * working)
    cumulative = max(16 * 1024**2, live * (nodes + 16384 + 1))
    fuel = max(1_000_000, nodes * 16384)
    cpu = max(2, math.ceil(working / 1024**2))
    native = {
        "request_byte_limit": max(wire + header, context_tokens * 12 + header),
        "reply_byte_limit": max(wire + header, context_tokens * 12 + header, 12 * generated + header),
        "template": {
            "request_bytes": template_request,
            "value_bytes": value_bytes,
            "output_bytes": output_bytes,
            "sequence_items": max(16384, envelope["max_messages"], calls),
            "value_nodes": nodes,
            "fuel": fuel,
            "cumulative_allocation_bytes": cumulative,
            "live_allocation_bytes": live,
            "cpu_seconds": cpu,
            "wall_seconds": max(5.0, 2.5 * cpu),
        },
    }
    if any(
        value > sys.maxsize
        for value in (
            template_request,
            value_bytes,
            output_bytes,
            live,
        )
    ) or any(value > 2 * sys.maxsize + 1 for value in (cumulative, fuel)):
        raise ValueError("benchmark_protocol_capacity_unavailable")
    policy = {
        "version": POLICY_VERSION,
        "context_tokens": context_tokens,
        "output_tokens": output_tokens,
        "step_limit": step_limit,
        "tokenizer": tokenizer.metadata(),
        "tool_profile": dict(profile),
        "envelope": dict(envelope),
        "limits": limits.metadata(),
        "response_body_bytes": max(64 * 1024, 12 * generated + envelope["response_envelope_bytes"]),
        "prompt_utf8_bytes": raw,
        "prompt_json_bytes": wire,
        "generated_utf8_bytes": generated,
        "discarded_assistant_utf8_bytes": discarded_raw,
        "discarded_assistant_json_bytes": discarded_wire,
        "argument_framing_bytes": argument_frame,
        "request_text_utf8_bytes": incoming_raw,
        "request_json_bytes": incoming_wire,
        "template_metadata_bytes": template_metadata_bytes,
        "native_policy": native,
    }
    digest = hashlib.sha256(json.dumps(policy, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return limits, policy, digest


def model_byte_policy(model, context_tokens, output_tokens, step_limit, envelope):
    """Derive the complete policy from metadata before loading native weights."""
    path = Path(model)
    tokenizer = tokenizer_byte_bound(path / "tokenizer.json")
    try:
        config_bytes = (path / "config.json").read_bytes()
        config = json.loads(config_bytes)
        codec = config.get("token_codec_config", {})
        profile = tool_profile(codec.get("prompt_template"))
        metadata_bytes = len(json.dumps(codec, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    except (OSError, ValueError, TypeError, AttributeError, UnicodeError):
        raise ValueError("benchmark_protocol_capacity_unavailable") from None
    return benchmark_byte_policy(
        context_tokens,
        output_tokens,
        step_limit,
        tokenizer,
        envelope,
        template_metadata_bytes=metadata_bytes,
        profile=profile,
    )
