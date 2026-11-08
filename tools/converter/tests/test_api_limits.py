"""Numeric full-context bounds and modest synthetic transport/parser fixtures."""

import hashlib
import http.client
import io
import json
import socket
import sys
from collections import namedtuple
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from stark_convert.api_limits import (
    PINNED_BASH_SCHEMA_SHA256,
    PRODUCTION_BYTE_LIMITS,
    SERIALIZER_VERSION,
    APIByteLimits,
    TokenizerByteBound,
    _byte_alphabet,
    benchmark_byte_policy,
    tokenizer_byte_bound,
    tool_profile,
    tool_profile_digest,
)
from stark_convert.api_request import parse_chat_request
from stark_convert.api_tools import (
    QWEN3_TEMPLATE_SHA256,
    QWEN38_XML_TEMPLATE_SHA256,
    ToolTextStream,
    parse_tool_output,
    template_messages,
    tool_parser_for,
)
from stark_convert.local_api import APIError, LocalAPI

MIB = 1024**2
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
        },
    },
}


def isolated_limits(size=2 * MIB):
    return APIByteLimits(size, size, size, size, size, size, size, size, True)


def envelope(steps):
    return {
        "serializer_version": SERIALIZER_VERSION,
        "max_calls": 16,
        "max_messages": 2 + (steps - 1) * 17,
        "tool_schema_sha256": PINNED_BASH_SCHEMA_SHA256,
        "request_envelope_bytes": 4096,
        "response_envelope_bytes": 4096,
    }


@pytest.mark.parametrize(
    "context,normalizer,raw,wire,argument_wire,expected_raw,expected_wire",
    [
        (40960, "NFC", 128, 192, 224, 80 * MIB, 140 * MIB),
        (262144, "NFC", 128, 192, 224, 512 * MIB, 896 * MIB),
        (1048576, "none", 512, 512, 512, 512 * MIB, 512 * MIB),
        (262144, "none", 63, 192, 224, 63 * 262144, 56 * MIB),
    ],
)
def test_full_context_bounds_are_numeric_without_allocating_payloads(
    context,
    normalizer,
    raw,
    wire,
    argument_wire,
    expected_raw,
    expected_wire,
):
    bound = TokenizerByteBound("b" * 64, normalizer, raw, wire, raw, argument_wire)
    profile = tool_profile_digest(QWEN3_TEMPLATE_SHA256 if context == 40960 else QWEN38_XML_TEMPLATE_SHA256)
    limits, policy, digest = benchmark_byte_policy(context, 4096, 400, bound, envelope(400), profile=profile)
    assert policy["prompt_utf8_bytes"] == expected_raw and policy["prompt_json_bytes"] == expected_wire
    discarded = 399 * 4096 * raw
    assert limits.message_text_bytes == expected_raw + discarded and limits.tool_result_bytes >= expected_raw
    assert (
        limits.pending_body_bytes
        >= limits.text_body_bytes
        == (expected_wire + 6 * discarded + policy["argument_framing_bytes"] + 4096)
    )
    assert policy["response_body_bytes"] >= 12 * 4096 * raw + 4096
    assert digest == hashlib.sha256(json.dumps(policy, sort_keys=True, allow_nan=False).encode()).hexdigest()
    with pytest.raises(FrozenInstanceError):
        limits.argument_bytes = 1
    native = policy["native_policy"]
    assert native["request_byte_limit"] >= context * 12
    assert native["reply_byte_limit"] >= 12 * policy["generated_utf8_bytes"]
    assert native["template"]["request_bytes"] >= expected_wire
    assert native["template"]["output_bytes"] == expected_raw
    assert native["template"]["value_bytes"] >= 2 * expected_raw + 36 * native["template"]["value_nodes"]
    assert native["template"]["cumulative_allocation_bytes"] <= 2 * sys.maxsize + 1


def test_generic_argument_policy_can_use_unsigned_counter_without_allocating():
    _, policy, _ = benchmark_byte_policy(
        262144,
        4096,
        400,
        TokenizerByteBound("b" * 64, "NFC", 128, 192, 128, 224),
        envelope(400),
        profile=tool_profile_digest(QWEN3_TEMPLATE_SHA256),
    )
    assert sys.maxsize < policy["native_policy"]["template"]["cumulative_allocation_bytes"] <= 2 * sys.maxsize + 1


def test_protocol_counter_overflow_is_unavailable_without_reducing_context():
    with pytest.raises(ValueError, match="^benchmark_protocol_capacity_unavailable$"):
        benchmark_byte_policy(
            1048576,
            4096,
            1000000,
            TokenizerByteBound("b" * 64, "none", 512, 512, 512, 512),
            envelope(1000000),
            profile=tool_profile_digest(QWEN3_TEMPLATE_SHA256),
        )


def test_desktop_freezes_explicit_native_limits_and_preserves_default_worker_call_shape(tmp_path, monkeypatch):
    from stark_convert import desktop as module
    from stark_convert import template_runtime

    _, policy, _ = benchmark_byte_policy(
        32768,
        4096,
        4,
        TokenizerByteBound("b" * 64, "none", 4, 6, 4, 7),
        envelope(4),
        profile=tool_profile_digest(QWEN3_TEMPLATE_SHA256),
    )
    fake_limits = namedtuple("FakeLimits", policy["native_policy"]["template"])
    monkeypatch.setattr(template_runtime, "TemplateRuntimeLimits", fake_limits, raising=False)
    monkeypatch.setattr(module, "model_valid", lambda _path: True)
    monkeypatch.setattr(module, "read_model_config", lambda _path: {})
    templates = []
    workers = []
    monkeypatch.setattr(module, "load_template", lambda *_args, **kwargs: (templates.append(kwargs), None))

    class Intercepted(Exception):
        pass

    def worker(*_args, **kwargs):
        workers.append(kwargs)
        raise Intercepted

    monkeypatch.setattr(module, "Worker", worker)
    default = module.Desktop(tmp_path / "worker", tmp_path / "default", io.StringIO())
    explicit = module.Desktop(tmp_path / "worker", tmp_path / "isolated", io.StringIO(), benchmark_byte_policy=policy)
    expected_request = policy["native_policy"]["request_byte_limit"]
    expected_template_request = policy["native_policy"]["template"]["request_bytes"]
    policy["native_policy"]["request_byte_limit"] = 1
    policy["native_policy"]["template"]["request_bytes"] = 1
    for instance in (default, explicit):
        instance.models = [{"id": "sample", "path": str(tmp_path)}]
        with pytest.raises(Intercepted):
            instance.run(instance.epoch, "sample")
    assert templates[0] == {} and set(workers[0]) == {"gpu_memory_callback"}
    assert workers[1]["request_byte_limit"] == expected_request
    assert templates[1]["runtime_limits"].request_bytes == expected_template_request
    with pytest.raises(AttributeError):
        templates[1]["runtime_limits"].request_bytes = 1


def synthetic_tokenizer():
    vocab = {character: index for index, character in enumerate(_byte_alphabet())}
    vocab["a" * 512] = 256
    return {
        "model": {
            "type": "BPE",
            "unk_token": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "vocab": vocab,
        },
        "decoder": {"type": "ByteLevel"},
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
        "added_tokens": [],
        "post_processor": None,
        "padding": None,
        "truncation": None,
    }


def test_tokenizer_metadata_bound_and_fail_closed_stripping(tmp_path):
    path = tmp_path / "tokenizer.json"
    data = synthetic_tokenizer()
    path.write_text(json.dumps(data))
    bound = tokenizer_byte_bound(path)
    assert bound.max_token_bytes == bound.max_decoded_token_bytes == 512
    data["added_tokens"] = [{"content": "a", "lstrip": True, "rstrip": False}]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="^tokenizer_byte_bound_unsupported$"):
        tokenizer_byte_bound(path)


def test_literal_added_tokens_need_not_be_in_bpe_vocabulary(tmp_path):
    path = tmp_path / "tokenizer.json"
    data = synthetic_tokenizer()
    literal = "\N{GREEK SMALL LETTER ALPHA}" * 300
    data["added_tokens"] = [{"id": 1000, "content": literal, "lstrip": False, "rstrip": False}]
    path.write_text(json.dumps(data))
    bound = tokenizer_byte_bound(path)
    assert bound.max_token_bytes == bound.max_decoded_token_bytes == 600
    assert bound.max_json_token_bytes == 600
    data["added_tokens"][0]["id"] = -1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="^tokenizer_byte_bound_unsupported$"):
        tokenizer_byte_bound(path)


@pytest.mark.parametrize(
    "fixture,nodes",
    [("qwen3_4b_chat_template.jinja", 16384), ("glm53_chat_template.jinja", 3), ("flint_chat_template.jinja", 3)],
)
def test_source_pinned_argument_profiles_match_actual_production_resolver(fixture, nodes):
    source = (Path(__file__).parent / "fixtures" / fixture).read_text()
    profile = tool_profile(source)
    assert profile["parser"] == tool_parser_for({"prompt_template": source})
    assert profile["argument_nodes_per_call"] == nodes
    assert profile["schema_sha256"] == PINNED_BASH_SCHEMA_SHA256
    with pytest.raises(ValueError, match="^benchmark_tool_profile_unsupported$"):
        tool_profile(source + "\nprivate marker")


def test_argument_json_weights_cover_the_two_string_serializations(tmp_path):
    for value in ("\x00", '"', "\\", "\N{GREEK SMALL LETTER ALPHA}", "\N{MUSICAL SYMBOL G CLEF}"):
        canonical = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        encoded = json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        variable = len(encoded) - 6  # Outer quotes and two escaped inner quotes.
        expected = sum(7 if byte < 32 else 4 if byte in (34, 92) else 1 for byte in value.encode("utf-8"))
        assert variable <= expected
    data = synthetic_tokenizer()
    del data["model"]["vocab"]["a" * 512]
    control = next(character for character, byte in _byte_alphabet().items() if byte == 0)
    data["model"]["vocab"][control * 32] = 257
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(data))
    bound = tokenizer_byte_bound(path)
    assert bound.max_json_token_bytes == 192 and bound.max_argument_json_token_bytes == 224


def test_large_history_and_tool_result_survive_isolated_parser_losslessly():
    value = "x" * (300 * 1024)
    arguments = json.dumps({"key": "y" * (80 * 1024)})
    payload = {
        "tools": [TOOL],
        "messages": [
            {"role": "user", "content": "lookup"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_ok", "type": "function", "function": {"name": "lookup", "arguments": arguments}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_ok", "content": value},
        ],
    }
    with pytest.raises(APIError):
        parse_chat_request(payload, 4096, 32000, allow_tools=True)
    limits = isolated_limits()
    parsed = parse_chat_request(payload, 4096, 32000, allow_tools=True, byte_limits=limits)
    observed = hashlib.sha256(parsed["messages"][-1]["content"].encode()).digest()
    assert observed == hashlib.sha256(value.encode()).digest()
    rendered = template_messages(parsed["messages"], byte_limits=limits)
    assert len(rendered[1]["tool_calls"][0]["function"]["arguments"]["key"]) == 80 * 1024
    with pytest.raises(APIError):
        parse_chat_request(payload, 4096, 32000, allow_tools=True)


def test_large_generated_tool_value_passes_custom_stream_and_parser():
    value = "x" * (600 * 1024)
    text = "<tool_call><function=lookup><parameter=key>" + value + "</parameter></function></tool_call>"
    options = {"parser": "qwen38-xml-v1", "finish_reason": "eos", "call_id_prefix": "fixture"}
    with pytest.raises(APIError):
        parse_tool_output(text, [TOOL], **options)
    limits = isolated_limits()
    stream = ToolTextStream(lambda _text: None, parser=options["parser"], byte_limits=limits)
    stream.feed(text)
    message, finish = parse_tool_output(text, [TOOL], byte_limits=limits, **options)
    assert finish == "tool_calls" and not stream.tail
    result = json.loads(message["tool_calls"][0]["function"]["arguments"])["key"]
    assert hashlib.sha256(result.encode()).digest() == hashlib.sha256(value.encode()).digest()
    with pytest.raises(APIError) as error:
        parse_tool_output(text, [TOOL], byte_limits=isolated_limits(1024), **options)
    assert error.value.code == "api_resource_limit" and error.value.status == 503


def test_announced_maximum_does_not_allocate_or_read_at_that_size():
    class ShortStream(io.BytesIO):
        def read(self, size):
            assert size <= 64 * 1024
            return super().read(size)

    with pytest.raises(APIError) as error:
        LocalAPI._read_body(ShortStream(b"{}"), 768 * MIB)
    assert error.value.status == 400


def test_http_override_passes_same_immutable_limits_and_keeps_default_instance():
    limits = isolated_limits()
    seen = []

    def complete(payload, *, byte_limits):
        seen.append(byte_limits)
        parsed = parse_chat_request(payload, 4096, 32000, byte_limits=byte_limits)
        return {"content_bytes": len(parsed["messages"][0]["content"])}

    default = LocalAPI(lambda: [], lambda _payload: {}, api_key="a" * 64, port=0)
    custom = LocalAPI(lambda: [], complete, api_key="b" * 64, port=0, byte_limits=limits)
    body = json.dumps({"messages": [{"role": "user", "content": "x" * (MIB + 1024)}]}).encode()
    for api, expected in ((default, 413), (custom, 200)):
        info = api.start()
        port = int(info["base_url"].split(":")[2].split("/")[0])
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body if expected == 200 else None,
                {
                    "Authorization": "Bearer " + info["api_key"],
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
            )
            response = connection.getresponse()
            assert response.status == expected
            response.read()
        finally:
            connection.close()
            api.close()
    assert seen == [limits] and seen[0] is limits
    assert default._byte_limits is PRODUCTION_BYTE_LIMITS


def test_large_content_length_is_policy_derived_and_read_in_bounded_chunks():
    limits = isolated_limits(20 * MIB)
    api = LocalAPI(lambda: [], lambda **_options: {}, api_key="a" * 64, port=0, byte_limits=limits)
    info = api.start()
    port = int(info["base_url"].split(":")[2].split("/")[0])
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            b"{}",
            {
                "Authorization": "Bearer " + info["api_key"],
                "Content-Type": "application/json",
                "Content-Length": "10000000",
            },
        )
        connection.sock.shutdown(socket.SHUT_WR)
        response = connection.getresponse()
        assert response.status == 400  # Admitted header; incomplete body, never the old 7-digit 413.
        response.read()
    finally:
        connection.close()
        api.close()
    assert api._pending_body_bytes == 0


@pytest.mark.parametrize("kind", ["unknown_tool", "too_many_calls"])
def test_genuine_protocol_violations_keep_format_error_classification(kind):
    call = "<tool_call><function=lookup><parameter=key>x</parameter></function></tool_call>"
    text = call.replace("lookup", "unknown") if kind == "unknown_tool" else call * 17
    with pytest.raises(APIError) as error:
        parse_tool_output(
            text,
            [TOOL],
            parser="qwen38-xml-v1",
            finish_reason="eos",
            call_id_prefix="fixture",
            byte_limits=isolated_limits(),
        )
    assert error.value.code == "invalid_tool_call" and error.value.status == 502


@pytest.mark.parametrize("kind", ["nodes", "decoder_recursion"])
def test_ambiguous_argument_guard_is_unscored_resource_failure_in_isolated_path(kind):
    if kind == "nodes":
        body = json.dumps({"name": "lookup", "arguments": {"key": [0] * 16384}})
    else:
        body = '{"name":"lookup","arguments":{"key":' + "[" * 1100 + "0" + "]" * 1100 + "}}"
    text = "<tool_call>" + body + "</tool_call>"
    with pytest.raises(APIError) as error:
        parse_tool_output(
            text,
            [TOOL],
            parser="qwen3-hermes-v1",
            finish_reason="eos",
            call_id_prefix="fixture",
            byte_limits=isolated_limits(),
        )
    assert error.value.code == "api_resource_limit" and error.value.status == 503
    with pytest.raises(APIError) as production:
        parse_tool_output(text, [TOOL], parser="qwen3-hermes-v1", finish_reason="eos", call_id_prefix="fixture")
    assert production.value.code == "invalid_tool_call"
