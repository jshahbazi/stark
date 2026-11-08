"""Source-bound protocol checks; synthetic calls do not qualify model behavior."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from stark_convert.api_tools import (
    FLINT_TEMPLATE_SHA256,
    GLM53_TEMPLATE_SHA256,
    MAX_ARGUMENT_BYTES,
    QWEN25_TEMPLATE_SHA256,
    ToolTextStream,
    parse_tool_output,
    template_messages,
    tool_parser_for,
    tool_support_for,
    validate_parser_tools,
)
from stark_convert.chat import _ChatTemplate
from stark_convert.local_api import APIError
from stark_convert.model_inventory import model_tool_support

FIXTURES = Path(__file__).parent / "fixtures"
GLM = "glm53-xml-v1"
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
    },
}
OUTPUT = "<tool_call>lookup<arg_key>key</arg_key><arg_value>synthetic</arg_value></tool_call>"


def parse(text, tools=None, **kwargs):
    return parse_tool_output(
        text,
        [TOOL] if tools is None else tools,
        parser=GLM,
        finish_reason=kwargs.pop("finish_reason", "eos"),
        call_id_prefix="test",
        **kwargs,
    )


@pytest.mark.parametrize(
    "fixture,digest,parser",
    [
        ("glm53", GLM53_TEMPLATE_SHA256, GLM),
        ("flint", FLINT_TEMPLATE_SHA256, GLM),
        ("qwen25", QWEN25_TEMPLATE_SHA256, "qwen3-hermes-v1"),
    ],
)
def test_reviewed_source_and_harmless_program_variants(fixture, digest, parser):
    source = (FIXTURES / f"{fixture}_chat_template.jinja").read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == digest
    assert tool_parser_for({"prompt_template": source}) == parser
    assert tool_parser_for({"prompt_template": "{# maintenance #}" + source}) == parser
    assert tool_parser_for({"prompt_template": source.replace("<tool_call>", "<invoke>")}) is None


@pytest.mark.parametrize("fixture", ["glm53", "qwen25", "flint"])
def test_native_render_typed_call_roundtrip_and_parallel_result_history(fixture):
    source = (FIXTURES / f"{fixture}_chat_template.jinja").read_text()
    parser = tool_parser_for({"prompt_template": source})
    values = {"s": "\n café\n", "b": True, "n": None, "i": 3, "f": 2.5, "a": [1, False], "o": {"a": None}}
    types = dict(zip(values, ["string", "boolean", "null", "integer", "number", "array", "object"], strict=True))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {k: {"type": v} for k, v in types.items()}},
            },
        }
    ]
    calls = [
        {"id": f"call_{i}", "type": "function", "function": {"name": "lookup", "arguments": json.dumps(values)}}
        for i in range(2)
    ]
    messages = [
        {"role": "user", "content": "Synthetic request"},
        {"role": "assistant", "content": None, "tool_calls": calls},
    ]
    template = _ChatTemplate(source)
    rendered = template.render(messages=template_messages(messages), tools=tools, add_generation_prompt=False)
    if fixture in {"glm53", "flint"}:
        generated = rendered.rsplit("<|assistant|>", 1)[1].strip()
    else:
        generated = rendered.rsplit("<|im_start|>assistant", 1)[1].split("<|im_end|>")[0].strip()
    message, reason = parse_tool_output(generated, tools, parser=parser, finish_reason="eos", call_id_prefix="test")
    assert reason == "tool_calls" and len(message["tool_calls"]) == 2
    assert all(json.loads(c["function"]["arguments"]) == values for c in message["tool_calls"])
    messages += [
        {"role": "tool", "tool_call_id": "call_1", "content": "SECOND_RESULT"},
        {"role": "tool", "tool_call_id": "call_0", "content": "FIRST_RESULT"},
    ]
    history = template.render(messages=template_messages(messages), tools=tools, add_generation_prompt=True)
    assert history.index("FIRST_RESULT") < history.index("SECOND_RESULT")
    assert (
        "lookup" in history and ("<|observation|>" if fixture in {"glm53", "flint"} else "<tool_response>") in history
    )
    assert history.endswith(
        {"glm53": "<think>", "flint": "<|assistant|>", "qwen25": "<|im_start|>assistant\n"}[fixture]
    )


@pytest.mark.parametrize("fixture", ["glm53", "flint", "qwen25"])
def test_generated_suffix_uses_actual_prompt_thinking_state(fixture):
    source = (FIXTURES / f"{fixture}_chat_template.jinja").read_text()
    template = _ChatTemplate(source)
    prompt = template.render(
        messages=[{"role": "user", "content": "Synthetic request"}],
        tools=[TOOL],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    initial_thinking = prompt.endswith("<think>")
    assert initial_thinking is (fixture == "glm53")
    call = (
        '<tool_call>{"name":"lookup","arguments":{"key":"synthetic"}}</tool_call>' if fixture == "qwen25" else OUTPUT
    )
    suffix = "Synthetic reasoning</think>" + call
    options = dict(parser=tool_parser_for({"prompt_template": source}), finish_reason="eos", call_id_prefix="test")
    if initial_thinking:
        message, reason = parse_tool_output(suffix, [TOOL], initial_thinking=True, **options)
        assert reason == "tool_calls" and len(message["tool_calls"]) == 1
        # A call inside the still-open thought block is never executable.
        with pytest.raises(APIError):
            parse_tool_output(call, [TOOL], initial_thinking=True, **options)
    else:
        with pytest.raises(APIError):
            parse_tool_output(suffix, [TOOL], **options)
    with pytest.raises(APIError):
        parse_tool_output(suffix, [TOOL], initial_thinking=False, **options)


@pytest.mark.parametrize(
    "text",
    [
        "<tool_call>unknown</tool_call>",
        OUTPUT.replace("<tool_call>lookup", "<tool_call> lookup"),
        OUTPUT.replace("<tool_call>lookup", "<tool_call>lookup\n"),
        "<tool_call>lookup<arg_key>unknown</arg_key><arg_value>x</arg_value></tool_call>",
        OUTPUT.replace("</tool_call>", "<arg_key>key</arg_key><arg_value>x</arg_value></tool_call>"),
        OUTPUT[:-1],
        OUTPUT + "trailing",
        OUTPUT.replace("</arg_value>", ""),
        OUTPUT.replace("<arg_value>", "<arg_value><think>"),
        "<arg_value>x</arg_value>" + OUTPUT,
        "<arg_key>key</arg_key>",
        "</tool_call>" + OUTPUT,
        "```\n" + OUTPUT,
        "> " + OUTPUT,
        "<think>unfinished" + OUTPUT,
        OUTPUT.replace("<arg_key>key</arg_key>", "<arg_key> key</arg_key>"),
        OUTPUT.replace("synthetic", "<arg_key>nested</arg_key>"),
    ],
)
def test_glm_rejects_malformed_quoted_unknown_or_duplicate_calls(text):
    with pytest.raises(APIError) as error:
        parse(text)
    assert error.value.code == "invalid_tool_call"
    assert "synthetic" not in error.value.message


@pytest.mark.parametrize("finish", ["length", "context", "cancelled", "error"])
def test_glm_never_releases_unfinished_calls(finish):
    with pytest.raises(APIError, match="incomplete"):
        parse(OUTPUT, finish_reason=finish)


def test_glm_choice_counts_limits_and_empty_arguments():
    assert len(parse("<think>ready</think>" + OUTPUT)[0]["tool_calls"]) == 1
    assert json.loads(parse("<tool_call>lookup</tool_call>")[0]["tool_calls"][0]["function"]["arguments"]) == {}
    for text, options in [
        (OUTPUT, {"tool_choice": "none"}),
        (OUTPUT * 2, {"parallel_tool_calls": False}),
        (OUTPUT * 17, {}),
        (OUTPUT.replace("synthetic", "x" * (MAX_ARGUMENT_BYTES + 1)), {}),
    ]:
        with pytest.raises(APIError) as error:
            parse(text, **options)
        assert error.value.code == "invalid_tool_call"
    with pytest.raises(APIError) as error:
        parse("ordinary text", tool_choice="required")
    assert error.value.code == "tool_choice_not_satisfied"


@pytest.mark.parametrize(
    "schema,value,code",
    [
        ({}, "x", "tool_argument_type_unsupported"),
        ({"type": ["string", "integer"]}, "3", "invalid_tool_call"),
        ({"type": "boolean"}, "True", "invalid_tool_call"),
        ({"type": "number"}, "NaN", "invalid_tool_call"),
        ({"type": "object"}, '{"a":1,"a":2}', "invalid_tool_call"),
    ],
)
def test_glm_does_not_guess_argument_types(schema, value, code):
    tool = {
        "type": "function",
        "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"key": schema}}},
    }
    with pytest.raises(APIError) as error:
        parse(OUTPUT.replace("synthetic", value), [tool])
    assert error.value.code == code


def test_glm_schema_rejection_and_streaming_all_marker_splits():
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"$ref": "#/missing"}}}
    with pytest.raises(APIError) as error:
        validate_parser_tools([tool], GLM)
    assert error.value.code == "tool_schema_unsupported"
    text = "Synthetic preamble\n" + OUTPUT
    for split in range(len(text) + 1):
        delivered = []
        stream = ToolTextStream(delivered.append)
        stream.feed(text[:split])
        stream.feed(text[split:])
        assert "".join(delivered) == "Synthetic preamble"
        stream.finish(parse(text)[0]["content"])
        assert "".join(delivered) == "Synthetic preamble"


def test_import_assessment_uses_actual_template_and_ignores_claimed_capability(tmp_path):
    path = tmp_path / "config.json"
    assert model_tool_support(tmp_path)["status"] == "unchecked"
    config = {
        "token_codec_config": {"type": "ChatCodecConfig", "prompt_template": "{{ messages }}"},
        "supports_tools": True,
        "tool_parser": GLM,
    }
    path.write_text(json.dumps(config))
    assert model_tool_support(tmp_path)["status"] == "unavailable"
    source = (FIXTURES / "glm53_chat_template.jinja").read_text()
    config["token_codec_config"]["prompt_template"] = source
    path.write_text(json.dumps(config))
    assessment = model_tool_support(tmp_path)
    assert assessment == tool_support_for(config["token_codec_config"])
    assert assessment["status"] == "available" and assessment["model_behavior"] == "unverified"
    config["token_codec_config"]["prompt_template"] = source + "changed"
    path.write_text(json.dumps(config))
    assert model_tool_support(tmp_path)["status"] == "unavailable"


def validator_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "validate_tool_calling.py"
    spec = importlib.util.spec_from_file_location("tool_qualification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "wrong_argument",
        "wrong_result",
        "truncated_call",
        "native_error",
        "cancelled_result",
        "unclosed_answer",
        "call_inside_thinking",
    ],
)
def test_qualification_checks_behavior_and_never_returns_model_content(failure):
    module = validator_module()
    source = (FIXTURES / "glm53_chat_template.jinja").read_text()

    class SyntheticWorker:
        def __init__(self):
            self.requests = []

        def request(self, payload):
            self.requests.append(payload)
            index = len(self.requests)
            if index == 1:
                text = "Synthetic reasoning</think>" + OUTPUT.replace(
                    "synthetic", "wrong" if failure == "wrong_argument" else "alpha"
                )
                if failure == "call_inside_thinking":
                    text = text.removeprefix("Synthetic reasoning</think>")
                if failure == "native_error":
                    raise ValueError("private native diagnostic")
            elif index == 2 and failure not in {
                "wrong_argument",
                "truncated_call",
                "native_error",
                "call_inside_thinking",
            }:
                assert "K7V3M9" in payload["rendered_prompt"] and "<|observation|>" in payload["rendered_prompt"]
                text = "Synthetic reasoning</think>" + ("wrong" if failure == "wrong_result" else "K7V3M9")
            else:
                text = "Synthetic reasoning</think>HELLO"
            if failure == "unclosed_answer" and index > 1:
                text = text.removeprefix("Synthetic reasoning</think>")
            finish = "length" if failure == "truncated_call" and index == 1 else "eos"
            if failure == "cancelled_result" and index == 2:
                finish = "cancelled"
            return {"text": text, "finish_reason": finish}

    worker = SyntheticWorker()
    checks = module.run_checks(worker, _ChatTemplate(source), {"prompt_template": source}, context=8192, maximum=2048)
    assert all(checks.values()) == (failure is None)
    assert set(checks) == {"single_call_and_arguments", "tool_result_use", "ordinary_chat"}
    assert all(type(v) is bool for v in checks.values())
    assert checks["ordinary_chat"] is (failure != "unclosed_answer")


@pytest.mark.parametrize(
    "mutated",
    [
        None,
        "model.safetensors",
        "generation_config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "new_metadata",
        "helper",
    ],
)
def test_qualification_seals_inputs_closes_worker_and_writes_only_private_aggregate(
    tmp_path, monkeypatch, capsys, mutated
):
    module = validator_module()
    source = (FIXTURES / "glm53_chat_template.jinja").read_text()
    (tmp_path / "config.json").write_text(
        json.dumps({"token_codec_config": {"type": "ChatCodecConfig", "prompt_template": source}})
    )
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"synthetic weight bytes")
    for name in ("generation_config.json", "tokenizer_config.json", "model.safetensors.index.json"):
        (tmp_path / name).write_bytes(b"synthetic original data")
    executable = tmp_path / "worker"
    executable.write_bytes(b"synthetic worker")
    helper = tmp_path / "template-helper"
    helper.write_bytes(b"synthetic original data")
    output = tmp_path / "result.json"
    closed = []

    class SyntheticWorker:
        def __init__(self, *args):
            pass

        def read(self):
            if mutated:
                if mutated == "new_metadata":
                    (tmp_path / "processor_config.json").write_text("{}")
                    return {"ready": True, "mode": "text-generation-v1"}
                weights = helper if mutated == "helper" else tmp_path / mutated
                before = weights.stat()
                weights.write_bytes(b"x" * before.st_size)
                os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns))
            return {"ready": True, "mode": "text-generation-v1"}

        def close(self):
            closed.append(True)

    monkeypatch.setattr(module, "Worker", SyntheticWorker)
    monkeypatch.setattr(module, "_template_executable", lambda: helper)
    monkeypatch.setattr(
        module,
        "load_template",
        lambda model: ({"prompt_template": source}, SimpleNamespace(_executable=helper, _source=source)),
    )
    monkeypatch.setattr(
        module,
        "run_checks",
        lambda *args, **kwargs: {"single_call_and_arguments": True, "tool_result_use": True, "ordinary_chat": True},
    )
    monkeypatch.setattr(
        "sys.argv", ["validator", "--model", str(tmp_path), "--worker", str(executable), "--output", str(output)]
    )
    assert module.main() == (1 if mutated else 0) and closed == [True]
    report = json.loads(output.read_text())
    assert report["passed"] is (not mutated) and report["artifact_unchanged"] is (not mutated)
    assert output.stat().st_mode & 0o777 == 0o600
    assert "<tool_call>" not in output.read_text() and "rendered_prompt" not in output.read_text()
    assert "K7V3M9" not in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        module.main()
    assert closed == [True]


def test_qualification_rejects_external_streaming_even_without_root_bank_files(tmp_path, monkeypatch, capsys):
    module = validator_module()
    (tmp_path / "config.json").write_text('{"expert_streaming":{"manifest":"nested/manifest.json"}}')
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"synthetic")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        ["validator", "--model", str(tmp_path), "--worker", str(tmp_path / "absent-worker"), "--output", str(output)],
    )
    monkeypatch.setattr(
        module, "Worker", lambda *args: pytest.fail("External streaming must be rejected before launch")
    )
    assert module.main() == 1
    assert json.loads(output.read_text())["passed"] is False
    assert "manifest.json" not in output.read_text() and "manifest.json" not in capsys.readouterr().out


@pytest.mark.parametrize("fixture", ["glm53", "flint", "qwen25"])
def test_qualification_end_to_end_uses_native_renderer_and_real_private_worker_pipe(
    tmp_path, monkeypatch, capsys, fixture
):
    module = validator_module()
    source = (FIXTURES / f"{fixture}_chat_template.jinja").read_text()
    (tmp_path / "config.json").write_text(
        json.dumps({"token_codec_config": {"type": "ChatCodecConfig", "prompt_template": source}})
    )
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"synthetic weight bytes")
    call = (
        '<tool_call>{"name":"lookup","arguments":{"key":"alpha"}}</tool_call>'
        if fixture == "qwen25"
        else OUTPUT.replace("synthetic", "alpha")
    )
    prefix = "Synthetic reasoning</think>" if fixture == "glm53" else ""
    replies = [prefix + call, prefix + "K7V3M9", prefix + "HELLO"]
    executable = tmp_path / "synthetic-worker"
    executable.write_text(
        f"#!{sys.executable}\n"
        + "import json,sys\n"
        + "print(json.dumps({'ready':True,'mode':'text-generation-v1'}),flush=True)\n"
        + "replies="
        + repr(replies)
        + "\n"
        + "for line in sys.stdin:\n"
        + "    request=json.loads(line)\n"
        + "    assert request['operation']=='generate'\n"
        + "    print(json.dumps({'text':replies.pop(0),'finish_reason':'eos'}),flush=True)\n"
    )
    executable.chmod(0o700)
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv", ["validator", "--model", str(tmp_path), "--worker", str(executable), "--output", str(output)]
    )
    assert module.main() == 0
    report = json.loads(output.read_text())
    assert report["passed"] and report["artifact_unchanged"] and all(report["checks"].values())
    assert report["template_executable_sha256"] == module.sha256(module._template_executable())
    assert "K7V3M9" not in output.read_text() and "Synthetic reasoning" not in capsys.readouterr().out
