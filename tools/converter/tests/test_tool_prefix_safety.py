"""Quoted examples never become executable calls; every payload is synthetic."""

import json
import urllib.error
import urllib.request

import pytest
from test_api_tools import OUTPUT, output, xml_call
from test_desktop import desktop as desktop
from test_desktop import executable as executable
from test_desktop import model as model
from test_desktop import tool_desktop as tool_desktop

from stark_convert.local_api import APIError, LocalAPI

PARSERS = ["qwen3-hermes-v1", "qwen38-xml-v1", "qwen35-xml-v1"]


def wire(parser):
    return OUTPUT if parser == "qwen3-hermes-v1" else xml_call()


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("width", range(1, 9))
def test_unclosed_backtick_runs_cannot_wrap_executable_calls(parser, width):
    text = "Synthetic example: " + "`" * width + "quoted example\n" + wire(parser)
    with pytest.raises(APIError) as error:
        output(text, parser=parser)
    assert error.value.status == 502 and error.value.code == "invalid_tool_call"


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("opening,closing", [(1, 2), (2, 1), (1, 4), (2, 5)])
def test_mismatched_backtick_runs_do_not_close_code_spans(parser, opening, closing):
    text = "`" * opening + "Synthetic example" + "`" * closing + "\n" + wire(parser)
    with pytest.raises(APIError) as error:
        output(text, parser=parser)
    assert error.value.code == "invalid_tool_call"


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("width", range(3, 9))
def test_tilde_code_fences_cannot_wrap_executable_calls(parser, width):
    text = "~" * width + "json\nSynthetic example\n" + wire(parser)
    with pytest.raises(APIError) as error:
        output(text, parser=parser)
    assert error.value.code == "invalid_tool_call"


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize(
    "prefix",
    [
        "<!-- Synthetic unclosed comment\n",
        "<!-- Synthetic closed comment -->\n",
        "~~~\nSynthetic closed code fence\n~~~\n",
        "~~~Synthetic inline tilde marker.\n",
        "<think>Synthetic completed thought.</think>\n``Synthetic unclosed example\n",
        "> ",
        "Example:\n> ",
        " \t> ",
        "\t\t> ",
        "Synthetic introduction.\n  > ",
        ">> Synthetic nested block quote.\n",
        "<pre>\n",
        '<PRE class="synthetic">\n',
        "<code>\n",
        "<CODE data-example=synthetic>\n",
        "<blockquote cite=synthetic>\n",
        '<Q class="synthetic">\n',
        "</pre>\n",
        "</CODE >\n",
        "</BLOCKQUOTE>\n",
        "</q>\n",
        "<pre>Synthetic closed example.</pre>\n",
        "<code>Synthetic closed example.</code>\n",
        "<blockquote>Synthetic closed quotation.</blockquote>\n",
        "<q>Synthetic closed quotation.</q>\n",
    ],
)
def test_ambiguous_comment_and_fence_prefixes_are_conservatively_rejected(parser, prefix):
    with pytest.raises(APIError) as error:
        output(prefix + wire(parser), parser=parser)
    assert error.value.code == "invalid_tool_call"


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize(
    "prefix",
    [
        "Synthetic `lookup` mention.\n",
        "Synthetic ``lookup`` mention.\n",
        "Synthetic ``quoted ` value`` mention.\n",
        "Synthetic `quoted ' and \" value` mention.\n",
        "Synthetic `backslash does not escape the closer\\` mention.\n",
        '<think>Synthetic completed thought with unclosed " and `` <!-- ~~~ markers.</think>\n',
        "Synthetic comparison A > B.\n",
        "<preview>Synthetic label.</preview>\n",
        "<coder>Synthetic label.</coder>\n",
        "<query>Synthetic label.</query>\n",
        "<blockquoteish>Synthetic label.</blockquoteish>\n",
        "<qwerty>Synthetic label.</qwerty>\n",
    ],
)
def test_closed_inline_code_and_completed_thinking_allow_later_real_calls(parser, prefix):
    message, reason = output(prefix + wire(parser), parser=parser)
    assert reason == "tool_calls" and len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["function"] == {"name": "lookup", "arguments": '{"key":"a"}'}
    assert message["content"] == prefix.rstrip()


@pytest.mark.parametrize(
    "prefix",
    ["``Synthetic example\n", "~~~\nSynthetic example\n", "<!-- Synthetic example\n", "> ", "<PRE class=synthetic>"],
)
def test_quoted_tool_http_failures_emit_no_calls_and_release_request_resources(tool_desktop, prefix):
    desktop, payload, _, reply = tool_desktop
    worker = desktop.worker
    valid = reply["text"]
    reply["text"] = prefix + valid
    api = LocalAPI(desktop.api_models, desktop.api_complete, api_key="b" * 64, port=0, cancellable=True)
    info = api.start()

    def request(stream=False):
        return urllib.request.Request(
            info["base_url"] + "/chat/completions",
            data=json.dumps({**payload, "stream": stream}).encode(),
            headers={"Authorization": "Bearer " + "b" * 64, "Content-Type": "application/json"},
        )

    def resources_released():
        assert desktop.worker is worker
        assert desktop.job is None and desktop.api_queue == []
        assert worker.process.poll() is None

    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request(), timeout=3)
        assert error.value.code == 502
        assert json.loads(error.value.read())["error"]["code"] == "invalid_tool_call"
        resources_released()

        with urllib.request.urlopen(request(stream=True), timeout=3) as response:
            assert response.status == 200
            body = response.read()
        chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith(b"data: {")]
        assert [item["error"]["code"] for item in chunks if "error" in item] == ["invalid_tool_call"]
        calls = [
            call
            for chunk in chunks
            for choice in chunk.get("choices", [])
            for call in choice["delta"].get("tool_calls", [])
        ]
        assert not calls
        assert not any(
            choice.get("finish_reason") == "tool_calls" for chunk in chunks for choice in chunk.get("choices", [])
        )
        resources_released()

        reply["text"] = valid
        with urllib.request.urlopen(request(), timeout=3) as response:
            recovered = json.load(response)
        assert recovered["choices"][0]["finish_reason"] == "tool_calls"
        assert len(recovered["choices"][0]["message"]["tool_calls"]) == 1
        resources_released()
    finally:
        api.close()
