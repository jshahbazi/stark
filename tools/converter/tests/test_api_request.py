import pytest

from stark_convert.api_request import parse_chat_request
from stark_convert.local_api import APIError


def parse(**kwargs):
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    payload.update(kwargs)
    return parse_chat_request(payload, 512, 32000)


def test_normalizes_messages_and_defaults():
    result = parse(
        messages=[
            {"role": "developer", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            {"role": "assistant", "content": ""},
        ]
    )
    assert result["messages"] == [{"role": "system", "content": "ab"}, {"role": "assistant", "content": ""}]
    assert result["sampling"] == {"mode": "greedy"}


def test_stochastic_and_aliases():
    result = parse(max_completion_tokens=12, temperature=0.7, top_p=0.8, seed=4, stop=["END"])
    assert result["maximum"] == 12
    assert result["sampling"] == {"mode": "stochastic", "temperature": 0.7, "top_p": 0.8}
    assert result["seed"] == 4 and result["stop"] == ["END"]


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": [{"role": "user", "content": "x"}], "n": 2},
        {"messages": [{"role": "user", "content": "x"}], "temperature": 3},
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 2, "max_completion_tokens": 2},
        {"messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_object"}},
    ],
)
def test_rejects_unsupported_values(payload):
    with pytest.raises(APIError) as error:
        parse_chat_request(payload, 512, 32000)
    assert error.value.status == 400


def test_accepts_neutral_compatibility_flags():
    result = parse(
        logprobs=False,
        top_logprobs=0,
        presence_penalty=0,
        frequency_penalty=None,
        user="u",
        metadata={"x": 1},
    )
    assert result["maximum"] == 512


@pytest.mark.parametrize(
    "extra",
    [
        {"messages": [{"role": [], "content": "x"}]},
        {"temperature": 10**500},
        {"top_p": float("nan")},
        {"min_p": -0.1},
        {"top_k": 0},
        {"top_k": 50000},
        {"stop": [""]},
        {"stream_options": {"include_usage": True}},
        {"metadata": {"invalid": float("inf")}},
    ],
)
def test_malformed_fields_are_precise_client_errors(extra):
    with pytest.raises(APIError) as error:
        parse(**extra)
    assert error.value.status == 400


def test_native_sampling_extensions_and_full_reply_limit():
    request = parse_chat_request(
        {
            "messages": [{"role": "user", "content": "x"}],
            "temperature": 1,
            "top_k": 20,
            "min_p": 0.05,
            "max_tokens": 2048,
        },
        4096,
        32000,
    )
    assert request["maximum"] == 2048
    assert request["sampling"] == {"mode": "stochastic", "temperature": 1, "top_k": 20, "min_p": 0.05}


def test_invalid_unicode_is_rejected_before_worker_dispatch():
    with pytest.raises(APIError):
        parse(messages=[{"role": "user", "content": "\ud800"}])
    with pytest.raises(APIError):
        parse(stop="\ud800")
