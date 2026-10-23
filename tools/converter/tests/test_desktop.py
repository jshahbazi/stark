"""Model library and lifecycle checks using private synthetic workers."""

import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import psutil
import pytest

from stark_convert.desktop import Desktop, DesktopError, converter_progress, metric


@pytest.fixture
def model(tmp_path):
    directory = tmp_path / "local model"
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps(
            {
                "decoder_config": {},
                "token_codec_config": {
                    "type": "ChatCodecConfig",
                    "default_system_prompt": "synthetic system",
                    "prompt_template": (
                        "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}"
                        "{% if add_generation_prompt %}assistant:{% endif %}"
                    ),
                },
            }
        )
    )
    (directory / "tokenizer.json").write_text("{}")
    (directory / "model.safetensors").write_bytes(b"synthetic model fixture")
    return directory


@pytest.fixture
def executable(tmp_path):
    path = tmp_path / "fake worker"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, time\n"
        "print(json.dumps({'ready':True,'mode':'text-generation-v1',"
        "'max_context_tokens':8192,'max_new_tokens':4096,'streaming':'text-delta-v1'}),flush=True)\n"
        "for line in sys.stdin:\n"
        "    request=json.loads(line)\n"
        "    if request['operation']=='tokenize': last_prompt=request['text']\n"
        "    if request['operation']=='generate' and 'wait fixture' in last_prompt: time.sleep(30)\n"
        "    if request['operation']=='generate' and 'failure fixture' in last_prompt: raise SystemExit(2)\n"
        "    if request.get('stream'):\n"
        "        for chunk in ['synthetic ', 'answer']: print(json.dumps({'event':'delta','text':chunk}),flush=True)\n"
        "    reply={'token_count':20,'token_ids':[2]*20} if request['operation']=='tokenize' else {\n"
        "        'text':'synthetic answer','input_tokens':20,'output_tokens':4,'finish_reason':'eos',\n"
        "        'input_tokens_per_second':100.0,'output_tokens_per_second':50.0,\n"
        "        'prefill_seconds':0.2,'decode_seconds':0.08}\n"
        "    print(json.dumps(reply),flush=True)\n"
    )
    path.chmod(0o755)
    return path


@pytest.fixture
def desktop(tmp_path, executable):
    instance = Desktop(executable, tmp_path / "library", io.StringIO(), timeout=2)
    yield instance
    instance.stop(quiet=True)


def events(desktop):
    return [json.loads(line) for line in desktop.output.getvalue().splitlines()]


def finish(desktop):
    thread = desktop.job
    if thread is not None:
        thread.join(timeout=5)
        assert not thread.is_alive()


def installed(desktop, model):
    desktop.dispatch({"command": "install", "kind": "local", "source": str(model)})
    finish(desktop)
    assert len(desktop.models) == 1
    return desktop.models[0]["id"]


def running(desktop, model):
    identifier = installed(desktop, model)
    desktop.dispatch({"command": "run", "model_id": identifier})
    finish(desktop)
    assert desktop.active_model == identifier
    return identifier


def test_api_completions_are_stateless_and_share_loaded_worker(desktop, model):
    from stark_convert.local_api import APIError

    with pytest.raises(APIError) as unavailable:
        desktop.api_complete({"model": "missing", "messages": [{"role": "user", "content": "hello"}]})
    assert unavailable.value.status == 503
    identifier = running(desktop, model)
    process = desktop.worker.process
    history = list(desktop.messages)
    payload = {"model": identifier, "messages": [{"role": "user", "content": "synthetic API request"}]}
    response = desktop.api_complete(payload)
    assert response["model"] == identifier
    assert response["choices"][0]["message"]["content"] == "synthetic answer"
    assert response["usage"] == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    assert desktop.messages == history
    assert desktop.worker.process is process
    assert desktop.api_models()[0]["id"] == identifier
    assert desktop.job is None
    assert all(e["event"] != "reply" for e in events(desktop))
    desktop.dispatch({"command": "generate", "text": "synthetic desktop request"})
    finish(desktop)
    assert len(desktop.messages) == 3
    assert desktop.worker.process is process


@pytest.mark.parametrize(
    "extra",
    [
        {"temperature": 3},
        {"temperature": False},
        {"n": 2},
        {"tools": []},
        {"max_tokens": True},
        {"max_tokens": 4097},
        {"max_tokens": 1, "max_completion_tokens": 1},
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "unsupported"}]}]},
        {"model": "unknown"},
    ],
)
def test_api_rejects_unsupported_requests_without_affecting_worker(desktop, model, extra):
    from stark_convert.local_api import APIError

    identifier = running(desktop, model)
    process = desktop.worker.process
    with pytest.raises(APIError) as error:
        desktop.api_complete({"model": identifier, "messages": [{"role": "user", "content": "hello"}], **extra})
    assert error.value.status in (400, 404)
    assert desktop.worker.process is process
    assert desktop.job is None


def test_api_busy_and_stop_fence_inflight_request(desktop, model):
    import threading

    from stark_convert.local_api import APIError

    identifier = running(desktop, model)
    errors = []
    payload = {"model": identifier, "messages": [{"role": "user", "content": "wait fixture"}]}

    def request():
        try:
            desktop.api_complete(payload)
        except APIError as error:
            errors.append(error)

    thread = threading.Thread(target=request)
    thread.start()
    deadline = time.monotonic() + 2
    while desktop.job is None and time.monotonic() < deadline:
        time.sleep(0.01)
    desktop.api_queue_timeout = 0.05
    with pytest.raises(APIError) as busy:
        desktop.api_complete(payload)
    assert busy.value.status == 429
    desktop.dispatch({"command": "generate", "text": "competing request"})
    assert any("Wait for the current operation" in e.get("message", "") for e in events(desktop))
    desktop.stop()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert len(errors) == 1 and errors[0].status == 503
    assert desktop.worker is None
    assert desktop.api_models() == []


def test_api_live_http_uses_desktop_model(desktop, model):
    import urllib.request

    identifier = running(desktop, model)
    desktop.start_api()
    details = next(e for e in events(desktop) if e["event"] == "connection")
    try:
        request = urllib.request.Request(
            details["base_url"] + "/chat/completions",
            data=json.dumps(
                {"model": identifier, "messages": [{"role": "user", "content": "hello"}], "stream": True}
            ).encode(),
            headers={"Authorization": "Bearer " + details["api_key"], "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read()
        chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith(b"data: {")]
        text = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
        assert text == "synthetic answer"
        assert b"data: [DONE]" in body
        assert details["api_key"] not in desktop.registry.read_text()
    finally:
        desktop.api.close()


def test_api_worker_failure_clears_job_and_can_restart(desktop, model):
    from stark_convert.local_api import APIError

    identifier = running(desktop, model)
    with pytest.raises(APIError) as failed:
        desktop.api_complete({"model": identifier, "messages": [{"role": "user", "content": "failure fixture"}]})
    assert failed.value.status == 503
    assert desktop.job is None
    assert desktop.worker is None
    desktop.dispatch({"command": "run", "model_id": identifier})
    finish(desktop)
    assert desktop.active_model == identifier


def test_api_streams_native_deltas_and_retains_worker(desktop, model):
    identifier = running(desktop, model)
    process = desktop.worker.process
    chunks = []
    response = desktop.api_complete(
        {"model": identifier, "messages": [{"role": "user", "content": "hello"}], "stream": True},
        on_delta=chunks.append,
    )
    assert chunks == ["synthetic ", "answer"]
    assert "".join(chunks) == response["choices"][0]["message"]["content"]
    assert desktop.worker.process is process
    assert desktop.job is None


def test_api_stream_disconnect_fences_worker_and_clears_job(desktop, model):
    from stark_convert.local_api import APIError

    identifier = running(desktop, model)
    process = desktop.worker.process

    def disconnected(_):
        raise BrokenPipeError

    with pytest.raises(APIError) as failed:
        desktop.api_complete(
            {"model": identifier, "messages": [{"role": "user", "content": "hello"}], "stream": True},
            on_delta=disconnected,
        )
    assert failed.value.status == 503
    assert desktop.worker is None
    assert desktop.job is None
    assert process.poll() is not None


def test_old_worker_streaming_is_explicitly_rejected(desktop, model):
    from stark_convert.local_api import APIError

    identifier = running(desktop, model)
    desktop.worker.streaming = False
    with pytest.raises(APIError) as failed:
        desktop.api_complete(
            {"model": identifier, "messages": [{"role": "user", "content": "hello"}], "stream": True},
            on_delta=lambda _: None,
        )
    assert failed.value.code == "streaming_unavailable"
    assert desktop.worker is not None
    assert desktop.job is None


def test_api_context_rejection_keeps_loaded_worker(desktop, model):
    from stark_convert.local_api import APIError

    identifier = running(desktop, model)
    process = desktop.worker.process
    desktop.api_context = 21
    with pytest.raises(APIError) as oversized:
        desktop.api_complete({"model": identifier, "messages": [{"role": "user", "content": "hello"}]})
    assert oversized.value.code == "context_length_exceeded"
    assert desktop.job is None
    assert desktop.worker.process is process


def test_api_sampling_limits_aliases_and_message_parts(desktop, model):
    running(desktop, model)
    seen = []
    infer = desktop.infer

    def capture(messages, maximum, **options):
        seen.append((messages, maximum, options))
        return infer(messages, maximum, **options)

    desktop.infer = capture
    result = desktop.api_complete(
        {
            "model": desktop.models[0]["name"],
            "temperature": 0.7,
            "top_p": 0.8,
            "seed": 42,
            "max_tokens": 1000,
            "messages": [
                {"role": "developer", "content": "system"},
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
        }
    )
    assert result["model"] == desktop.active_model
    assert seen[0][0] == [{"role": "system", "content": "system"}, {"role": "user", "content": "hello"}]
    assert seen[0][1] == 1000
    assert seen[0][2]["sampling"] == {"mode": "stochastic", "temperature": 0.7, "top_p": 0.8}
    assert seen[0][2]["seed"] == 42


def test_stale_epoch_and_expired_deadline_preserve_worker(desktop, model):
    import threading

    from stark_convert.local_api import APIError

    running(desktop, model)
    process = desktop.worker.process
    with patch.object(desktop.worker, "request") as request:
        with pytest.raises(APIError) as stale:
            desktop.infer([{"role": "user", "content": "hello"}], 16, epoch=desktop.epoch - 1)
        assert stale.value.code == "model_changed"
        request.assert_not_called()
    control = threading.Event()
    control.reason = "deadline"
    control.set()
    with pytest.raises(APIError) as expired:
        desktop.api_complete({"messages": [{"role": "user", "content": "hello"}]}, cancelled=control)
    assert expired.value.status == 504
    assert desktop.worker.process is process and process.poll() is None


def test_api_fifo_queue_and_waiter_cancellation(desktop, model):
    import threading

    from stark_convert.local_api import APIError

    running(desktop, model)
    entered, release = threading.Event(), threading.Event()
    order, errors = [], []

    def infer(messages, maximum, **options):
        text = messages[-1]["content"]
        order.append(text)
        if text == "first":
            entered.set()
            assert release.wait(2)
        return {"text": "answer", "input_tokens": 1, "output_tokens": 1, "finish_reason": "eos"}

    desktop.infer = infer

    def call(text, control=None):
        try:
            desktop.api_complete({"messages": [{"role": "user", "content": text}]}, cancelled=control)
        except APIError as error:
            errors.append(error.code)

    def queued(count):
        deadline = time.monotonic() + 2
        while len(desktop.api_queue) != count and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(desktop.api_queue) == count

    first = threading.Thread(target=call, args=("first",))
    cancelled = threading.Event()
    second = threading.Thread(target=call, args=("cancelled", cancelled))
    third = threading.Thread(target=call, args=("third",))
    first.start()
    assert entered.wait(2)
    second.start()
    queued(1)
    third.start()
    queued(2)
    desktop.api_queue_limit = 2
    with pytest.raises(APIError) as full:
        desktop.api_complete({"messages": [{"role": "user", "content": "over limit"}]})
    assert full.value.status == 429 and full.value.code == "queue_full"
    cancelled.set()
    second.join(timeout=2)
    release.set()
    first.join(timeout=2)
    third.join(timeout=2)
    assert not any(t.is_alive() for t in (first, second, third))
    assert order == ["first", "third"]
    assert errors == ["request_cancelled"]
    assert desktop.api_queue == [] and desktop.job is None


@pytest.fixture
def score_executable(tmp_path):
    path = tmp_path / "fake score worker"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, time\n"
        "assert sys.argv[1]=='--score-model-path'\n"
        "print(json.dumps({'ready':True,'mode':'exact-token-scoring-cache-v1','vocab_size':1000}),flush=True)\n"
        "prompt=''\n"
        "for line in sys.stdin:\n"
        "    request=json.loads(line)\n"
        "    if request['operation']=='tokenize':\n"
        "        text=request['text']\n"
        "        if len(text)>1: prompt=text\n"
        "        ids=[ord(text)] if len(text)==1 else [1,2,3]\n"
        "        reply={'token_ids':ids,'token_count':len(ids)}\n"
        "    elif request['operation']=='reset_cache': reply={'cache_reset':True}\n"
        "    else:\n"
        "        if 'wait fixture' in prompt: time.sleep(30)\n"
        "        if 'failure fixture' in prompt:\n"
        "            print('private exception fixture',file=sys.stderr,flush=True)\n"
        "            raise SystemExit(2)\n"
        "        candidates=request['candidate_token_ids']\n"
        "        p=[0.1/(len(candidates)-1)]*(len(candidates)-1)+[0.9]\n"
        "        if 'bad readout fixture' in prompt: p[0]=float('nan')\n"
        "        reply={'candidate_token_ids':candidates,'candidate_probabilities':p,\n"
        "               'input_tokens':len(request['input_token_ids']),\n"
        "               'temperature':request['temperature'],'truncated':False}\n"
        "        if 'context fixture' in prompt: reply={'error':'context_overflow'}\n"
        "    print(json.dumps(reply),flush=True)\n"
    )
    path.chmod(0o755)
    return path


def running_decision(desktop, model, score_executable):
    identifier = installed(desktop, model)
    desktop.score_executable = score_executable
    desktop.dispatch({"command": "run", "model_id": identifier, "mode": "decision"})
    finish(desktop)
    assert desktop.active_model == identifier and desktop.active_mode == "decision"
    return identifier


def test_api_rejects_decision_worker_and_does_not_touch_protocol(desktop, model, score_executable):
    from stark_convert.local_api import APIError

    identifier = running_decision(desktop, model, score_executable)
    assert desktop.api_models() == []
    status = desktop.api_status()
    assert status["model_loaded"] is False
    assert status["capabilities"]["stop_sequences"] is False
    assert status["capabilities"]["cancellation"] is False
    with patch.object(desktop.worker, "request", wraps=desktop.worker.request) as request:
        with pytest.raises(APIError) as error:
            desktop.api_complete({"model": identifier, "messages": [{"role": "user", "content": "fixture"}]})
        assert error.value.code == "model_not_loaded"
        request.assert_not_called()


def decision_command(**overrides):
    return {
        "command": "decide",
        "text": "synthetic decision text",
        "question": "synthetic decision question",
        "answers": ["First", "Second", "Third"],
        **overrides,
    }


def test_decision_private_pipe_roundtrip_uses_input_order_and_keeps_library_content_free(
    desktop, model, score_executable
):
    running_decision(desktop, model, score_executable)
    worker = desktop.worker
    desktop.dispatch(decision_command(answers=["Third", "First", "Second"]))
    finish(desktop)
    result = next(event for event in events(desktop) if event["event"] == "decision")
    assert result == {
        "event": "decision",
        "choice_index": 2,
        "answers": ["Third", "First", "Second"],
        "probabilities": [0.05, 0.05, 0.9],
        "input_tokens": 3,
    }
    assert desktop.worker is worker
    assert desktop.messages == []
    assert "synthetic decision" not in desktop.registry.read_text()
    assert set(json.loads(desktop.registry.read_text())[0]) == {"id", "name", "path", "source"}
    assert not any(key in result for key in ("logits", "token_ids", "candidate_token_ids"))
    ready = next(event for event in events(desktop) if event["event"] == "ready")
    assert ready["mode"] == "decision"


def test_manual_mode_switch_reaps_old_worker_and_reloads_selected_model(desktop, model, score_executable):
    identifier = running_decision(desktop, model, score_executable)
    score_process = desktop.worker.process
    desktop.dispatch({"command": "run", "model_id": identifier, "mode": "chat"})
    finish(desktop)
    assert score_process.poll() is not None
    assert desktop.active_mode == "chat"
    chat_process = desktop.worker.process
    desktop.dispatch({"command": "run", "model_id": identifier, "mode": "decision"})
    finish(desktop)
    assert chat_process.poll() is not None
    assert desktop.active_mode == "decision"
    assert desktop.worker.process.pid != score_process.pid


@pytest.mark.parametrize("count", [2, 16])
def test_decision_minimum_and_maximum_answer_sets_roundtrip(desktop, model, score_executable, count):
    running_decision(desktop, model, score_executable)
    labels = [f"Answer {index}" for index in range(count)]
    desktop.dispatch(decision_command(answers=labels))
    finish(desktop)
    result = next(event for event in events(desktop) if event["event"] == "decision")
    assert result["answers"] == labels
    assert len(result["probabilities"]) == count
    assert result["choice_index"] == count - 1


def test_mode_commands_are_guarded_and_invalid_mode_preserves_loaded_worker(desktop, model, score_executable):
    identifier = running_decision(desktop, model, score_executable)
    process = desktop.worker.process
    with pytest.raises(DesktopError, match="Choose Chat or Decision"):
        desktop.dispatch({"command": "run", "model_id": identifier, "mode": "unknown"})
    assert desktop.worker.process is process
    with pytest.raises(DesktopError, match="Chat mode"):
        desktop.dispatch(advanced_options())
    desktop.dispatch({"command": "generate", "text": "synthetic text"})
    finish(desktop)
    assert desktop.worker.process is process
    assert not any(event["event"] == "reply" for event in events(desktop))
    desktop.dispatch({"command": "run", "model_id": identifier})
    finish(desktop)
    desktop.dispatch(decision_command())
    finish(desktop)
    assert not any(event["event"] == "decision" for event in events(desktop))


@pytest.mark.parametrize("answers", [["single"], ["duplicate", " Duplicate "], ["first", ""], None])
def test_bad_decision_answers_reject_before_starting_or_writing_to_worker(desktop, model, score_executable, answers):
    running_decision(desktop, model, score_executable)
    with patch.object(desktop.worker, "request", wraps=desktop.worker.request) as request:
        with pytest.raises(DesktopError):
            desktop.dispatch(decision_command(answers=answers))
        request.assert_not_called()
    assert desktop.job is None


@pytest.mark.parametrize("text", ["failure fixture", "bad readout fixture"])
def test_decision_native_or_protocol_failure_fences_worker_without_diagnostic_payloads(
    desktop, model, score_executable, text
):
    running_decision(desktop, model, score_executable)
    process = desktop.worker.process
    desktop.dispatch(decision_command(text=text))
    finish(desktop)
    assert desktop.worker is None and process.poll() is not None
    assert desktop.active_mode is None
    assert not any(event["event"] == "decision" for event in events(desktop))
    assert text not in desktop.output.getvalue()
    assert "private exception fixture" not in desktop.output.getvalue()


def test_native_model_context_rejection_preserves_worker_and_allows_retry(desktop, model, score_executable):
    running_decision(desktop, model, score_executable)
    worker = desktop.worker
    desktop.dispatch(decision_command(text="context fixture"))
    finish(desktop)
    assert desktop.worker is worker
    assert not any(event["event"] == "decision" for event in events(desktop))
    assert any("context limit" in event.get("message", "") for event in events(desktop))
    desktop.dispatch(decision_command())
    finish(desktop)
    assert len([event for event in events(desktop) if event["event"] == "decision"]) == 1


def test_open_reasoning_boundary_is_recoverable_and_does_not_write_to_worker(desktop, model, score_executable):
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    running_decision(desktop, model, score_executable)
    worker, original_template, original_codec = desktop.worker, desktop.template, desktop.codec
    path = Path(__file__).resolve().parents[3] / "crates/legacy/hanashi/configs/rendering/qwen3-thinking.json"
    desktop.template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
        json.loads(path.read_text())["jinja"]["template"]
    )
    desktop.codec = {"end_of_thinking_tag": "\n</think>"}
    with patch.object(worker, "request", wraps=worker.request) as request:
        desktop.dispatch(decision_command(text="literal </think> in supplied evidence"))
        finish(desktop)
        request.assert_not_called()
    assert desktop.worker is worker and worker.process.poll() is None
    assert events(desktop)[-1]["state"] == "ready"
    assert any("starts in reasoning mode" in event.get("message", "") for event in events(desktop))
    assert not any(event["event"] == "decision" for event in events(desktop))
    desktop.template, desktop.codec = original_template, original_codec
    desktop.dispatch(decision_command(text="literal <think> without a closing tag"))
    finish(desktop)
    assert len([event for event in events(desktop) if event["event"] == "decision"]) == 1


def test_stop_cancels_complete_decision_and_fences_late_results(desktop, model, score_executable):
    import threading

    identifier = running_decision(desktop, model, score_executable)
    process = desktop.worker.process
    entered = threading.Event()
    request = desktop.worker.request

    def blocking_request(payload):
        if payload["operation"] == "score":
            entered.set()
        return request(payload)

    with patch.object(desktop.worker, "request", side_effect=blocking_request):
        desktop.dispatch(decision_command(text="wait fixture"))
        assert entered.wait(timeout=2)
        desktop.dispatch(decision_command())
        assert any("Wait for the current operation" in event.get("message", "") for event in events(desktop))
        started = time.monotonic()
        desktop.stop()
        assert time.monotonic() - started < 5
    assert process.poll() is not None
    assert not any(event["event"] == "decision" for event in events(desktop))
    desktop.dispatch({"command": "run", "model_id": identifier, "mode": "decision"})
    finish(desktop)
    assert desktop.worker.process.pid != process.pid


def advanced_options(**overrides):
    return {
        "command": "configure",
        "system_prompt": "custom system fixture",
        "context_tokens": 2048,
        "max_new_tokens": 128,
        "thinking": True,
        **overrides,
    }


def test_advanced_options_render_reset_and_generate_without_reloading(desktop, model):
    running(desktop, model)
    process = desktop.worker.process
    desktop.dispatch({"command": "generate", "text": "previous turn fixture"})
    finish(desktop)
    desktop.dispatch(advanced_options())
    assert desktop.messages == [{"role": "system", "content": "custom system fixture"}]
    assert desktop.worker.process is process
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    desktop.template = ImmutableSandboxedEnvironment().from_string(
        "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}thinking={{ enable_thinking }}"
    )
    with patch.object(desktop.worker, "request", wraps=desktop.worker.request) as request:
        desktop.dispatch({"command": "generate", "text": "new turn fixture"})
        finish(desktop)
        payload = request.call_args_list[-1].args[0]
        tokenization = request.call_args_list[0].args[0]
    assert payload["context_tokens"] == 2048
    assert payload["max_new_tokens"] == 128
    assert tokenization["text"] == "system:custom system fixture;user:new turn fixture;thinking=True"
    assert payload["input_token_ids"] == [2] * 20
    desktop.dispatch({"command": "reset"})
    assert desktop.messages == [{"role": "system", "content": "custom system fixture"}]
    assert "custom system fixture" not in desktop.registry.read_text()
    assert "custom system fixture" not in desktop.output.getvalue()


def test_advanced_options_model_default_empty_override_and_restart(desktop, model):
    identifier = running(desktop, model)
    desktop.dispatch(advanced_options(system_prompt=""))
    assert desktop.messages == []
    desktop.dispatch({"command": "run", "model_id": identifier})
    finish(desktop)
    assert desktop.messages == []
    assert (desktop.context, desktop.maximum, desktop.thinking) == (2048, 128, True)
    desktop.dispatch(advanced_options(system_prompt=None, thinking=False))
    assert desktop.messages == [{"role": "system", "content": "synthetic system"}]
    assert desktop.thinking is False
    replacement = Desktop(desktop.executable, desktop.data_dir, io.StringIO())
    assert replacement.system_prompt is None
    assert (replacement.context, replacement.maximum, replacement.thinking) == (8192, 512, False)


@pytest.mark.parametrize(
    "overrides",
    [
        {"context_tokens": True},
        {"context_tokens": 1},
        {"context_tokens": 8193},
        {"context_tokens": 100.5},
        {"context_tokens": "2048"},
        {"context_tokens": None},
        {"max_new_tokens": 0},
        {"max_new_tokens": -1},
        {"max_new_tokens": 4097},
        {"max_new_tokens": 2048},
        {"max_new_tokens": False},
        {"max_new_tokens": float("nan")},
        {"thinking": "true"},
        {"thinking": 1},
        {"system_prompt": []},
        {"unknown": "private fixture"},
    ],
)
def test_advanced_options_invalid_changes_are_atomic(desktop, model, overrides):
    running(desktop, model)
    before = list(desktop.messages)
    with pytest.raises(DesktopError, match="Invalid advanced options"):
        desktop.dispatch(advanced_options(**overrides))
    assert desktop.messages == before
    assert (desktop.context, desktop.maximum, desktop.thinking, desktop.system_prompt) == (8192, 512, False, None)
    assert not any(e["event"] == "configured" for e in events(desktop))


def test_advanced_options_reject_missing_fields_no_model_and_busy(desktop, model):
    with pytest.raises(DesktopError, match="Run a model"):
        desktop.dispatch(advanced_options())
    running(desktop, model)
    with pytest.raises(DesktopError, match="Invalid advanced options"):
        desktop.dispatch({"command": "configure"})
    desktop.dispatch({"command": "generate", "text": "wait fixture"})
    with pytest.raises(DesktopError, match="Wait for the current operation"):
        desktop.dispatch(advanced_options())
    desktop.dispatch({"command": "stop"})


def test_model_change_clamps_options_to_advertised_limits(desktop, model):
    identifier = running(desktop, model)
    desktop.dispatch(advanced_options())
    with patch(
        "stark_convert.desktop.Worker.read",
        return_value={
            "ready": True,
            "mode": "text-generation-v1",
            "max_context_tokens": 512,
            "max_new_tokens": 64,
        },
    ):
        desktop.dispatch({"command": "run", "model_id": identifier})
        finish(desktop)
    assert (desktop.context, desktop.maximum) == (512, 64)
    ready = [e for e in events(desktop) if e["event"] == "ready"][-1]
    assert ready["context_tokens"] == 512 and ready["max_new_tokens"] == 64
    assert ready["max_context_tokens"] == 512 and ready["max_output_tokens"] == 64


def test_output_limit_metadata_reserves_prompt_capacity(desktop, model):
    identifier = installed(desktop, model)
    with patch(
        "stark_convert.desktop.Worker.read",
        return_value={
            "ready": True,
            "mode": "text-generation-v1",
            "max_context_tokens": 512,
            "max_new_tokens": 4096,
        },
    ):
        desktop.dispatch({"command": "run", "model_id": identifier})
        finish(desktop)
    ready = [event for event in events(desktop) if event["event"] == "ready"][-1]
    assert ready["max_output_tokens"] == 511
    assert desktop.api_models()[0]["max_output_tokens"] == 511


def test_advanced_context_budget_rejection_leaves_history_unchanged(desktop, model):
    running(desktop, model)
    desktop.dispatch(advanced_options(context_tokens=21, max_new_tokens=2))
    before = list(desktop.messages)
    with patch.object(desktop.worker, "request", wraps=desktop.worker.request) as request:
        desktop.dispatch({"command": "generate", "text": "budget fixture"})
        finish(desktop)
        assert [c.args[0]["operation"] for c in request.call_args_list] == ["tokenize"]
    assert desktop.messages == before


def test_local_registration_is_private_atomic_persistent_and_deduplicated(desktop, model):
    identifier = installed(desktop, model)
    assert model.is_dir()
    assert desktop.models[0]["path"] == str(model)
    assert desktop.registry.stat().st_mode & 0o777 == 0o600
    assert desktop.data_dir.stat().st_mode & 0o777 == 0o700
    assert desktop.model_dir.stat().st_mode & 0o777 == 0o700
    assert not list(desktop.data_dir.glob(".models-*"))
    assert installed(desktop, model) == identifier
    replacement = Desktop(desktop.executable, desktop.data_dir, io.StringIO())
    assert replacement.models == desktop.models
    assert set(json.loads(desktop.registry.read_text())[0]) == {"id", "name", "path", "source"}


def test_generation_metrics_history_reset_and_stop(desktop, model):
    running(desktop, model)
    process = desktop.worker.process
    desktop.dispatch({"command": "generate", "text": "synthetic request"})
    finish(desktop)
    reply = next(e for e in events(desktop) if e["event"] == "reply")
    assert reply["text"] == "synthetic answer"
    assert reply["input_tokens_per_second"] == 100.0
    assert reply["output_tokens_per_second"] == 50.0
    assert len(desktop.messages) == 3
    assert "synthetic request" not in desktop.registry.read_text()
    desktop.dispatch({"command": "reset"})
    assert desktop.messages == [{"role": "system", "content": "synthetic system"}]
    desktop.dispatch({"command": "stop"})
    assert process.poll() is not None
    assert desktop.worker is None
    assert desktop.messages == []


def test_stop_interrupts_generation_then_restart_uses_a_new_worker(desktop, model):
    identifier = running(desktop, model)
    process = desktop.worker.process
    desktop.dispatch({"command": "generate", "text": "wait fixture"})
    started = time.monotonic()
    desktop.dispatch({"command": "stop"})
    assert time.monotonic() - started < 5
    assert process.poll() is not None
    assert not any(e["event"] == "reply" for e in events(desktop))
    desktop.dispatch({"command": "run", "model_id": identifier})
    finish(desktop)
    assert desktop.worker.process.pid != process.pid
    assert desktop.active_model == identifier


def test_worker_failure_fences_process_without_private_diagnostics(desktop, model):
    running(desktop, model)
    process = desktop.worker.process
    desktop.dispatch({"command": "generate", "text": "failure fixture"})
    finish(desktop)
    assert process.poll() is not None
    assert desktop.worker is None
    assert "failure fixture" not in desktop.output.getvalue()
    assert any(e["event"] == "error" for e in events(desktop))


def test_idle_worker_exit_is_reported_and_reaped(desktop, model):
    running(desktop, model)
    process = desktop.worker.process
    process.kill()
    process.wait()
    desktop.monitor()
    assert desktop.worker is None
    assert events(desktop)[-1]["state"] == "error"


def test_raw_conversion_is_offline_and_cancel_reaps_converter(desktop, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    commands = []
    original = subprocess.Popen

    def spawn(command, **kwargs):
        commands.append(command)
        return original([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)

    with patch("stark_convert.desktop.subprocess.Popen", side_effect=spawn):
        desktop.dispatch({"command": "install", "kind": "local", "source": str(checkpoint)})
        deadline = time.monotonic() + 3
        while desktop.converter is None and time.monotonic() < deadline:
            time.sleep(0.01)
        process = desktop.converter
        assert process is not None
        desktop.stop()
    assert process.poll() is not None
    assert "--offline" in commands[0]
    assert "hf-generic" in commands[0]
    assert desktop.models == []
    assert not list(desktop.model_dir.iterdir())


@pytest.mark.parametrize(
    "repo, generic", [("example/model", True), ("synthetic-model", True), ("Qwen/Qwen3-0.6B", False)]
)
@pytest.mark.parametrize(
    "source", ["{repo}", "https://huggingface.co/{repo}", " https://huggingface.co/{repo}/?download=true#model-card "]
)
def test_huggingface_conversion_publishes_complete_model(desktop, model, repo, generic, source):
    original = subprocess.Popen
    commands = []

    def spawn(command, **kwargs):
        commands.append(command)
        target = command[command.index("--output") + 1]
        return original(
            [sys.executable, "-c", "import shutil,sys;shutil.copytree(sys.argv[1],sys.argv[2])", str(model), target],
            **kwargs,
        )

    with patch("stark_convert.desktop.subprocess.Popen", side_effect=spawn):
        desktop.dispatch({"command": "install", "kind": "huggingface", "source": source.format(repo=repo)})
        finish(desktop)
        desktop.dispatch({"command": "install", "kind": "huggingface", "source": repo})
        finish(desktop)
        desktop.dispatch({"command": "install", "kind": "huggingface", "source": f"https://huggingface.co/{repo}"})
        finish(desktop)
    assert len(commands) == 1
    assert commands[0][commands[0].index("stark_convert.cli") + 1] == repo
    assert "--offline" not in commands[0]
    assert ("--metadata-profile" in commands[0]) is generic
    assert ("hf-generic" in commands[0]) is generic
    assert len(desktop.models) == 1
    assert desktop.models[0]["source"] == "huggingface:" + repo
    assert desktop.models[0]["name"] == repo.split("/")[-1]
    assert Path(desktop.models[0]["path"]).parent == desktop.model_dir
    assert not list(desktop.model_dir.glob(".install-*"))


def progress_events(desktop, phase=None):
    return [e for e in events(desktop) if e["event"] == "progress" and (phase is None or e["phase"] == phase)]


def progress_spawn(model, lines, *, exit_code=0, pause=False):
    original = subprocess.Popen

    def spawn(command, **kwargs):
        descriptor = command[command.index("--progress-fd") + 1]
        target = command[command.index("--output") + 1]
        script = (
            "import os,shutil,sys,time\n"
            "stream=os.fdopen(int(sys.argv[1]),'wb',buffering=0)\n"
            "for line in " + repr(lines) + ": stream.write(line+b'\\n')\n"
            "print('private diagnostic fixture',flush=True)\n"
        )
        if pause:
            script += "time.sleep(30)\n"
        if exit_code == 0:
            script += "shutil.copytree(sys.argv[2],sys.argv[3])\n"
        script += f"raise SystemExit({exit_code})\n"
        return original([sys.executable, "-c", script, descriptor, str(model), target], **kwargs)

    return spawn


def progress_line(phase, state, completed, total):
    return json.dumps({"phase": phase, "state": state, "completed_bytes": completed, "total_bytes": total}).encode()


def test_converter_progress_is_numeric_private_bounded_and_completed_after_publication(desktop, model):
    lines = [
        b"{malformed",
        b"private fixture" * 1000,
        json.dumps({"phase": "download", "state": ["private fixture"]}).encode(),
        json.dumps(
            {
                "phase": "download",
                "state": "running",
                "completed_bytes": 1,
                "total_bytes": 2,
                "private": "private fixture",
            }
        ).encode(),
        progress_line("download", "running", 10, 100),
        progress_line("download", "running", 50, 100),
        progress_line("download", "running", 20, 100),
        progress_line("download", "complete", 100, 100),
        progress_line("conversion", "running", 100, 200),
        progress_line("conversion", "complete", 200, 200),
    ]
    original_persist = desktop.persist

    def persist():
        assert not any(e["state"] == "complete" for e in progress_events(desktop, "conversion"))
        original_persist()

    with (
        patch("stark_convert.desktop.subprocess.Popen", side_effect=progress_spawn(model, lines)),
        patch.object(desktop, "persist", side_effect=persist),
    ):
        desktop.dispatch({"command": "install", "kind": "huggingface", "source": "example/model"})
        finish(desktop)
    assert len(desktop.models) == 1
    assert "private fixture" not in desktop.output.getvalue()
    assert "private diagnostic" not in desktop.output.getvalue()
    download = progress_events(desktop, "download")
    assert [e["completed_bytes"] for e in download] == [0, 10, 50, 100]
    conversion = progress_events(desktop, "conversion")
    assert [e["state"] for e in conversion] == ["waiting", "running", "running", "complete"]
    assert conversion[-1]["completed_bytes"] == conversion[-1]["total_bytes"] == 200
    for event in progress_events(desktop):
        assert set(event) == {"event", "phase", "state", "completed_bytes", "total_bytes"}
    all_events = events(desktop)
    assert all_events.index(conversion[-1]) < next(i for i, e in enumerate(all_events) if e["event"] == "installed")


def test_progress_skips_local_converted_and_duplicate_models(desktop, model):
    installed(desktop, model)
    assert [e["state"] for e in progress_events(desktop, "download")] == ["skipped"]
    assert progress_events(desktop, "conversion")[-1]["state"] == "skipped"
    desktop.output = io.StringIO()
    installed(desktop, model)
    assert progress_events(desktop, "download")[-1]["state"] == "skipped"
    assert progress_events(desktop, "conversion")[-1]["state"] == "skipped"


def test_failed_conversion_does_not_complete_progress_or_publish_model(desktop, model):
    lines = [progress_line("download", "complete", 100, 100), progress_line("conversion", "running", 5, 100)]
    with patch("stark_convert.desktop.subprocess.Popen", side_effect=progress_spawn(model, lines, exit_code=2)):
        desktop.dispatch({"command": "install", "kind": "huggingface", "source": "example/model"})
        finish(desktop)
    assert not desktop.models
    assert progress_events(desktop, "conversion")[-1]["state"] == "failed"
    assert not any(e["state"] == "complete" for e in progress_events(desktop, "conversion"))
    assert not list(desktop.model_dir.iterdir())


def test_registry_failure_keeps_conversion_progress_failed(desktop, model):
    lines = [progress_line("download", "complete", 100, 100), progress_line("conversion", "complete", 200, 200)]
    with (
        patch("stark_convert.desktop.subprocess.Popen", side_effect=progress_spawn(model, lines)),
        patch.object(desktop, "persist", side_effect=OSError("private fixture")),
    ):
        desktop.dispatch({"command": "install", "kind": "huggingface", "source": "example/model"})
        finish(desktop)
    assert not desktop.models
    assert progress_events(desktop, "conversion")[-1]["state"] == "failed"
    assert not any(e["event"] == "installed" for e in events(desktop))
    assert "private fixture" not in desktop.output.getvalue()
    assert not list(desktop.model_dir.iterdir())


def test_cancelled_progress_reaps_converter_and_cannot_emit_late_completion(desktop, model):
    lines = [progress_line("download", "running", 5, 100)]
    with patch("stark_convert.desktop.subprocess.Popen", side_effect=progress_spawn(model, lines, pause=True)):
        desktop.dispatch({"command": "install", "kind": "huggingface", "source": "example/model"})
        deadline = time.monotonic() + 3
        while not any(e["completed_bytes"] == 5 for e in progress_events(desktop)) and time.monotonic() < deadline:
            time.sleep(0.01)
        process = desktop.converter
        old_epoch = desktop.epoch
        assert process is not None
        desktop.stop()
        desktop.progress(old_epoch, "download", "complete", 100, 100)
    assert process.poll() is not None
    assert desktop.converter is None
    assert not desktop.models
    assert progress_events(desktop, "download")[-1]["state"] == "cancelled"
    assert progress_events(desktop, "conversion")[-1]["state"] == "cancelled"
    assert not any(e["state"] == "complete" for e in progress_events(desktop))
    assert not list(desktop.model_dir.iterdir())


def test_progress_pipe_descriptors_close_when_converter_spawn_fails(desktop, model):
    original_pipe = os.pipe
    descriptors = []

    def pipe():
        result = original_pipe()
        descriptors.extend(result)
        return result

    with (
        patch("stark_convert.desktop.os.pipe", side_effect=pipe),
        patch("stark_convert.desktop.subprocess.Popen", side_effect=OSError("private fixture")),
    ):
        desktop.dispatch({"command": "install", "kind": "huggingface", "source": "example/model"})
        finish(desktop)
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert progress_events(desktop, "download")[-1]["state"] == "failed"
    assert "private fixture" not in desktop.output.getvalue()
    assert not list(desktop.model_dir.iterdir())


def test_exited_converter_cannot_leave_a_descendant_holding_progress_pipe(desktop, model, tmp_path):
    original = subprocess.Popen
    child_pid_file = tmp_path / "descendant.pid"
    child_pid = None

    def spawn(command, **kwargs):
        descriptor = command[command.index("--progress-fd") + 1]
        target = command[command.index("--output") + 1]
        script = (
            "import os,shutil,subprocess,sys\n"
            "from pathlib import Path\n"
            "fd=int(sys.argv[1])\n"
            "code='import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)'\n"
            "child=subprocess.Popen([sys.executable,'-c',code],pass_fds=(fd,))\n"
            "Path(sys.argv[4]).write_text(str(child.pid))\n"
            "shutil.copytree(sys.argv[2],sys.argv[3])\n"
        )
        return original([sys.executable, "-c", script, descriptor, str(model), target, str(child_pid_file)], **kwargs)

    try:
        with patch("stark_convert.desktop.subprocess.Popen", side_effect=spawn):
            desktop.dispatch({"command": "install", "kind": "huggingface", "source": "example/model"})
            finish(desktop)
        child_pid = int(child_pid_file.read_text())
        deadline = time.monotonic() + 3
        while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
            try:
                if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)
        assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        assert len(desktop.models) == 1
        assert desktop.converter is None
    finally:
        if child_pid is None and child_pid_file.exists():
            child_pid = int(child_pid_file.read_text())
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize(
    "patch_value",
    [
        {"phase": []},
        {"phase": "other"},
        {"state": []},
        {"state": "other"},
        {"completed_bytes": True},
        {"completed_bytes": -1},
        {"completed_bytes": 3.5},
        {"completed_bytes": 11},
        {"total_bytes": False},
        {"total_bytes": float("nan")},
        {"total_bytes": -1},
        {"state": "complete"},
        {"private": "private fixture"},
    ],
)
def test_converter_progress_rejects_invalid_schema(patch_value):
    value = {"phase": "download", "state": "running", "completed_bytes": 5, "total_bytes": 10, **patch_value}
    assert converter_progress(value) is None


@pytest.mark.parametrize("payload", [None, [], {}, {"command": "wat"}, {"command": "generate", "text": 3}])
def test_malformed_commands_are_rejected(desktop, payload):
    with pytest.raises(DesktopError, match="Invalid app command"):
        desktop.dispatch(payload)


def test_missing_local_and_invalid_hf_sources_report_fixed_errors(desktop):
    for kind, source in [("local", "/missing synthetic fixture"), ("huggingface", "https://invalid/example")]:
        desktop.dispatch({"command": "install", "kind": kind, "source": source})
        finish(desktop)
    assert len([e for e in events(desktop) if e["event"] == "error"]) == 2
    assert desktop.models == []


def test_old_or_nonfinite_metrics_are_unavailable():
    for value in [None, float("inf"), float("nan"), -1, True, "200"]:
        assert metric({"speed": value}, "speed") is None
    assert metric({}, "speed") is None
    assert metric({"speed": 0}, "speed") == 0


def test_disappearing_event_consumer_does_not_expose_exception_details(desktop):
    desktop.output.close()
    desktop.library()
    assert desktop.output_failed


@pytest.mark.parametrize("exit_mode", ["eof", "sigterm"])
def test_app_exit_reaps_active_native_worker(tmp_path, model, executable, exit_mode):
    library = tmp_path / "exit library"
    prepare = Desktop(executable, library, io.StringIO())
    installed(prepare, model)
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "stark_convert.desktop",
            "--worker",
            str(executable),
            "--data-dir",
            str(library),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert json.loads(process.stdout.readline())["event"] == "models"
        process.stdout.readline()
        process.stdin.write(json.dumps({"command": "run", "model_id": prepare.models[0]["id"]}) + "\n")
        process.stdin.flush()
        while json.loads(process.stdout.readline())["event"] != "ready":
            pass
        children = psutil.Process(process.pid).children()
        assert len(children) == 1
        child_pid = children[0].pid
        if exit_mode == "eof":
            process.stdin.close()
        else:
            process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 0
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert process.stderr.read() == ""
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
