"""Image-enabled terminal chat lifecycle using synthetic images and workers."""

import json
from argparse import Namespace
from unittest.mock import patch

import pytest

from stark_convert.chat import ChatError, main, run_chat
from stark_convert.media import ImageData

PNG = b"\x89PNG\r\n\x1a\nsynthetic"


@pytest.fixture
def model(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "token_codec_config": {
                    "type": "ChatCodecConfig",
                    "bos_token": "<bos>",
                    "default_system_prompt": "synthetic system",
                    "prompt_template": (
                        "{{ bos_token }}{% for m in messages %}{{ m.role }}:"
                        "{% if m.content is string %}{{ m.content }}{% else %}"
                        "{% for part in m.content %}{% if part.type == 'image' %}[image]"
                        "{% else %}{{ part.text }}{% endif %}{% endfor %}{% endif %};"
                        "{% endfor %}assistant:"
                    ),
                }
            }
        )
    )
    return tmp_path


class FakeWorker:
    def __init__(self, *, images=True):
        self.images = images
        self.requests = []
        self.image_requests = []
        self.closed = False

    def read(self):
        return {
            "ready": True,
            "mode": "text-generation-v1",
            "max_context_tokens": 8192,
            "max_new_tokens": 4096,
        }

    def request(self, payload):
        self.requests.append(payload)
        return {"token_count": 20} if payload["operation"] == "tokenize" else {"text": "synthetic reply"}

    def request_image(self, prompt, image, options, on_delta=None, cancelled=None):
        self.image_requests.append((prompt, image, dict(options)))
        return {
            "text": "synthetic image reply",
            "input_tokens": 24,
            "output_tokens": 3,
            "finish_reason": "eos",
        }

    def close(self):
        self.closed = True


def options(model, **overrides):
    values = {
        "model_path": model,
        "image": None,
        "context_tokens": 1024,
        "max_new_tokens": 32,
        "system": None,
        "thinking": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_image_cli_attaches_placeholder_once_and_reuses_owned_image_after_reset(model, tmp_path):
    image_path = tmp_path / "synthetic.png"
    image_path.write_bytes(PNG)
    worker = FakeWorker()
    prompts = "first question\nsecond question\n/reset\nthird question\n/exit\n"

    with patch("builtins.input", side_effect=prompts.splitlines()):
        assert run_chat(options(model, image=image_path), worker) == 0

    assert [entry[0] for entry in worker.image_requests] == [
        "<bos>system:synthetic system;user:[image]first question;assistant:",
        "<bos>system:synthetic system;user:[image]first question;assistant:synthetic image reply;"
        "user:second question;assistant:",
        "<bos>system:synthetic system;user:[image]third question;assistant:",
    ]
    assert all(entry[1] is worker.image_requests[0][1] for entry in worker.image_requests)
    image = worker.image_requests[0][1]
    assert isinstance(image, ImageData)
    assert image.data == PNG and image.media_type == "image/png"
    assert all(
        entry[2] == {"context_tokens": 1024, "max_new_tokens": 32, "sampling": {"mode": "greedy"}}
        for entry in worker.image_requests
    )
    assert worker.requests == []


def test_image_option_requires_enabled_images_before_reading_path(model, tmp_path):
    worker = FakeWorker(images=False)
    with pytest.raises(ChatError, match="unavailable"):
        run_chat(options(model, image=tmp_path / "does-not-exist.png"), worker)


@pytest.mark.parametrize("payload", [b"not an image", b"\xff\xd8\xffsynthetic-no-eoi"])
def test_invalid_image_file_has_fixed_error_without_path(model, tmp_path, payload):
    image_path = tmp_path / "private-name.png"
    image_path.write_bytes(payload)
    worker = FakeWorker()
    with pytest.raises(ChatError, match="Image file is invalid") as error:
        run_chat(options(model, image=image_path), worker)
    assert str(image_path) not in str(error.value)


def test_main_parses_image_path_and_cleans_worker(model, tmp_path, monkeypatch, capsys):
    image_path = tmp_path / "synthetic.png"
    image_path.write_bytes(PNG)
    worker = FakeWorker()
    monkeypatch.setattr("stark_convert.chat.Worker", lambda *args: worker)
    inputs = iter(["question", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    assert main(["--model-path", str(model), "--image", str(image_path)]) == 0
    assert len(worker.image_requests) == 1
    assert worker.closed
    output = capsys.readouterr()
    assert str(image_path) not in output.out + output.err
    assert "synthetic image reply" in output.out


def test_image_file_must_be_regular_and_size_is_bounded(model, tmp_path, monkeypatch):
    from stark_convert import chat

    image_path = tmp_path / "too-large.png"
    image_path.write_bytes(PNG + b"x" * 16)
    monkeypatch.setattr(chat, "MAX_IMAGE_BYTES", len(PNG) + 2)
    worker = FakeWorker()
    with pytest.raises(ChatError, match="Image file is invalid"):
        run_chat(options(model, image=image_path), worker)

    directory = tmp_path / "directory.png"
    directory.mkdir()
    with pytest.raises(ChatError, match="Image file is invalid"):
        run_chat(options(model, image=directory), worker)
