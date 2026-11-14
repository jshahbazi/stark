"""Templates without a system role fold one leading system message into the first user turn.

Model-free. Synthetic templates only, plus the published Mistral 7B v0.3 MLX
export's `default` entry read from a local snapshot when present (referenced by
hash, never committed).
"""

import hashlib
import json
import os
from pathlib import Path

import pytest
from test_auto_dtype import FUTURE, source

from stark_convert import cli
from stark_convert.chat import ChatError, _ChatTemplate, load_template
from stark_convert.sources import resolve_source
from stark_convert.template_compat import normalize_chat_template
from stark_convert.template_runtime import (
    SYSTEM_ROLE_FOLD,
    SYSTEM_ROLE_POLICY,
    IsolatedTemplate,
    SystemRoleError,
    TemplateRuntimeError,
    fold_system_message,
    probe_system_role,
)

# Same structure as the refusing program: strict alternation, no system role.
ALTERNATING = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('roles') }}{% endif %}"
    "{% if message['role'] == 'user' %}{{ '[U]' + message['content'] + '[/U]' }}"
    "{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}"
    "{% else %}{{ raise_exception('roles') }}{% endif %}{% endfor %}"
)
ACCEPTING = "{% for message in messages %}<{{ message['role'] }}>{{ message['content'] }}{% endfor %}"
NO_USER_TURNS = "{{ raise_exception('never') }}"
PUBLISHED_DEFAULT_SHA256 = "a6fc0e6bc936bd3aecf4b1e4c1208778b2f916f7e55e01bccd3b232f976b9216"
SNAPSHOT = Path(
    os.environ.get(
        "STARK_MISTRAL_MLX_SNAPSHOT",
        "/Users/johnshahbazian/development/stark/.claude/worktrees/agent-a7362ac2d0c2919ce/runtime/"
        "dense-mlx-qwen2-mistral/sources/mistral-7b-v03-instruct-mlx4",
    )
)
SYSTEM = {"role": "system", "content": "System text."}
USER = {"role": "user", "content": "First user text."}
ASSISTANT = {"role": "assistant", "content": "Assistant text."}
LATER = {"role": "user", "content": "Later user text."}
COMMON = {"add_generation_prompt": True, "bos_token": "<s>", "eos_token": "</s>"}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_conversations_without_system_messages_are_returned_unchanged():
    messages = [USER, ASSISTANT, LATER]
    assert fold_system_message(messages) is messages
    assert fold_system_message(None) is None


def test_fold_targets_the_first_user_message_with_a_fixed_separator():
    folded = fold_system_message([SYSTEM, USER, ASSISTANT, LATER])
    assert folded == [{"role": "user", "content": "System text.\n\nFirst user text."}, ASSISTANT, LATER]
    assert SYSTEM_ROLE_POLICY == {"policy": "fold_into_first_user", "separator": "\n\n"}


@pytest.mark.parametrize(
    "messages,reason",
    [
        ([SYSTEM, USER, dict(SYSTEM)], "system_message_repeated"),
        ([USER, SYSTEM], "system_message_not_first"),
        ([{"role": "system", "content": [{"type": "text", "text": "x"}]}, USER], "system_content_not_text"),
        ([{"role": "system", "content": None}, USER], "system_content_not_text"),
        ([SYSTEM], "system_message_without_user"),
        ([SYSTEM, ASSISTANT, USER], "system_message_without_user"),
        ([SYSTEM, {"role": "user", "content": [{"type": "text", "text": "x"}]}], "user_content_not_text"),
    ],
)
def test_fold_fails_closed(messages, reason):
    with pytest.raises(SystemRoleError) as error:
        fold_system_message(messages)
    assert error.value.reason == reason


def test_probe_detects_only_templates_that_refuse_the_system_role():
    assert probe_system_role(ALTERNATING, bos_token="<s>", eos_token="</s>") == SYSTEM_ROLE_POLICY
    assert probe_system_role(ACCEPTING) is None
    # A program that cannot render a plain user turn is not classified.
    assert probe_system_role(NO_USER_TURNS) is None
    # A program that also refuses the folded turns gets no policy.
    refuses_long_user = (
        "{% for message in messages %}{% if message['role'] != 'user' or message['content'] | length > 20 %}"
        "{{ raise_exception('no') }}{% endif %}{{ message['content'] }}{% endfor %}"
    )
    assert probe_system_role(refuses_long_user) is None


def test_probe_detects_the_published_mistral_default_entry():
    path = SNAPSHOT / "tokenizer_config.json"
    if not path.is_file():
        pytest.skip("published Mistral MLX snapshot is not available locally")
    entries = json.loads(path.read_text())["chat_template"]
    default = normalize_chat_template(next(item["template"] for item in entries if item["name"] == "default"))
    assert digest(default) == PUBLISHED_DEFAULT_SHA256
    assert probe_system_role(default, bos_token="<s>", eos_token="</s>") == SYSTEM_ROLE_POLICY
    template = IsolatedTemplate(default, system_role=SYSTEM_ROLE_FOLD)
    first = template.render(messages=[SYSTEM, USER], **COMMON)
    second = template.render(messages=[SYSTEM, USER, ASSISTANT, LATER], **COMMON)
    assert second.startswith(first) and len(second) > len(first)
    with pytest.raises(TemplateRuntimeError):
        IsolatedTemplate(default).render(messages=[SYSTEM, USER], **COMMON)


def test_folded_rendering_is_a_stable_prefix_across_turns():
    template = IsolatedTemplate(ALTERNATING, system_role=SYSTEM_ROLE_FOLD)
    conversation = [SYSTEM, USER]
    renders = [template.render(messages=conversation, **COMMON)]
    for index in range(3):
        conversation = [
            *conversation,
            {"role": "assistant", "content": f"Reply {index}."},
            {"role": "user", "content": f"Turn {index}."},
        ]
        renders.append(template.render(messages=conversation, **COMMON))
    assert all(later.startswith(earlier) for earlier, later in zip(renders, renders[1:], strict=False))
    assert renders[0] == template.render(
        messages=[{"role": "user", "content": "System text.\n\nFirst user text."}], **COMMON
    )
    # Folding into the last user message instead would rewrite earlier turns.
    last_fold = IsolatedTemplate(ALTERNATING)
    early = last_fold.render(messages=[{"role": "user", "content": "System text.\n\nFirst user text."}], **COMMON)
    late = last_fold.render(
        messages=[USER, ASSISTANT, {"role": "user", "content": "System text.\n\nLater user text."}], **COMMON
    )
    assert not late.startswith(early)


def test_without_the_policy_rendering_is_unchanged():
    plain = IsolatedTemplate(ACCEPTING)
    folded = IsolatedTemplate(ACCEPTING, system_role=SYSTEM_ROLE_FOLD)
    messages = [USER, ASSISTANT, LATER]
    assert plain.render(messages=messages) == folded.render(messages=messages)
    assert "<system>" in plain.render(messages=[SYSTEM, USER])
    with pytest.raises(TemplateRuntimeError):
        IsolatedTemplate(ACCEPTING, system_role="fold_into_last_user")


def test_chat_template_reports_each_fold_failure_clearly():
    template = _ChatTemplate(ALTERNATING, system_role=SYSTEM_ROLE_FOLD)
    with pytest.raises(ChatError, match="send at most one system message"):
        template.render(messages=[SYSTEM, USER, dict(SYSTEM)], **COMMON)
    with pytest.raises(ChatError, match="a user message must follow the system message"):
        template.render(messages=[SYSTEM], **COMMON)


def artifact(path, template, receipt):
    path.mkdir()
    config = {
        "type": "LanguageModelConfig",
        "decoder_config": {},
        "token_codec_config": {"type": "ChatCodecConfig", "prompt_template": template, "bos_token": "<s>"},
    }
    (path / "config.json").write_text(json.dumps(config))
    if receipt is not None:
        (path / "conversion.json").write_text(json.dumps(receipt))
    return path


def test_loader_honours_only_the_exact_recorded_policy(tmp_path):
    folded = artifact(tmp_path / "folded", ALTERNATING, {"chat_template_system_role": SYSTEM_ROLE_POLICY})
    _codec, template = load_template(folded)
    assert template.render(messages=[SYSTEM, USER], **COMMON) == template.render(
        messages=[{"role": "user", "content": "System text.\n\nFirst user text."}], **COMMON
    )
    plain = artifact(tmp_path / "plain", ALTERNATING, {})
    _codec, template = load_template(plain)
    with pytest.raises(ChatError, match="unsupported or exceeded"):
        template.render(messages=[SYSTEM, USER], **COMMON)
    for name, policy in (
        ("unknown", {"policy": "fold_into_last_user", "separator": "\n\n"}),
        ("separator", {"policy": "fold_into_first_user", "separator": " "}),
        ("string", "fold_into_first_user"),
    ):
        broken = artifact(tmp_path / name, ALTERNATING, {"chat_template_system_role": policy})
        with pytest.raises(ChatError, match="Cannot read chat metadata"):
            load_template(broken)


def convert(tmp_path, path, name):
    snapshot = resolve_source(path, repo=FUTURE, metadata_profile="hf-generic")
    return cli.convert(snapshot, tmp_path / name, repo=FUTURE, chunk_bytes=4096)


def rewrite_template(path, value):
    config = json.loads((path / "tokenizer_config.json").read_text())
    config["chat_template"] = value
    (path / "tokenizer_config.json").write_text(json.dumps(config))


def test_conversion_records_the_policy_only_for_refusing_templates(tmp_path):
    path = source(tmp_path, "qwen2-f16")
    rewrite_template(path, ALTERNATING)
    refusing = convert(tmp_path, path, "refusing")
    assert refusing["chat_template_system_role"] == SYSTEM_ROLE_POLICY
    stored = json.loads((tmp_path / "refusing/conversion.json").read_text())
    assert stored["chat_template_system_role"] == SYSTEM_ROLE_POLICY
    rewrite_template(path, ACCEPTING)
    accepting = convert(tmp_path, path, "accepting")
    assert "chat_template_system_role" not in accepting
    # The flag never reaches config.json, which the native loader reads strictly.
    for name in ("refusing", "accepting"):
        stored_config = (tmp_path / name / "config.json").read_text()
        assert "chat_template_system_role" not in stored_config and SYSTEM_ROLE_FOLD not in stored_config
    _codec, template = load_template(tmp_path / "refusing")
    assert template.render(messages=[SYSTEM, USER], **COMMON)
