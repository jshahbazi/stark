"""Explicit self-contained metadata for future checkpoints of known architectures.

Architecture support does not determine reasoning, parser, system-prompt or
external metadata defaults. Generic conversion reads only the chosen source.
"""

from .registry import CONFIG_ADAPTERS
from .streaming import ConversionError

GENERIC_PROFILE = "hf-generic"
GENERIC_REQUIRED = {
    "model_config": "config.json",
    "tokenizer": "tokenizer.json",
    "tokenizer_config": "tokenizer_config.json",
}
GENERIC_OPTIONAL = {
    "generation_config": "generation_config.json",
    "chat_template": "chat_template.jinja",
}


def generic_files(repo: str | None, names: tuple[str, ...]) -> tuple[dict, ...]:
    """Pin required sidecars and the presence of every supported optional sidecar."""
    if not set(GENERIC_REQUIRED.values()) <= set(names):
        raise ConversionError("Generic metadata source is incomplete")
    selected = {**GENERIC_REQUIRED, **{role: name for role, name in GENERIC_OPTIONAL.items() if name in names}}
    return tuple({"role": role, "repo": repo, "filename": name} for role, name in sorted(selected.items()))


def generic_entry(repo: str | None, architecture: str, names: tuple[str, ...]) -> dict:
    """Describe plain metadata without inheriting a registered model's defaults."""
    config_class = next((key for key, value in CONFIG_ADAPTERS.items() if value == architecture), None)
    if config_class is None:
        raise ConversionError("Generic metadata architecture is unsupported")
    entry = {
        "repo": repo,
        "config_class": config_class,
        "metadata_profile": GENERIC_PROFILE,
        "files": {rule["role"]: {"repo": repo, "filename": rule["filename"]} for rule in generic_files(repo, names)},
        "metadata": {
            "parser_id": None,
            "reasoning_id": None,
            "end_of_thinking_id": None,
            "roles": {"system": "system", "user": "user", "assistant": "assistant"},
        },
    }
    if architecture == "modernbert":
        entry["model_kind"] = "ClassifierModelSpec"
    return entry
