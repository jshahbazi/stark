"""Finite JevK5 fixture rendering and typed readout for private acceptance.

Adapted from JevK5 (Copyright 2026 the JevK5 authors), Apache-2.0.
The decision prompt derives from SemIf (Copyright TheoLeeCJ), MIT.
See licenses/jevk5-LICENSE and licenses/jevk5-NOTICE. Modified to retain only
the single-pass, at-most-sixteen-option path used by this acceptance corpus.
"""
import json

LETTERS = "ABCDEFGHIJKLMNOP"
SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
          "Respond with only its uppercase letter, with no explanation or reasoning.")
CHAT_TEMPLATE = ("<|im_start|>system\n{system}<|im_end|>\n"
                 "<|im_start|>user\n{user}<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def decision_options(question):
    criteria = question.get("criteria")
    if question["type"] == "noul":
        pairs = [(key, (criteria or {}).get(key) or f"The proposition is {key}.") for key in ("true", "false")]
    elif question["type"] == "choice":
        if isinstance(criteria, list):
            criteria = dict.fromkeys(criteria)
        pairs = [(key, value or key) for key, value in criteria.items()]
    elif question["type"] == "score":
        pairs = [(str(index), level) for index, level in enumerate(criteria)]
    else:
        raise ValueError("fixture_question_invalid")
    if not 2 <= len(pairs) <= len(LETTERS):
        raise ValueError("fixture_option_count_invalid")
    return [(key, f"{key}: {description}") for key, description in pairs]


def prompt_text(state, criterion, options):
    payload = {"evidence": state, "criterion": criterion,
               "options": [{"letter": LETTERS[index], "description": description} for index, description in enumerate(options)]}
    return CHAT_TEMPLATE.format(system=SYSTEM, user=json.dumps(payload, ensure_ascii=False))


def answer(question, probabilities, tokens):
    kind = question["type"]
    output = {"type": kind, "confidence": max(probabilities.values()), "input_tokens": tokens}
    if kind == "noul":
        output["noul"] = probabilities["true"]
    elif kind == "choice":
        output.update(choice=max(probabilities, key=probabilities.get), probabilities=probabilities)
    else:
        output.update(score=sum(int(key) * value for key, value in probabilities.items()), probabilities=probabilities)
    return output
