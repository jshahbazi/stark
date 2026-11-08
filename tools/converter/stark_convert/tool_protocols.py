"""Reviewed additional tool programs and their render-only history adapters.

Source revisions and unchanged fixtures are recorded under tests/fixtures/tool_templates.
These registrations describe protocols, never checkpoint task competence.
"""

import hashlib
import json
import keyword
import re
from importlib.resources import files

from .template_compat import MAX_TEMPLATE_BYTES, template_program_sha256

# Raw source and converter-normalized source hashes. The converter removes only
# Transformers' generation-tracking tags; the original fixtures stay unchanged.
TEMPLATE_HASHES = {
    "5504d163d879bfb6fd4135b7fa3d56768b2e69dd718338f7b146750289f46ab0": "lfm2-pythonic-v1",
    "db61fb01017bd82401d3ffca4f8e066cd56ff6d38d2a30c4058770c0bf7ab49b": "functiongemma-v1",
    "0dbaf70819c970c9bacd7125be8e9db46b7bc0e6ed3bc303d92bd58fb9dd604a": "functiongemma-v1",
    "febf589225c9728ab791f52e8897d7607a823d45368f0a4c92fa68997b40cce9": "phi4-json-v1",
    "08962c2f15d56767854b46dfc4070b37f4c443551833bba65b417191735f3187": "granite33-json-v1",
    "e16746b40344d6c5b5265988e0328a0bf7277be86f1c335156eae07e29c82826": "mistral-json-v1",
    "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4": "llama-json-v1",
    "e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65": "llama-json-v1",
    "0bea7711316005a1c37c9941a7ee2a26311d223c979abee416aab0a402fc5b56": "granite33-json-v1",
    "09d68eeb5c0eaa551adea6cc088a05c7298966f395f203126e76b74f5a77a8ff": "granite30-json-v1",
    "872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788": "smollm2-json-v1",
    "3e0e275a527f249670056e0078e474eadbdd007c134b4fdfae5a7b3089702814": "smollm3-hermes-v1",
    "5389af391984d661fc19bb01456dd698df94babd9e054e6656f1953c586167d0": "smollm3-hermes-v1",
    "ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6": "lfm2-pythonic-v1",
    "2e7245c2c41abbc5cb19e67f18ee7d7999bd88f6f74eca7eef80362b0d124e24": "lfm2-pythonic-v1",
    "88ad4f413fb9967035d2d6bfc0570266adcf0bfc9d5501278ed3da00bf1b6397": "qwen3-hermes-v1",
    "2dbcdd619c0c5830b42a1a1d08de864a3e9df6a2d7bffa0e981b7dcf93892cd9": "qwen3-hermes-v1",
}
TEMPLATE_PROGRAMS = {
    "55ff13408d61d3cde699acb45fddd155abbccc68b943fdc1bcc89561d39b0b28": "lfm2-pythonic-v1",
    "16eb40376f3dd05b4fa6adfa6ec4603d8c03ace3d91a4a83d55a0a1cd07c9b6d": "functiongemma-v1",
    "5bf9ad01e979cbe145818337ff144bc8cb280793c63b063aee03e404b1e0b360": "functiongemma-v1",
    "1cf2a9d562d08814b91466fa42c56c8a656d13c1088d29c1add5d1ac21b84009": "phi4-json-v1",
    "fe4cefe90a431b88138e882192891494712dc65c9bd2d50264e984c957ee10a8": "granite33-json-v1",
    "152f9ceacd72be7c708cb2ec9e317f243d8272a9f404a1452797ae2492a184a2": "mistral-json-v1",
    "ac8a185b982fa2d2457459bbe86bd64680248fe536dae5531cee537c36f9ab6b": "llama-json-v1",
    "c9a73db2f091123af3ea6862594741bf44c5fc77241603bdd11fb9ba01059734": "llama-json-v1",
    "b55e692ca03fc5a1791041c5b804b5d46689b76942f9c7d1d42b79a0de349545": "granite33-json-v1",
    "bf894d352517d84d8ba8052ef76c597bad4d04e208d2329aa44718e3e34bb7cc": "granite30-json-v1",
    "ff802fefdd364f94905651d6f8abf16d9001cb8b4d8b1fb9b72f1cab2645fecf": "smollm2-json-v1",
    "96229b4a4c5f52570ad056a7d53857a52675848e1d4ac983d6922678fbd4c5ed": "smollm3-hermes-v1",
    "e3b6d05880efbac801343bb51b37397624dba782d869b6a6a13a43e9d9a64501": "lfm2-pythonic-v1",
    "0934a9a5ef6b9e105bbe456ab6273d386fb10a74cb397d191c2a8f7779f01c43": "qwen3-hermes-v1",
}

# (opening marker, closing marker, payload kind, argument field)
FORMATS = {
    "functiongemma-v1": ("<start_function_call>", "<end_function_call>", "functiongemma", "arguments"),
    "mistral-json-v1": ("[TOOL_CALLS]", "", "list", "arguments"),
    "llama-json-v1": ("<|python_tag|>", "", "object", "parameters"),
    "granite33-json-v1": ("<|tool_call|>", "", "list", "arguments"),
    "granite30-json-v1": ("<|tool_call|>", "", "list", "arguments"),
    "smollm2-json-v1": ("<tool_call>", "</tool_call>", "list", "arguments"),
    "lfm2-pythonic-v1": ("<|tool_call_start|>", "<|tool_call_end|>", "pythonic", "arguments"),
    "phi4-json-v1": ("<|tool_call|>", "<|/tool_call|>", "list", "arguments"),
}
_FG_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
_FG_RESERVED = ("<escape>", "<start_function_", "<end_function_")
_FG_TYPES = {"string", "boolean", "null", "integer", "number", "array", "object"}


def functiongemma_values_supported(value):
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str) and any(marker in item for marker in _FG_RESERVED):
            return False
        if isinstance(item, dict):
            if any(not isinstance(key, str) or _FG_KEY.fullmatch(key) is None for key in item):
                return False
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return True


def validate_functiongemma_tools(tools):
    from .local_api import APIError

    def fail():
        raise APIError(
            400,
            "This tool format requires named parameters with concrete types and representable keys and strings.",
            "tool_schema_unsupported",
        )

    for tool in tools:
        function = tool["function"]
        pending = [function]
        while pending:
            item = pending.pop()
            if isinstance(item, str) and any(marker in item for marker in _FG_RESERVED):
                fail()
            if isinstance(item, dict):
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
        parameters = dict(function.get("parameters", {}))
        parameters.setdefault("type", "object")
        schemas = [parameters]
        while schemas:
            schema = schemas.pop()
            if (
                not isinstance(schema, dict)
                or not isinstance(schema.get("type"), str)
                or schema["type"] not in _FG_TYPES
            ):
                fail()
            if {"$ref", "anyOf", "oneOf", "allOf", "patternProperties"}.intersection(schema):
                fail()
            if schema["type"] == "object":
                if schema.get("additionalProperties", False) is not False:
                    fail()
                properties = schema.get("properties", {})
                if not isinstance(properties, dict) or any(
                    _FG_KEY.fullmatch(key) is None
                    or key in {"description", "type", "properties", "required", "nullable"}
                    for key in properties
                ):
                    fail()
                schemas.extend(properties.values())
            if schema["type"] == "array" and schema.get("items"):
                schemas.append(schema["items"])


def runtime_template_source(source):
    """Apply the reviewed Mistral serializer patch, keeping source provenance."""
    if not isinstance(source, str) or len(source) > MAX_TEMPLATE_BYTES:
        return source
    try:
        encoded = source.encode("utf-8")
    except UnicodeEncodeError:
        return source
    if len(encoded) > MAX_TEMPLATE_BYTES:
        return source
    digest = hashlib.sha256(encoded).hexdigest()
    if (
        TEMPLATE_HASHES.get(digest) == "mistral-json-v1"
        or TEMPLATE_PROGRAMS.get(template_program_sha256(source)) == "mistral-json-v1"
    ):
        patched = files("stark_convert").joinpath("chat_templates/mistral_v03_tools.jinja").read_text(encoding="utf-8")
        if (
            hashlib.sha256(patched.encode()).hexdigest()
            != "dfbff0166eb003d00c7a23594654f9783f70efab223f91e2bddc0a8e0fa059ed"
        ):
            raise ValueError("Reviewed template patch changed")
        return patched
    return source


_SMOLLM2_INSTRUCTIONS = (
    "You are an expert in composing functions. You are given a question and a set of possible functions.\n"
    "Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\n"
    "If none of the functions can be used, point it out and refuse to answer.\n"
    "If the given question lacks the parameters required by the function, also point it out.\n\n"
    "You have access to the following tools:\n<tools>{tools}</tools>\n\n"
    "The output MUST strictly adhere to the following format, and NO other text MUST be included.\n"
    "The example format is as follows. Please make sure the parameter type is correct. "
    "If no function call is needed, please make the tool calls an empty list '[]'.\n"
    '<tool_call>[\n{{"name": "func_name1", "arguments": {{"argument1": "value1", "argument2": "value2"}}}},\n'
    "... (more tool calls as required)\n]</tool_call>"
)


def adapt_tools(tools, parser):
    if not tools:
        return None
    if parser == "functiongemma-v1":
        return [
            {
                **tool,
                "function": {
                    **tool["function"],
                    "parameters": {"type": "object", **tool["function"].get("parameters", {})},
                },
            }
            for tool in tools
        ]
    if parser == "smollm3-hermes-v1":
        return [json.dumps(tool["function"], ensure_ascii=False, allow_nan=False) for tool in tools]
    if parser in {"granite33-json-v1", "granite30-json-v1", "lfm2-pythonic-v1"}:
        return [dict(tool["function"]) for tool in tools]
    if parser == "mistral-json-v1":
        return [
            {
                **tool,
                "function": {
                    key: json.dumps(value, ensure_ascii=False)[1:-1] if isinstance(value, str) else value
                    for key, value in tool["function"].items()
                },
            }
            for tool in tools
        ]
    return tools


def protocol_stops(parser):
    return ["<start_function_response>"] if parser == "functiongemma-v1" else []


def protocol_stop_ids(worker, parser, *, active, cancelled=None):
    """Use the tokenizer's native stop token, preserving four client strings."""
    from .local_api import APIError

    result = []
    for marker in protocol_stops(parser) if active else []:
        reply = worker.request(
            {"operation": "tokenize", "text": marker, "add_special_tokens": False},
            **({"cancelled": cancelled} if cancelled is not None else {}),
        )
        ids = reply.get("token_ids")
        if not isinstance(ids, list) or any(type(identifier) is not int or identifier < 0 for identifier in ids):
            raise ValueError("Invalid tokenizer reply")
        if reply.get("token_count") != 1 or len(ids) != 1:
            raise APIError(
                400,
                "This model's tokenizer cannot represent the required tool stop marker.",
                "tool_tokenizer_unsupported",
            )
        result.extend(ids)
    return result


def adapt_history(messages, parser, tools):
    from .local_api import APIError

    def unsupported():
        return APIError(
            400, "This model's template supports one tool call per assistant turn.", "tool_history_unsupported"
        )

    identifiers = {}
    call_names = {}
    for message in messages:
        calls = message.get("tool_calls", [])
        if parser == "lfm2-pythonic-v1" and any(
            any(
                not name.isascii() or len(name) > 64 or not name.isidentifier() or keyword.iskeyword(name)
                for name in [call["function"]["name"], *call["function"]["arguments"]]
            )
            for call in calls
        ):
            raise APIError(400, "This tool history has names this model cannot represent.", "tool_history_unsupported")
        if parser == "llama-json-v1" and len(calls) > 1:
            raise unsupported()
        if calls and parser in {"mistral-json-v1", "llama-json-v1"} and (message.get("content") or "").strip():
            raise APIError(
                400, "This model requires a dedicated tool-call assistant turn.", "tool_history_unsupported"
            )
        for call in calls:
            identifiers[call["id"]] = "s" + format(len(identifiers), "08x")
            call_names[call["id"]] = call["function"]["name"]
        if parser == "functiongemma-v1":
            if calls and any(not functiongemma_values_supported(call["function"]["arguments"]) for call in calls):
                raise APIError(
                    400, "This tool history contains values this model cannot represent.", "tool_history_unsupported"
                )
            if message["role"] == "tool":
                if not functiongemma_values_supported(message["content"]):
                    raise APIError(
                        400,
                        "This tool result contains values this model cannot represent.",
                        "tool_history_unsupported",
                    )
                message["content"] = {
                    "name": call_names[message["tool_call_id"]],
                    "response": {"value": message["content"]},
                }
            continue
        if parser == "mistral-json-v1":
            if calls:
                message["tool_calls"] = [{**call, "id": identifiers[call["id"]]} for call in calls]
            if message["role"] == "tool":
                message["tool_call_id"] = identifiers[message["tool_call_id"]]
                message["content"] = json.dumps(message["content"], ensure_ascii=False, allow_nan=False)
        elif parser in {
            "granite33-json-v1",
            "granite30-json-v1",
            "smollm2-json-v1",
            "smollm3-hermes-v1",
            "phi4-json-v1",
        }:
            if calls:
                functions = [call["function"] for call in calls]
                content = message.get("content") or ""
                if parser.startswith("granite"):
                    if content.strip():
                        raise APIError(
                            400,
                            "This model requires a dedicated tool-call assistant turn.",
                            "tool_history_unsupported",
                        )
                    content = json.dumps(functions, ensure_ascii=False, allow_nan=False)
                    if parser == "granite30-json-v1":
                        message["role"] = "assistant_tool_call"
                    else:
                        content = "<|tool_call|>" + content
                elif parser == "phi4-json-v1":
                    content += (
                        "<|tool_call|>" + json.dumps(functions, ensure_ascii=False, allow_nan=False) + "<|/tool_call|>"
                    )
                elif parser == "smollm2-json-v1":
                    content += (
                        "<tool_call>" + json.dumps(functions, ensure_ascii=False, allow_nan=False) + "</tool_call>"
                    )
                else:
                    content += "".join(
                        "<tool_call>" + json.dumps(function, ensure_ascii=False, allow_nan=False) + "</tool_call>"
                        for function in functions
                    )
                message["content"] = content
                message.pop("tool_calls", None)
            if message["role"] == "tool" and parser.startswith("granite"):
                message["role"] = "tool_response"
            if message["role"] == "tool" and parser == "phi4-json-v1":
                message["role"] = "tool_response"
    if parser == "smollm2-json-v1" and tools:
        instruction = _SMOLLM2_INSTRUCTIONS.format(tools=json.dumps(tools, ensure_ascii=False, allow_nan=False))
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += "\n\n" + instruction
        else:
            messages.insert(0, {"role": "system", "content": instruction})
    if parser == "phi4-json-v1" and tools:
        if not messages or messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": "You are a helpful assistant with some tools."})
        messages[0]["tools"] = json.dumps([tool["function"] for tool in tools], ensure_ascii=False, allow_nan=False)
    return messages
