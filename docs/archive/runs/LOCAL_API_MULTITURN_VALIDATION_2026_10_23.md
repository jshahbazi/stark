# Local API multi-turn fix validation — 2026-10-23

This run qualifies a request-parser change on top of `6c96520d81af2990b5761a62cc116742d5fc87c3`. The candidate `tools/converter/stark_convert/api_request.py` has SHA-256 `ae5e38ae1f8435794d0f36253a8e803a8a8592da4d3ec9c906bf023f49ae18d1`. The native workers and embedded Python dependencies were reused from the published Stark 0.3.0 runtime. No application was rebuilt or released, and the installed user application was preserved.

## Reproduction and cause

Cherry Studio 2.1.7 was tested with a temporary profile, a custom OpenAI provider, Chat mode, default streaming enabled, and temperature/output-limit overrides disabled. Model discovery, provider verification, and a first chat turn succeeded against the unchanged parser. The second turn returned HTTP 400 with `Unsupported message fields.`

A private diagnostic wrapper recorded only message roles, recognized field names, value types, empty-value flags, and validation status. The rejected history contained three messages: user, assistant, user. The assistant message contained exactly `role`, `content`, and `reasoning_content`. Its `content` was a nonempty string; `reasoning_content` was an empty string. No tool fields or structured reasoning objects were present. The old parser allowed only `role`, `content`, and `name`.

The candidate accepts `reasoning_content` only as a valid Unicode string on assistant messages and preserves it for the model template. Empty strings are accepted. Unsupported tool/function-call fields, structured reasoning, other reasoning field names, and malformed values remain rejected. This does not add a separate reasoning output channel or change sampling defaults.

## Candidate client result

The private bridge was restarted with the exact candidate Python source and the released runtime's generation worker and dependencies. Cherry's connection URL and ephemeral key were updated. Retrying the previously rejected follow-up in the same conversation completed successfully. The diagnostic wrapper confirmed that the same three-message shape, including empty assistant `reasoning_content`, was accepted. The GUI showed a completed second response with no HTTP/API error and no active Stop control.

This establishes the observed streaming second-turn regression and its correction for this client configuration and model fixture. Additional third-turn and fresh buffered-conversation candidate retests were not completed because the desktop locked. No third-turn request reached the diagnostic wrapper before that interruption. Broader client settings, nonempty reasoning returned by a real model, and other models remain unqualified.

## Automated checks

- Two parser cases, covering empty and nonempty assistant reasoning strings, and a synthetic-worker/template propagation case failed against the old parser and passed after the change.
- Regression cases reject reasoning on non-assistant roles, invalid Unicode, null/non-string reasoning, structured reasoning objects, alternate reasoning field names, and tool/function-call history.
- All 158 focused API, desktop, and HTTP-server checks passed.
- The complete locked converter suite passed: 4,105 passed and 88 skipped in 107.03 seconds, using `uv run --locked --group dev pytest tests -q` from `tools/converter`.
- Ruff lint and format checks passed for all 80 converter Python files. Git whitespace checks passed.
- An independent read-only review of the patch found no actionable issues.

The model-free propagation test verifies that nonempty synthetic reasoning reaches the template, preserves the loaded worker, and does not alter the application's separate chat history. It does not establish model quality or performance.

## Runtime and evidence identity

The native run used macOS 26.7.1 on arm64 hardware model `Mac17,7`, the runtime's Python 3.13.13, and the existing `qwen3-dense` fixture. The released runtime records source revision `4798f6e98239e0a2afd393f1a8ce8cde14dfa613`. The source checkout's `desktop.py` and `local_api.py` were unchanged from the reproduction base.

| Artifact | SHA-256 |
| --- | --- |
| Candidate `api_request.py` | `ae5e38ae1f8435794d0f36253a8e803a8a8592da4d3ec9c906bf023f49ae18d1` |
| Unchanged `desktop.py` | `9c9a78d9def92ef0e390241f4024c3ce56206a7075898ccb403ae3d8d4c1b968` |
| Unchanged `local_api.py` | `244db2376bd6e15e81c976b05318782f243988d31b05a70649f6cf73fb09655e` |
| Native `stark-generate` worker | `583f7e06760a1288fc5965e4c86e5e670a0314cbb876e1a6067482c409beb49a` |
| Native `stark-score` worker, present but unused | `b2a7dfa817c0b39b7aba1d55e5a8baf0dcaaba3c8d7cdfe2f0af5f8fb9949b73` |
| Fixture configuration | `dfa5763dc220e871b86a6119536f5c1c2862a5137a473eaf9d812be3cc53924e` |
| Fixture tokenizer | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| Fixture weights | `87c3344a80b7376392cc5c44d5369680b68936e6930bb8ac476ca65a136f64c7` |
| Cherry Studio 2.1.7 distribution ZIP | `dd242e453735951dbea6a7755903da4ec2d5e19945fad0214e4f8a0db8622ff7` |
| Private diagnostic bridge source | `855e041c58d0b2b6bcb258ff5a8f03514daed1d66c5ac140d1f4fe9a61e3dd8f` |
| Private controller source | `6b4dec25c083113efc55f73300ed174489bf9a9787f93dbddc057034ece99da0` |
| Baseline message-shape record | `e1c18f624027a50b0a1d3ef8abeed264932c8d3e211c96a7dcd77287c5e03b8e` |
| Qualified candidate message-shape snapshot | `8e026c1b610720d7fc0f71687c806cd2c207ab218afaca092711cf820c29d3dc` |

Native test profiles, harnesses, and diagnostic records remain private. Published evidence is limited to aggregate results and artifact hashes. Prompts, responses, token vectors, credentials, probabilities, and exception dumps were not logged by the diagnostic harness.

## Follow-up after desktop unlock

The additional candidate GUI checks interrupted at the initial snapshot were completed after the desktop was unlocked, using the same candidate parser, model fixture, runtime, and diagnostic wrapper:

| Cherry configuration | Observed result |
| --- | --- |
| Existing conversation, streaming enabled, third turn | Response completed with no HTTP/API error or active Stop control; three completed-response Regenerate controls were present. |
| New conversation, Stream output disabled, first turn | Response completed with no HTTP/API error or active Stop control. |
| Same new conversation, Stream output disabled, second turn | Response completed with no HTTP/API error or active Stop control; two completed-response Regenerate controls were present. |

The Model settings were checked again after the new conversation: Stream output, Temperature override, and Max tokens override were all disabled. Mode qualification is based on the actual client settings observed in the GUI. The diagnostic wrapper did not record the top-level `stream` flag, so these results do not independently establish its on-wire value.

The new immutable message-shape snapshot contains five accepted parser calls, with message counts of 3, 5, 1, 2, and 3 in order. The five-message history contains two prior assistant messages, each carrying an empty string `reasoning_content`; the final three-message history contains one such assistant message. One additional system/user request was accepted, without inspecting its purpose. The message-shape evidence qualifies request validation; completed responses were verified separately in the GUI.

The initial one-request snapshot and its hash above are preserved. The follow-up snapshot has SHA-256 `b6633fe2c3c5cb48caf01d9d63e5b4b6366f51d835b47da14d596a8432303da9`. The private candidate bridge was stopped after capturing this evidence. Production source and the automated-check results are unchanged. Other models and nonempty reasoning returned by a real model remain unqualified.
