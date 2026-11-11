# 20261110-TOOL-QUALITY-01: bounded tool output and continuation replay

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | 20261110-TOOL-QUALITY-01 / 2026-11-10 / 2026-11-10 / Codex |
| Kind | Model-bound tool protocol and answer quality |
| Attempt state | Completed; bounded functional checks |
| Result and acceptance criterion | Four of five artifacts passed the native tool workflow; Qwen3 Dense passed all three tool checks but missed the ordinary-chat exact-match check. Separate arithmetic controls are recorded below. Receipts contain aggregate checks and hashes only. |
| Evidence scope | Installed worker tool parsing, tool-result handling and autonomous tool-result continuation, plus separate bounded arithmetic replay; not an OpenWork UI test |
| Promotion decision | No broad model-quality, API or OpenWork reliability claim; no parser defect demonstrated |

## Scope

Run serially on installed JevK5, Qwen3 Dense, Qwen3 4B, Qwen3.8 Flash Next mixed and Lily artifacts. The tracked `scripts/validate_tool_calling.py` smoke uses one required synthetic tool call, synthetic tool-result use, autonomous continuation with tools still enabled and no additional user message, and ordinary chat. Runs used greedy sampling, MTP disabled, an 8,192-token context and a 256-token output cap. JevK5, Dense and Qwen3 4B receipts bind validator SHA-256 `3220433e4bde58748e54f560535140b0e98277761e12b799626abb49c01eda33`; mixed and Lily bind `70489bfd6b03b73a00b963f5e18df1def225478822320e1680f4d59082625b1f`, reflecting a later validator edit with the same four checks.

The smoke is not an OpenWork execution test and does not reproduce file write/read cycles. A separate ignored harness compared the original 17×19 arithmetic task, with and without a synthetic multiply tool, under model-default and installed OpenWork static base-system arms, at greedy and temperature 0.2 with seed 0. It used the installed worker, MTP disabled, context 8,192, output cap 128, and same-scope/fresh-cache controls for the static-system greedy arm. The static base prompt is not the complete dynamically assembled OpenWork system prompt.

## Evidence and limits

- Aggregate receipts are private and ignored under `runtime/tool-quality-replay/`: `jevk5.json`, `qwen3-dense-installed.json`, `qwen3-4b.json`, `qwen38-mixed-installed.json`, and `qwen38-lily-installed.json`. Arithmetic receipts are `jevk5-openwork-17x19.json`, `qwen3-dense-openwork-17x19-rerun.json`, and `qwen3-4b-openwork-17x19.json`.
- All five artifacts passed the single-call/arguments, tool-result-use and autonomous-continuation checks. Ordinary-chat exact match passed on JevK5, Qwen3 4B, mixed and Lily; it failed on Dense. Dense's bounded check is limited to that exact synthetic chat prompt and does not show a tool parser failure.
- JevK5 returned the required integer-only plain and post-tool answer in all four arithmetic arms. Dense and Qwen3 4B receipts recorded the expected number in generated text, but did not establish an integer-only visible answer; therefore these are inconclusive format/visibility outcomes, not incorrect arithmetic. Qwen3 4B's post-tool exact-answer check passed under model-default system and failed under the static OpenWork prompt arm, while the expected integer was present in generated text. No tool-call parsing failure was observed.
- Installed artifact SHA-256 values: JevK5 `95af9333e174514b6da1415011eb401e668e33f009cd6db1488fe3950441b136`; Dense `878895d4427e5121269e82b04d4d5c454709820708bb05a050d0ecc94dcc2f68`; Qwen3 4B `34c36d33e71fd692dd074185b999819d31b98b2786c1de5b54552008f291d833`; mixed `f9e7932d958a7845fd24b2f7b3cd8fc8f6b971c32a51ce88001ea52912d85f57`; Lily `ac6320b984c1bac9760dbc10b3539c823f51d5a5871d01f53fe33825c5710cab`. All receipts recorded unchanged artifacts and the same worker SHA-256 `3cc9b3072d77bada79a23eb421196a73672aeb17734e0964a9ba282b392f9449`.
- Same-scope repeated and cache-reset hashes matched in the static-system greedy controls; the reset request reported zero reused input tokens. This limited check does not qualify general prefix-cache correctness.
- The deleted Qwen3 0.6B artifact could not be recovered directly. A separate [pinned-source recreation and actual-request replay](2026-11-10-tool-quality-qwen06-replay.md) subsequently demonstrated a tool-workflow failure; it does not claim metadata identity with the deleted artifact.
- This is a bounded one-seed functional check. It does not establish general quality, parity, timing, OpenWork completion reliability or broad tool reliability. Raw prompts, responses, tokens, probabilities, credentials and exception text were not persisted.

## Follow-up history

- Registered 2026-11-10 before execution. Native runs were performed serially; large model workers were closed between artifacts.

## Actual OpenWork request control

A temporary provider-scoped observer captured one owned JevK5 post-tool request
in process memory. It did not change the request or capture authentication
headers. A direct module observer did not load; an isolated package using the
same manifest/entry-point structure as the managed connector loaded and its
setup/hook counters were verified before the captured task.

The active and official OpenWork 0.18.65 ASAR files had the same SHA-256,
`5b1626b24dfbc2d18305addd8470ac2add043a07a8ece4c9e285eabc25cf8f27`.
The request receipt recorded:

| Aggregate | Observed |
| --- | --- |
| Message roles | One each: system, user, assistant, tool |
| System JSON bytes / SHA-256 | 30,347 / `7f87005630ce2d7f98caa21ea1c9b6d6d897cbb2daf3bdebcfe511769f6da01a` |
| Tools / JSON bytes / SHA-256 | 35 / 76,290 / `de9c5faf13ea82f1d7d69d0c40ffba24c8ad2bdbe59931b2e10e02462435c98e` |
| Explicit output limit | `max_completion_tokens: 231039` |
| Sampling and stops | No numeric temperature or seed in the receipt; no stop strings. These select Stark's greedy/seed-zero defaults. |

This supersedes the earlier hypothesis that the installed agent's static
temperature 0.2 necessarily reaches Stark. The static setting existed, but was
not forwarded in the observed request. The much larger system/tool context and
output budget are demonstrated differences from the bounded native tests;
neither is established as the cause of the earlier intermittent failure.

The observed task finished without an invalid-tool-call error and created the
exact disposable file contents. Complete read-back execution was not
independently established from its UI. After stopping the app worker, the same
captured input was replayed through the current Desktop API path with explicit
temperature zero, MTP disabled and a 256-token diagnostic output cap. It returned
one valid parsed tool call. That replay executed no tools and does not qualify
completion of the overall file task. Its first setup attempt lacked the native
template helper and is excluded; the successful replay used the installed helper.

Both temporary configuration injections were restored byte-for-byte and checked
against their original hashes. OpenWork was restarted to unload the observer,
capture processes exited and cleared their requests, and generated private
backup/socket directories were removed. Stark ended with Dense selected and
stopped; OpenWork was idle. The captured request/response bodies were not saved.

The original intermittent malformed-call and incorrect-answer observations are
still not causally isolated by these successful controls. The qualification
change fixes a test coverage gap; it is not evidence of a runtime repair.
