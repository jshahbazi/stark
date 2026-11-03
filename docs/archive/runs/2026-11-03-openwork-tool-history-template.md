# OpenWork follow-up tool-history rendering fix

Date: 2026-11-03 (America/New_York).
Code: `9703eff240819f3766bb839199dadd3eee9b92fe`, based on `b4fcc7e6`.

## Failure and cause

The user reported that OpenWork completed a tool-using reply, then rejected the
next prompt with Stark's `400 invalid_messages` template error. The running
model was `qwen38-flash-next-lily-q4-xl-0f94842e`, revision
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`. Its public template SHA-256 was
`c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.

Synthetic history reproduced the failure using that template and the native
helper. Plain two-turn history passed. History containing an assistant tool
call, a tool result, a completed answer and a new user message failed; the
Python Jinja test reference rendered the same synthetic history successfully.
No request body from the user's conversation was captured or replayed.

Stark disables MiniJinja's default features and had omitted
`adjacent_loop_items`. That removed `loop.previtem` and `loop.nextitem`, which
the Qwen3.5 and Lily templates use to group tool results into a user turn. A
final tool result could render with a missing opening turn delimiter. An
earlier tool result followed by another message attempted to read the missing
next item's role and failed evaluation.

The fix enables that feature on the existing pinned MiniJinja 2.24.0 dependency.
Native Rust rendering, dependency versions, sandbox restrictions, resource
limits and content-free error handling remain in place. Python Jinja is used
as a test reference; the application has no Python rendering fallback.

## Validation

- Before the fix, the new regression matrix had five failures and two passes.
  It detected both malformed tool-result boundaries and follow-up rejection.
- After rebuilding the helper with Rust 1.98.1, all seven new cases passed.
  They cover empty/single/multiple adjacent loop items, single/parallel tool
  results, a final tool result and a completed answer followed by a new turn.
- The existing public Qwen3.5 template fixture supplies tool-history parity
  coverage without adding a model artifact or private fixture.
- Eight synthetic cases using the installed Lily template matched the Jinja
  reference, covering single/parallel tools, tool results/follow-ups and both
  thinking settings. Calls were normalized through `template_messages()`.
- Full locked offline converter suite: **5,953 passed, 89 skipped** in 257.41 s.
- Native helper tests: **16 passed**. Clippy with warnings denied, pinned
  nightly Rust formatting, Ruff checks/formatting, repository policy and
  whitespace checks passed.
- Independent review found no actionable issue. The MiniJinja feature retains
  only a bounded number of adjacent values per loop, under existing limits.

The release helper build completed with a local `rust-objcopy` debug-stripping
warning about a missing `libLLVM.dylib`. Native protocol tests and the packaged
helper checks passed.

## Installed build

Signed development build `20261103201254829096` was installed to
`~/Applications/Stark.app` and launched. The app manifest records code commit
`9703eff2`; its dirty flag reflects the pre-existing untracked research note,
with no tracked diff. The existing generation and scoring workers were reused;
the template helper was rebuilt. The app uses Developer ID Team `S5BT2PRN5B`,
hardened runtime and a secure timestamp. This local build is not notarized.
Lily was reloaded through Manage Models; the app displayed **Ready** with the
same model ID and the new build number.

The signed staged app passed deep strict signature verification. Its embedded
Python, invoked with `-I -B`, rendered a synthetic Lily tool-history follow-up
through the packaged native helper and matched the reference. The signed
helper SHA-256 is
`73fe9b462b8d542b98ed0d91158323d1692e0362c76d8df17480588b04ffa704`.
Independent final verification confirmed that the installed executable and
helper were byte-identical to the staged files, deep strict signatures passed,
and the installed private Python/helper successfully rendered the same
synthetic Lily tool-history follow-up.

The user's failing OpenWork conversation was not replayed. Native inference
performance and model quality were not remeasured. Only synthetic rendering
data and aggregate results were used; no user prompt, response, tool argument,
token vector or credential was saved.
