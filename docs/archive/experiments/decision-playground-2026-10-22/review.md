# Decision playground follow-up review

Review date: 2026-10-22 (America/Los_Angeles). Reviewed implementation: `c871686a0583ea56e04211e7d2615596e7bf859b`.

This records follow-up evidence separately from [the initial validation](validation.md). Independent reviews covered the Swift controller and views, Python readout and worker lifecycle, and distribution packaging.

## Corrections

1. The combined distribution builder passed its generation worker to the app builder but omitted the scoring worker. This could select a missing, stale or unrelated `target/release/stark-score`. The caller now passes both workers from the distribution's own `libexec` directory. Its regression test verifies both paths originate in that bundle.
2. Decision scoring assumed `enable_thinking=False` established an answer position. The shipped always-on `qwen3-thinking` template can still end inside an open reasoning block, where letter logits represent reasoning continuations. The caller now compares rendering with and without `add_generation_prompt`, inspects only the appended assistant suffix, and rejects an open reasoning block before any native request. Supported boolean thinking metadata is explicitly disabled. Unknown answer-boundary contracts return a recoverable compatibility error.

The boundary check supports templates that append a distinct assistant prefix through `add_generation_prompt`. Supported reasoning controls are boolean `thinking` or `enable_thinking` fields with an explicit false value for `no_reasoning`, and the `</think>` terminator. Other contracts require separate support. Literal reasoning tags in user text, questions or answer labels do not participate in this suffix check.

## Verification after corrections

| Check | Result |
| --- | --- |
| Full converter suite | 4,022 passed, 88 skipped |
| Focused decision/desktop suite, included above | 137 passed |
| Combined distribution tests | 18 passed |
| Menu-bar packaging tests | 21 passed |
| Swift controller event/lifecycle checks | 7 passed |
| Ruff on modified backend/tests | Passed |
| App rebuild, embedded source match and local signing | Passed |
| Rebuilt package native model round trip | 7 aggregate checks passed |
| Native view interaction | Chat → Decision → Chat → Decision verified |

The added boundary tests exercise the shipped Qwen3 thinking template, supported direct-answer rendering families, unsupported reasoning contracts, literal tags in supplied input, and recovery with the loaded worker retained. Existing local Qwen3 dense and converted Qwen3.5 templates passed the boundary check.

The native round trip used the existing converted Qwen3.5-2B Q4 model. It repeated registration, chat loading/request, Decision loading/request, Chat reloading and worker shutdown through the rebuilt app's embedded controller. Answer ordering, normalized probabilities and replaced-worker termination passed. No native inference inputs or outputs were published or retained in validation artifacts. Accuracy and calibration remain unqualified.

Native screen interaction used an isolated app that compiled the production controller and SwiftUI views directly, with an independent bundle identifier and no worker connection. At a 540 × 660 point content size, both forms rendered without clipping and the manual switch selected the expected form. This check did not exercise a loaded-model request through the screen; that path was covered separately through the packaged controller. The installed Stark app was not replaced.

## Source binding

The rebuilt app records a dirty snapshot based on `c871686a`. These hashes identify the reviewed runtime sources and package independently of this report.

| Input | SHA-256 |
| --- | --- |
| `scripts/build_distribution.py` | `31290407cde7ca92f427dab2133822bfafcca213b8763cbd72a6384f7dceccf9` |
| `tools/converter/stark_convert/decision.py` | `615bae21f5285b790b4b7c6249899933cd3468645942ad7f84b3bdb2b1f6b2b5` |
| `apps/menu-bar/Sources/StarkApp.swift` | `793fe5e8263c373231c8dbcd0898a3f891632a41a905e8f13a274109947c80b1` |
| Rebuilt `menu-bar-manifest.json` | `4e0723a3a16d0521dd5e81d18c7546b3b327f31542492f324c040504f501edc7` |
| Packaged `stark-generate` | `5d81d0a55aeda2d7b81be793396a6c23dbaca597c6efe2b2a94570b5b51ce2f9` |
| Packaged `stark-score` | `f422ccbb7983a4cf015146fb12c5dfaccf3a358b6ed271f011dd5ce6a582ecfa` |

The rebuilt app is available locally under ignored `artifacts/menu-bar/Stark.app`. The original validation results and their source/binary bindings remain unchanged.
