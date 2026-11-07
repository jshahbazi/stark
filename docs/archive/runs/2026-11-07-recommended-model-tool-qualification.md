# Larger-model tool qualification

Date: 2026-11-07, America/New_York. Implementation source:
`542d8e9f05db5e6a34036f9e05b615ad1100cd94`, PR #95. This follow-up prioritizes
newer, larger models and actual Guided Setup candidates. Earlier small-model
and specialized-model runs retain their original outcomes and are deferred
from this qualification batch.

## Targets and recommendation boundaries

The first targets are the dense `Jundot/Qwen3.8-27B-oQ4e-mtp` checkpoint,
the native-source `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` artifact, and the
larger `Qwen/Qwen3-32B` artifact that Guided Setup can admit.

Guided Setup combines its two fixed fallback recipes with metadata-screened
catalog candidates. A bounded live metadata scan admitted 34 recipes across 14
families, with zero failed families and incomplete discovery windows. The
larger Qwen3 candidates included 8B, 14B and 32B. In a simulated 128 GiB resource
envelope, the capacity preference ranked Qwen2.5 32B variants before Qwen3-32B;
the coding capacity preference favored Qwen2.5-Coder-32B. These are planning
heuristics, with no task-quality measurement in the ranking.

Qwen3.5/3.8 hybrid architectures remain outside Guided Setup's dense-only
planning admission. The two Qwen3.8 results below qualify their Advanced/native
artifact paths. They do not add a hybrid memory planner or change recommendation
ranking. Larger Qwen2.5 recommendation candidates still need their own native
tool qualification.

## Environment and sealed inputs

Apple M5 Max, 128 GiB unified memory, macOS 26.7.2 arm64. The generation worker
comes from the successful PR #95 Apple silicon standalone build at the source
commit above, GitHub Actions run `42383934745`. CLI and bundled-app workers in
that archive have the same SHA-256:
`07c3de5e19ada5ee739bd22b62affd88f029169ba896463247e6a649a18b7292`.
The source checkout's Rust 1.98.1 native template helper SHA-256 is
`14d81380b4b0e3904ff5a0cb1048d7671ba2ba8818bf413365d47c8fcb66327a`.
The host uses locked uv/Python 3.13.13 dependencies. The installed app is not
replaced; its existing idle Qwen3-4B worker remains separate from the serial
qualification runs. These runs make no throughput or numerical-parity claim.

The dense 27B model is freshly converted from the already downloaded pinned
source with the production converter. The output is 17,078,038,512 bytes with
the original mixed affine precision and full MTP readout. The experimental
coarse-readout and longer-draft artifacts are excluded. Qwen3-32B is downloaded
at revision `9216db5781bf21249d130ec9da846c4624c16137` and converted to a
65,525,706,656-byte production BF16 artifact. Lily uses its existing
prepared native-source artifact. Every run hashes actual runtime metadata and
weight files, worker and helper before and after execution and checks file-set
membership. The API driver additionally seals the host toolchain and its own
source. Test messages, generated calls, values, API keys and token vectors stay
in memory/private pipes. Persisted reports contain aggregates and hashes.

## Native smoke results

The tracked `scripts/validate_tool_calling.py` runs greedy requests with MTP
disabled, maximum 2,048 new tokens and timeout 300 seconds.

| Exact artifact | Context | Call and arguments | Result use | Exact ordinary-chat reply | Inputs stable |
| --- | --- | --- | --- | --- | --- |
| Dense Qwen3.8-27B oQ4e MTP, revision `7dd276b77fcb27c982ebd8d8a434539345713fea` | 8,192 | Pass | Pass | Pass | Pass |
| Flash Next Lily Q4-XL, revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83` | 8,192 | Pass | Pass | Pass | Pass |
| Qwen3-32B BF16, revision `9216db5781bf21249d130ec9da846c4624c16137` | 32,768 | Pass | Pass | Fail | Pass |

The two Qwen3.8 templates select `qwen38-xml-v1`, with source/effective template
SHA-256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.
Qwen3-32B selects `qwen3-hermes-v1`, template SHA-256
`a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.
Parser selection follows the exact template.

## Production HTTP API checks

The private driver creates an isolated Desktop registry pointing to the
existing artifact, invokes the production `Desktop.run` lifecycle, and starts
the cancellable LocalAPI on an ephemeral loopback port with an in-memory key.
This exercises actual native admission, template rendering, request validation,
tool parsing and SSE emission. No model payload is copied for this setup.

The 17 checks cover required/named/auto calls in buffered and streamed modes;
none-mode buffered/streamed output and equality; actual returned call-ID history
and result use; ordinary chat; integer arguments; exact-prefix reuse; streamed
disconnect cancellation; the following request; cancellation clearing the
retained prefix; and the effective MTP setting observed on native requests.
The output budget is 512 new tokens, except short reuse/recovery replies.

| Exact artifact | Context capacity | Default MTP observed | Checks | Reuse boundary | Repeated-request reused tokens |
| --- | --- | --- | --- | --- | --- |
| Dense Qwen3.8-27B oQ4e MTP | 32,768 | 3 drafts | **17/17 passed** | 1,024 | 0, then 1,024 |
| Flash Next Lily Q4-XL | 8,192 | 2 drafts | **17/17 passed** | 4,096 | 0, then 4,096 |
| Qwen3-32B BF16 | 32,768 | Unsupported; no drafts requested | **14/17 passed** | 1,024 | 0, then 1,024 |

The reuse test renders and tokenizes a synthetic prefix until it crosses the
worker's advertised chunk boundary. Actual probe lengths were 1,304 tokens for
dense 27B, 5,144 for Lily and 1,305 for Qwen3-32B. Cached and initial normalized
replies matched.
After cancellation, the next request reused zero tokens and returned the same
normalized answer. The driver requires a server-side `request_cancelled`
result, released queue/job state and successful recovery.

These results qualify the specified bounded workflows and capacities. Larger
contexts, different quantizations, other checkpoints, and broad OpenWork task
competence require their own evidence.

### Qwen3-32B partial result

The three HTTP failures are the exact none-mode plain-text reply in buffered
and streamed modes and the exact ordinary greeting. Buffered and streamed
none-mode replies agreed. All native tool choices, result use, integer
arguments, reuse, cancellation, prefix invalidation and recovery checks passed.
The original two-of-three native smoke and 14/17 API reports remain failed
overall; the expected strings are not relaxed or reclassified as passing.

A separate aggregate-only diagnostic probes short exact replies with thinking
disabled and one greeting with thinking enabled. The disabled-thinking
greeting, ping and none-mode target replies differ only by terminal punctuation; the arithmetic
answer and completed thinking-enabled greeting match exactly. Each terminates
normally. This characterizes a bounded output-format failure on the tested
native path. It does not establish upstream numerical parity or general chat
quality. Qualification remains partial for this artifact.

## Harness corrections and preserved runs

The first dense-27B API run passed 16 checks but reported no reuse. The private
setup had not attached its LocalAPI instance to Desktop, so the host correctly
withheld a session reuse scope. Attaching that owned API fixed the harness;
the repeated 8K run passed 17/17. The final 32K run above uses the corrected
setup and worker-advertised chunk sizing.

The first Lily API run also passed 16 checks. Its reuse probe was below Lily's
4,096-token chunk. The driver now measures the tokenized prompt
against the advertised boundary. The corrected run passed 17/17. Prior failed
reports and their exact private driver source capsules are preserved. No engine
change was needed for either harness failure.

## Aggregate identities

Artifact digests below use the tracked validator's sorted-JSON file-hash
manifest encoding. The earliest private API driver used compact sorted JSON;
its differently encoded digests are not substituted for these identities.

| Artifact or private aggregate record | SHA-256 |
| --- | --- |
| Dense 27B production artifact manifest | `ea5741cfdf2def4453095a1c426205c96323c295b1120d2dfbc6a210c574edf5` |
| Lily native-source artifact manifest | `5847b70d9069d56563a4e4118a90030d13ed60ee2983fc62ab3dd7e9e0ed5d53` |
| Qwen3-32B BF16 artifact manifest | `195f55c28c9969910a6ac23d4d45d31c3104de2a8f3dd58bae39f20beafcce29` |
| Dense 27B native smoke JSON | `f7a2baf23d2d2bef0ada36644f77785d0566f1c8025d67753fc84728747717a8` |
| Lily native smoke JSON | `f4643efac9451249717fa88d967adda0319f98ca3a72d4c94c5aa8f10aa72d09` |
| Dense 27B final 32K API JSON | `2b9489c68535975ba53d478cbe4d242dca5a801b1d5d324672db61eb8a0795b5` |
| Lily final API JSON | `baaaaadafcf5ba0829144cdea079c7a43ffdc5f735e2059f9acb18da310bcc08` |
| Qwen3-32B native smoke JSON | `87e871c25475d35de463937878b619ba3b965cb16c62ee0fe3a394d9f2c0dfd0` |
| Qwen3-32B API JSON | `938f0dd0b0edf77bba4d8e2f118ebcb8c5ee3c4bd389b2f2c2c7e3c2f767fee9` |
| Qwen3-32B final five-case chat diagnostic JSON | `ebf53d97018647ca01ec6e99859f930809792df6e299ef88470e2ab18056861e` |
| Final chat diagnostic driver source | `9ac55942fde36430595bc8218deb9df027afaf11e0ea83247c510c266353e688` |
| Corrected adaptive API driver source | `3ba079036a5744b52796407266198162eefc00ce69c9fba725f69724446d9159` |

Six model-free checks of the API driver's JSON/SSE normalization and sanitized
admission-failure path passed, along with three model-free diagnostic checks.
Lint passed. Public reference pages
report these as artifact-bound behavioral checks; import assessment continues
to describe protocol compatibility separately.
