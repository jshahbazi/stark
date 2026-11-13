# Context and resource-budget findings

Date: 2026-11-08

## Outcome and scope

This report covers the complete **Audit inference size limits** conversation: the initial audit, shared-budget implementation, automatic execution for undeclared contexts, PR preparation, and the subsequent correctness review. It consolidates findings; it does not replace the source-bound run records linked below or backfill unrelated model optimization history.

The implementation is in [PR #99](https://github.com/jshahbazi/stark/pull/99), with final tested runtime revision `9a3e72161332af6a27287c471c86a6491a7d89f5`. At the original report's status check, that revision had 19 successful CI checks and the PR was open. PR #99 subsequently merged on 2026-11-08 as `ea098a244e25dfe5be934809c1fb87344183ac0c`. The installed app was not updated by this work.

The work removed shared policy limits that could reject requests before tokenization, added automatic output allowances and expanded memory admission and recovery. It established bounded compatibility results on specific Qwen artifacts and a synthetic recurrent fixture. **It established no measured speedup.** No inference kernels, weight precision or numerical execution policy were intentionally changed.

Future attempts are indexed in the [model experiment ledger](../../model-experiments/README.md). IDs in this report use the `20261108-CTX-` prefix.

## Requirements and decisions

| Decision | Final requirement |
| --- | --- |
| Small independent byte caps | Consolidate request, tool-result, rendered-prompt, transport and template allocation allowances into a shared resource policy. |
| User configuration | Do not add an app resource-budget setting. Select admission automatically; retain a developer/operator override. |
| Omitted output limit | Use the remaining admitted context, with ordinary model stop conditions, cancellation and deadlines. Explicit output limits remain supported. |
| Undeclared context capacity | Plan a finite execution allowance from loaded state geometry and available memory. Continue reporting model capacity as undeclared. |
| Stability | Reject unsafe requests recoverably, clear affected caches/prepared state, and permit a smaller retry on the same healthy worker. |
| Template execution | Preserve source, fuel, recursion, CPU/wall-time, sandbox and structural protections. |

The first implementation still required explicit output limits for undeclared contexts. That was an intermediate restriction; the user's follow-up required automatic planning for those models too. The proposed user-facing budget control was also dropped after the user clarified the desired interface.

## What the audit confirmed

The old 1 MiB HTTP/request/rendered-prompt limits, 256 KiB per-tool-result limit, 2 MiB template-helper request limit and 4,096-token output ceiling were Stark policies shared by the relevant entry points. They were not requirements of GLM or any other particular model. A byte cap cannot stand in for a token context limit because serialization, escaping and tokenization vary.

The final policy, described in [worker protocols](../../WORKER_PROTOCOLS.md#resource-budget), selects a logical budget `B` of 16–64 MiB from available RAM at startup and pins it for child workers/helpers. Encoded framing allows `6B + 2 MiB`. Tool results use aggregate message admission instead of a separate 256 KiB cap. Actual tokenized input plus requested output must fit the model/request context. Token-ID and generated-content storage remain resource bounded.

This does not eliminate every limit. Structural limits, image bounds, device buffer/index limits, resource admission and execution deadlines remain. The HTTP completion deadline is still 300 seconds, including queueing and execution after parsing. A legacy worker that omits modern framing metadata retains the 1 MiB compatibility fallback. Image preparation retains its context binding; image Auto may reject a window that cannot fit instead of shrinking that prepared transaction.

## Findings and attempt log

“Review” means a source/lifetime/geometry finding, not an observed model crash. “Model-free” includes synthetic backends and private-pipe protocol tests. Adoption in this table means implementation in the PR, not deployment.

| ID | Attempt or finding | What failed or was insufficient | Final disposition and evidence |
| --- | --- | --- | --- |
| CTX-01 | Audit shared byte/token caps | Requests that could fit a token window could encounter an earlier independent cap. Changing one layer would leave others. | Confirmed by source audit; coordinated change implemented. [S1](#source-and-run-records) |
| CTX-02 | User-configured fixed 16 MiB budget | Did not match the user's clarified preference; this was a design rejection, not a failed model run. | Replaced with automatic 16–64 MiB selection and a developer/operator override. [Policy](../../WORKER_PROTOCOLS.md#resource-budget) |
| CTX-03 | Shared budget and template admission | Transport escaping, parsed values and template execution need more RAM than logical text alone. Old tests assumed the removed 1 MiB boundary. | Derived framing/allocation allowances; template protections retained. Large-text, hostile expansion, cancellation, fuel and recovery controls passed. Stale expectations were updated. S1 |
| CTX-04 | Finite-context Auto and removal of the output ceiling | A fixed 512-token API default and 4,096-token worker ceiling prevented the requested behavior. | Native Qwen3.5 emitted 4,097 sampled tokens. HTTP Auto selected remaining context and completed short requests. This was not a throughput study. S1, S3 |
| CTX-05 | Manual output requirement for undeclared contexts | Initial safe fallback required a user limit and did not satisfy the follow-up request. | Superseded by native `plan_generation`; old workers without planning return recoverable update guidance for undeclared-context Auto. S2 |
| CTX-06 | Undeclared-context state planning, including Dense MTP | No declared maximum does not imply constant-size state. Attention, scratch and predictor state still need accounting. | Allocation inventories added, including target/predictor state and snapshots. Synthetic ShortConv HTTP/CLI worked. No trained undeclared-context or undeclared-context Dense MTP artifact was qualified. S2 |
| CTX-07 | macOS available-memory probe | The native smoke exposed a valid older kernel statistics prefix being rejected against a newer libc structure definition. | Accept only the initialized prefix needed by the probe, bound its returned count, and retain the zeroed tail. Native automatic execution then passed. S2 |
| CTX-08 | Integration onto current `main` | The original branch included an unrelated assessment; current main had newer vision variants, normalization and idle-maintenance code. Incomplete vision matches required correction. | Clean PR branch; unrelated assessment/website edits excluded; main's runtime paths preserved. Commit `fc7e83fa` and integration checks. |
| CTX-09 | Finite-context cache growth | Initial planning covered undeclared contexts. Removing the output cap also exposed longer lazy cache growth on finite generic, GLM and Qwen paths. This was found by review, not an induced OOM. | Finite text Auto uses advertised planning; explicit text requests get allocation admission. Family inventories added for GLM, Qwen, Flint and retained Lily. EXL3 retains its existing device phase bill. S3 |
| CTX-10 | HTTP parsing memory and concurrent readers | The first three-times-wire-size estimate did not cover mixed-Unicode parsing expansion; independent readers could over-admit together. | Shared twelve-times-raw parsing estimate for large frames, atomic HTTP reservations and release after parsing. These are estimates, not RSS limits. S3, S4 |
| CTX-11 | Hybrid resource error classification | Inventory/index failure could become `context_planning_unavailable` during generation and lead the host to close the worker. | Resource failures use `resource_budget_exceeded`; model-free retry coverage verifies terminal-frame consumption and worker reuse. S3 |
| CTX-12 | Device buffers and shader indices | Aggregate RAM fit does not prove that an individual buffer or its 32-bit element offsets are valid. | Checked products, per-buffer limits and index bounds added to relevant profiles. Geometry tests passed; no extreme-capacity hardware run was claimed. S3, S5 |
| CTX-13 | Scoring admission | The enlarged scorer input budget also needed native admission. Generation's one-pool scratch estimate missed scoring's independently retained chunk pools. | State charged once, every retained pool accounted for, and input may exactly fill context without an output slot. Dense and hybrid one-token scoring/retry smokes passed. S4 |
| CTX-14 | Scoring estimate refinement | Charging generation sampler/readback per chunk and doubling every chunk's inventory would unnecessarily restrict long scoring inputs. | Final vocabulary/readback charge occurs once; an extra scratch inventory covers final partial/split geometry. Conservative assumptions remain. Arithmetic and lifecycle tests passed. S4 |
| CTX-15 | Private app command parsing | Enlarging private-pipe framing left a parsing-memory gap before native inference admission. | Check while receiving and before decoding; drain a rejected frame through newline. Tests preserve the loaded worker and allow Reset/Stop and smaller retries. S4 |
| CTX-16 | Image decoder growth | Existing image qualification estimates a short 2,048-token decoder check, not the user's full requested output span. | Native Qwen/Gemma and retained hybrid image paths admit expanded decoder spans before state creation/prefill. Buffered/streaming discard-and-retry tests passed. Coverage is model-free. S5 |
| CTX-17 | Full sparse virtual reservations | A short active span can fit RAM while its selected full virtual cache exceeds a device or index limit. | Separate full/ring virtual-cache checks propagated through decoder and Dense MTP metadata, including predictor context bounds. No context-binding or numerical execution change. S5 |
| CTX-18 | Performance cost of bounded callbacks | Buffered generation now checks output through text/progress callbacks. Added decoding overhead is possible. | No matched timing comparison was run. Keep as an open performance question; do not label this work a speedup. S1 |
| CTX-19 | Early tokenizer provenance | S1 records a 63-character hexadecimal tokenizer digest, which is not a valid SHA-256 representation. Later records contain a different, valid 64-character digest. | **Unresolved.** Preserve S1. Recover its original receipt or mark that artifact identity incomplete; do not silently substitute the later digest. |

The intermediate full-suite failures described in the conversation were stale expectations for old byte limits and the earlier undeclared-context output policy. Subsequent focused/full runs passed after the policy and tests were reconciled. Those failures do not establish a failure of GLM, Qwen or another model's numerical execution. Conversely, an initial green CI run did not detect the missing finite/scoring/image admission paths; the independent review was necessary.

## Results by model and artifact

| Artifact or path | Execution evidence from this conversation | What it establishes | What remains unverified |
| --- | --- | --- | --- |
| Early local Qwen3.5 2B Q4 | Native 4,097 sampled-token completion with `length`; 1,155,080-byte HTTP body reached native tokenization (330,011 input tokens), returned context overflow at API context 4,096, then smaller retry returned 200 on the same PID. Auto used 18 input plus 4,078 output slots and emitted one completion token. S1 | Removal of the old ceiling, post-tokenization context enforcement, finite Auto and recovery for that run. | Early tokenizer identity is incomplete (CTX-19); weights hash was not recorded there. No long-generation quality, filled window or speed comparison. |
| Later retained Qwen3.5 2B Q4 | Two Auto requests each admitted context 262,144: 17 input plus 262,127 output slots; each emitted one token. Hybrid scorer follow-ups completed two one-token exact-fill requests per run without truncation. S3–S5 | Actual native generation admission, early stop, scoring exact-fill behavior and repeated-request health for the bound later artifact. | The 262,144-token allowance was not filled. No full-window stress, numerical parity, tool-behavior quality or performance gain was established. |
| Dense Qwen3 converted GGUF Q8 | Two exact-fill one-token scores per run, context one, no truncation, same worker. S4–S5 | Fresh-state scoring admission allows input to equal context. | No long-prompt score, parity or speed comparison. The conversation also logged a finite-worker planning/bounded-generation smoke with 8,192 total context and 8,191 output slots, but did not seal that auxiliary observation's binary identity in a dated record. |
| Synthetic ShortConv fixture | One recurrent layer, four-dimensional vocabulary/embeddings, six tensors and 288 payload bytes; no trained weights and no declared context. HTTP/CLI Auto worked; repeated HTTP returned 200 in the same worker/PID. Six input tokens received 16,777,216 output slots; EOS produced zero completion tokens. S2–S5 | Genuine undeclared capacity, automatic allowance selection, initial state creation, EOS and recovery without manual options. | The allowance was not filled. This does not qualify a trained model, long-run memory, quality, Dense MTP or throughput. |
| GLM-5.3 GGUF | Source review and profile tests for rounded cache growth, host/device inventory, transient replacements, score planes and device/index limits. S3–S5 | Implemented admission coverage for this backend. | No GLM model execution or full-window/performance qualification was performed in this conversation. |
| Optional GLM EXL3 | Source review reused the bridge's existing whole-context device phase bill and added host workspace admission. S3 | An existing admission contract was identified and preserved. | No new EXL3 artifact run, speed result or live memory qualification here. |
| Native Qwen and retained Lily/Flash-Next profiles | Geometry/KV-format, scratch, recurrent/PLE, MTP/snapshot and growth tests; later Qwen3.5 is the native smoke named above. S3–S5 | Backend implementation and bounded tests. | A Qwen3.5 smoke does not qualify Qwen3.8-Flash-Next or every Lily checkpoint/quantization. No Flash-Next speed study was run here. |
| Flint | Cache/window and workspace inventory tests. S3 | Code/model-free admission coverage. | No Flint artifact execution, quality or speed result here. |
| Generic attention/recurrent and Dense MTP paths | Checked geometry, pool, overflow, predictor and virtual-cache tests; actual artifacts are listed separately above. S2–S5 | Shared implementation and bounded invariants. | No blanket qualification of Llama, OLMo, Gemma text, Ripple, all quantizations, or a trained unknown-context Dense MTP artifact. |
| Native Qwen/Gemma3/Gemma4 and retained hybrid images | Model-free allocation/protocol tests, including expanded-token admission, prepared-state cleanup, buffered/streaming rejection and retry. S5 | Decoder admission and transaction recovery coverage. | No eligible converted image artifact at or below 15 GiB was found in the bounded local inventory. No new real-image hardware run, full-window image test or complete encoder peak estimate. |

### Artifact identity boundaries

The later Qwen3.5 artifact is bound by config SHA-256 `ca579bbcbd1b3e13bbedbcd3c17c2266ebdc8c357a07e8c7706580ddfb327c8e`, weights `a9e01fbf845a1d2af4727f4394b244815abd26b25d186bc49730249fb16c840d`, and tokenizer `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4`. S1's malformed tokenizer digest cannot establish that the early and later artifact tuples are identical.

The dense Qwen3 artifact uses the three-file manifest `2f717d2234906f32104c25c5f438ddf61c3d4c5605b6bd13f24c84e53b736499`. The synthetic fixture's config, weights, tokenizer and generator hashes are in S2. Source/model/worker identities must travel together; a family name or local directory name is insufficient.

## Source and run records

These are different validation snapshots, not interchangeable results. A record's “baseline” can precede its working patch; use its runtime-source manifest and executable hash as well as its stated commit. A record with several executables can describe several snapshots: do not assign its final source manifest to every earlier executable without a receipt that pairs them. The first manifests cover 849/850 files; later 1,075-file manifests also include vendored Lily. Do not compare them as identical manifest scopes.

| ID | Preserved record | Source binding and role |
| --- | --- | --- |
| S1 | [Initial resource-budget validation](2026-11-08-context-resource-budget.md) | Baseline `78090aae`, working patch `f2b1c886…`, runtime manifest `01743bac…`; early native/HTTP tests and template controls. Contains CTX-19's unresolved digest. |
| S2 | [Undeclared-context Auto](2026-11-08-unknown-context-auto.md) | Baseline `2acae950`, manifest `5041d316…`; synthetic native HTTP/CLI, composite state planning and macOS probe correction. |
| S3 | [PR correctness review](2026-11-08-context-budget-pr-review.md) | Baseline `fc7e83fa`, manifest `b68b4b66…`; finite/hybrid admission and HTTP parsing correction. |
| S4 | [Scoring and app input](2026-11-08-scoring-and-app-input-budget.md) | Baseline `cba64591`, manifest `14c0c350…`; scorer pool/readback accounting, exact-fill smokes and private-pipe parsing. |
| S5 | [Image decoder follow-up](2026-11-08-image-decoder-budget.md) | Baseline `a2670011`, manifest `52b42de0…`; image decoder and selected virtual-cache guards. |

Original feature commits were `2acae950` and `653fcefe` on `codex/context-resource-budget`. The clean PR replay used `9e4c8bc6` and `2815c7b5`, followed by integration fix `fc7e83fa`. Review corrections were committed as `cba64591`, `a2670011` and `9a3e7216`. Different hashes after replay do not make their validation receipts interchangeable.

The final reported checks at runtime revision `9a3e7216` were: 6,701 Python tests passed with 96 skipped; 555 CPU/grammar engine tests passed with 30 ignored; seven C++ string tests, two profile tests and three doctests passed; 133 generation tests passed with two ignored; five shared-budget tests passed; default and optimized scorer suites passed 65 tests each; 45 focused image/undeclared-context tests passed. Swift controller checks passed 134 cases earlier in the review. Counts from overlapping runs must not be summed or treated as model qualification. Release builds, Clippy, formatting, policy and all 19 CI checks passed for the named revisions/scopes.

## Performance interpretation and remaining work

There was no matched baseline/candidate timing study in this conversation. Token allowances, completion counts, test counts and CI success do not establish tok/s, first-token latency, peak RSS, quality or a speedup. The buffered callback change is a specific unmeasured overhead question. Existing optimization histories remain indexed separately in [benchmarks](../../benchmarks/README.md).

Follow-up work should be logged before execution:

1. Recover the original S1 tokenizer receipt; preserve the unresolved state if it cannot be recovered.
2. Measure tokenization, template rendering, planning, first-token latency, useful decode rate and observed peak memory against matching builds, including buffered callback overhead. Keep cold/warm and prefix-reuse conditions separate.
3. Qualify long input/output and filled allowances on each exact trained artifact. Add a genuine undeclared-context trained artifact and Dense MTP case when available.
4. Run real-image hardware validation; retain separate decoder and encoder memory accounting. Prepared-context resizing and a complete encoder peak inventory remain unimplemented here.
5. Measure conservative admission under memory pressure, including retained-state double charging. A current-memory observation is not a reservation and other processes can change it.
6. Track merge, release and installation separately. Runtime PR #99 is merged; release and installation require their own evidence.

Only aggregate results, hashes and reproducible conditions belong in public records. Prompts, completions, token vectors, probabilities, credentials, exception dumps, models and private runtime artifacts remain excluded.
