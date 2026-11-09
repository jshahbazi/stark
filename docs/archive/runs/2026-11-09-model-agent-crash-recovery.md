# Model-family agent crash recovery, 2026-11-09

This is the coordinating checkpoint for the interrupted default-image work.
The machine reboot ended the original agent sessions. The three implementation
worktrees, committed history, unfinished source edits and private validation
receipts survived. Replacement agents resumed those worktrees; no model weights
or previous evidence were deleted. The reboot cause has not been established.

The governing acceptance criteria remain in
[the default-image parity plan](../../IMAGE_DEFAULT_PARITY_PLAN.md): image
support is the default; preserve exact text caching, matched normal text
performance and supported text MTP. Image requests use zero drafts and invalidate
text state. Completion is per artifact, configuration and backend.

## Ownership and source checkpoints

| Owner | Branch and recovered HEAD | Worktree | Next gate |
| --- | --- | --- | --- |
| `qwen_recovery` | `codex/qwen-image-parity-next`, `48c5aa52beacdba117beae7b5b7bc71675f0f702` plus preserved edits | `stark-qwen-image-parity-next` | Fix inactive-head MTP0 dispatch; independently review, then qualify current source cache and speed |
| `gemma_recovery` | `codex/gemma-image-parity-next`, `dccb8ed0cd1e80f3cbe84746ab57a427aad0f8cf` plus preserved edits | `stark-gemma-image-parity-next` | Build the saved assistant arithmetic fix, run controls and repeat the unchanged E2B numerical oracle |
| `glm_recovery` | recovered `fecc27d551773d8197f0f9b2c55c6dba21fb9444`; recovery commits pushed through `31fd48dbd83e28e26a470abecada719362e79dc4` | `stark-glm-image-parity` | Finish peak-memory accounting before any further full-model execution |
| Coordinator | `codex/image-default-model-worklist` | `stark-image-default-model-worklist` | Maintain this checkpoint, serialize resource-heavy validation and integrate independent review |

Worktrees are siblings under `/Users/johnshahbazian/development/`.
The Qwen27 branch `codex/qwen-dense-image-parity` remains frozen at
`67284748f82f71bd60f3a73f57c5fb31120db505` for
[PR #114](https://github.com/jshahbazi/stark/pull/114). Follow-up source fixes
belong on the branches above. No additional PR was opened during recovery.

## Preserved recovery snapshots

Each implementation worktree contains the ignored private directory
`runtime/crash-recovery-unified-20261109T174122Z/`: complete current source-file
copies, status, normal binary-capable unified patches and per-file hashes.
Every working-tree patch passed `git apply --check --reverse` against its saved
source without modifying it. These are source snapshots, not persisted inference
or prefix-cache state.

| Snapshot | Working-tree patch SHA-256 | Bytes / complete files |
| --- | --- | --- |
| Qwen | `a3fbdd845ce7c44cc7a32bad834852244f88f195a04f6ab6cba2cc9da89ff46c` | 19,109 / 4 |
| Gemma | `df5a7f9868bfee7bf8f1fc103907ec6065faa902af6c1f93d5464e9c81049cca` | 40,259 / 9 |
| GLM | `78f2533bfbc7f76f537829d960fc0c012b07f6ba0f13eb41d1d150ae22eb550e` | 3,009 / 4 |

The earlier `runtime/crash-recovery-20261109T173249Z/` diff used an external
display formatter and is not an applyable recovery patch. It is retained for
history; use the verified unified snapshots above. Agents have continued editing
after these snapshots; newer candidate source must receive its own source seal.

## Evidence retained and remaining work

### Qwen

The completed headless paired benchmarks survive: official BF16 0.8B
303.618/304.108 tokens/s, affine 0.8B 659.888/661.392 and affine MoE 35B
39.988/39.960, respectively text-only/image-loaded. These short measurements
qualify their original binaries, artifacts and output budgets only.

Two completed pre-reboot lifecycle receipts were recovered and verified:

| Private receipt | Result | SHA-256 |
| --- | --- | --- |
| `runtime/bf16-08b-cache-04.json` | 30/30 | `5943494efafb473421fd0e0ecfd9a761679418736d36399f5de903221982626a` |
| `runtime/affine4-08b-cache-04.json` | 30/30 | `6a15054bc82a09d35847f16201f24c80a85ee320d9e9981c1d098d3aa1af445b` |

The matrices include real-image transitions and distinguish a 1,025-token
repeat miss from actual 1,024-token reuse on the 1,026-token case. They remain
bound to the original `4010b86e` worker; they do not qualify later source edits.

Post-reboot, the actual BF16-headed 0.8B artifact passed 26/26 bounded lifecycle
controls on the recovered `c69a6685` worker, including MTP3/0 switches, seeded
cache equality, scope/context/reset/error/cancellation, forbidden image MTP and
real image followed by fresh/repeated text MTP. Receipt
`runtime/bf16-08b-head-cache-recovery-01.json` SHA-256:
`1eb4e6a8eb50a13beecb31d14090e64cd7ba801376120af55dedde78384734a7`.

The headed short benchmark exposed an additional problem: MTP0 still entered
the predictor session, approximately halving normal BF16 0.8B throughput versus
the previous headless target. MTP3 text/image-loaded medians were
147.764/150.135 tokens/s; MTP0 was 157.330/149.369. The latter is a 5.06%
loading-mode loss and fails a strict 5% gate. This is an open regression.
The candidate dispatch fix selects the ordinary stream at MTP0 while retaining
head weights and exact target-only boundary snapshots; independent source
review and fresh binary-bound cache/performance checks are still required.

Also open: complete corrected affine MoE lifecycle matrix, independent BF16
head numerical reference, full BF16 MoE/expert-head qualification, other
sizes/quantizations, and the distinct 107.30 GB retained ddalcu artifact.
The ddalcu full-model load remains resource-deferred.
The detailed branch record is
[`2026-11-09-qwen-remaining-image-text-parity.md`](https://github.com/jshahbazi/stark/blob/codex/qwen-image-parity-next/docs/archive/runs/2026-11-09-qwen-remaining-image-text-parity.md).

### Gemma

Gemma 3 4B's 40-case lifecycle run and candidate text/image-loaded medians
68.10/67.04 tokens/s survive. The separate prior-worker baseline attempt had
swap/background activity and remains ineligible. Previous Gemma 4 E2B/E4B cache
evidence remains bound to `7468dbc4`; the earlier `81b85f88` qualification was
withdrawn because it replayed the full prompt.

The official E2B assistant converted and executed native proposals, but its
pinned Transformers oracle failed: masked-logit maximum absolute error
47.28125, relative L2 0.02197. Matching the first greedy proposal does not pass
that gate. Serving MTP remains disabled.

The saved G4AP-02 fix changes assistant-only BF16 query arithmetic and committed
KV handling. Recovery preserved it, added partial-rejection/zero-draft controls
and a conservative draft-position guard above 512, and resumed a distinct build.
An initial test compile failure from an unavailable tempfile helper was corrected
using the existing private test-directory helper; the failed attempt is retained.
Build success, CPU/Metal controls, unchanged E2B oracle, target verification,
cache integration and useful MTP performance remain separate open gates.
E4B and larger assistants require their own qualification.
See the
[Gemma branch ledger](https://github.com/jshahbazi/stark/blob/codex/gemma-image-parity-next/docs/model-experiments/2026-11-09-gemma-next-image-parity.md).

### GLM

The saved full GGUF worker loaded image-ready in 196.30 seconds and completed
an ordinary text control, but actual image generation failed
`resource_budget_exceeded`. A later text request also failed after lazy expert
bank growth. Vision component tests do not establish full image generation.

Recovery preserved that failure and the rejected admission relaxation in commit
`f68697e7`. Commit `31fd48db` isolates the macOS/aarch64 SHA-256 feature change.
Unchanged source identities, worker/helper seals and GGUF hardlink identities
were verified. Five complete metadata/sidecar hashes agreed across hardware,
software and independent references; this does not qualify decoder startup speed.

Header/source accounting finds 99,035,873,280 bytes of rounded expert-bank spans.
The existing resident-bank/static/state admission subtotal at context 4,096 is
approximately 108,692,065,600 bytes, before separately loaded vision, host
workspace and 305,266,688 bytes for two overlapping KDA snapshots. Implementing
the currently dormant predictor adds approximately 2.788 GB of weights.
These estimates do not prove residency or peak RSS. A full-resident retry is held;
bounded execution needs separate review and performance qualification.
Actual image execution, cache/lifecycle, matched text performance and GGUF
source-head MTP remain unfinished. EXL3 has separate backend-specific gates.
See the
[GLM admission record](https://github.com/jshahbazi/stark/blob/codex/glm-image-parity/docs/archive/runs/2026-11-09-glm-image-admission.md).

## Restart and resource rules

1. Preserve original receipts and failed attempts. Never relabel an old binary's
   result as evidence for new source. Seal new candidate source and binaries.
2. Resume unfinished gates; repeat completed checks only when a source change
   affects them or their original measurement was invalid.
3. Serialize Metal/model runs and coordinate memory-heavy builds. Use a distinct
   ignored target directory per candidate; start with small artifacts.
4. During recovery an installed Stark worker outside agent ownership occupied
   approximately 71 GiB RSS. Agents paused model/Metal launches and preserved that
   process. Resolve resource ownership before issuing another hardware benchmark.
5. Keep GLM and ddalcu full-model loads held until complete peak accounting is
   reviewed. Do not weaken admission based on a single free-memory observation.
6. Update family ledgers with source identity, aggregate results, remaining gate
   and the next command/recipe before handing off. Keep this coordinator record
   linked from the plan; store private native receipts under ignored runtime paths.
