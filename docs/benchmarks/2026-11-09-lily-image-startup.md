# Retained Lily image startup and Playground attachment

Registered on 2026-11-09. Owner: Codex. This record preserves the canceled
installed-build startup and the subsequent full-hash acceleration attempt.
The initial diagnosis is retrospective; candidate model execution was not run
before registering this record. See the [attempt ledger](../model-experiments/README.md).

## Identity and acceptance

| Field | Value |
| --- | --- |
| Baseline attempt | `20261109-LILY-STARTUP-01`: installed 0.18.9, exact source commit unavailable |
| Candidate attempt | `20261109-LILY-STARTUP-02`: source `b0c0d9fb` plus this branch's retained-image hashing patch |
| Follow-up attempt | `20261109-LILY-STARTUP-03`: reuse validated file digests and remove image-check UI messages |
| Kind / scope | Artifact compatibility, component correctness, startup and Playground lifecycle |
| State / result | Attempt 02 passed image attachment and shape recognition; its startup latency was rejected. Attempt 03 installed; two native vision startups passed, full UI timing pending |
| Acceptance | Reuse existing shards without a download, preserve every hash/admission check, load successfully, enable attachment and process a synthetic image in the Playground |
| Decision / disposition | Attempt 02 superseded for latency; attempt 03 awaiting model-bound validation, adopted on branch and installed locally, not merged or released |

## Hypothesis and change

The installed retained-image loader hashed 106.9 GB sequentially using the
software SHA-256 backend. The user canceled startup while the worker was active
at roughly one CPU core and before weight allocation. This is a canceled load,
not a measured load failure or confirmed timeout. An earlier diagnostic guess
of two full hashing passes was incorrect: the retained loader hashes each shard
once and subsequently rechecks filesystem signatures.

The candidate enables the locked `sha2` AArch64 assembly backend and hashes up to
four shards concurrently. Manifest order, per-file and aggregate digests,
no-follow/regular-file checks, size limits and pre/post-load filesystem
signatures remain enforced. Hash buffers are bounded to 32 MiB total across
four workers. Attempt 02 did not introduce a cache and was rejected after an
observed Ready upper bound of 31.753 seconds; the user reported a previous
startup of approximately eight seconds. That historical figure was not timed
by this run.

Attempt 03 retains full hashes during preparation and reuses them on subsequent
loads only when the manifest and each open descriptor's device, inode, length,
mtime and ctime (nanoseconds included) agree. The cache is bounded, regular,
non-symlinked, owned by the effective user and private (0600). Unrecognized,
stale or missing entries trigger a full hash, which still must match the
manifest. Load-time change checks remain. This is local filesystem-based digest
reuse, not a cryptographic guarantee against privileged metadata manipulation.
Preparation seeds the same format after actually hashing the existing payloads,
so the first worker need not repeat that scan. No inference state is persisted.

The Playground and connection-details verification notes are removed; internal
image capability/evidence fields remain. Installation image-check events use
the ordinary loading message rather than a separate image-check status.

## Artifact, source and environment

| Field | Recorded identity |
| --- | --- |
| Checkpoint | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` |
| Upstream revision | `0f94842e3682b49acace4daeb4cf2e65a6f94f83` |
| Payload | 55 retained mixed Q4/Q8/BF16 Safetensors shards; 106,905,155,192 bytes |
| Artifact profile | Schema 6, `qwen38-lily-retained-bf16-vision-v1`, version 1 |
| Artifact aggregate SHA-256 | `18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4` |
| Image manifest SHA-256 | `cd1e467d66062bdeeea841123ccdef48a84a3a60f468f3a88004e129c9768092` |
| Toolchains | Rust 1.98.1 release generator; nightly-2026-10-01 engine tests; locked uv Python 3.13.13 packaging |
| Candidate dependency | `sha2 = 0.10.9`, `asm-aarch64`; `sha2-asm = 0.6.4` pinned in Cargo.lock |
| Precision and inference state | Unchanged retained Lily decoder/tower precision, MTP rules and prefix-reuse policy; hashing ends before model construction |
| OS/hardware | MacBook Pro Mac17,7; Apple M5 Max, 18 cores; 128 GB; macOS 26.7.2 (25G332) |
| Baseline signed worker SHA-256 | `88b859168f87edac781e86fcc0e94126d373159baf6efb8abd352f91fc16aca2` |
| Attempt 02 unsigned/signed worker SHA-256 | `708494d6584b5ce3e7b3372827ba02c16a091d9557a28ca870b12c2a076857cc` / `3fbe4df9fc2bdc98d935341ef062ebe40e72cac2cf04c21535e402007fceefac` |
| Attempt 03 unsigned/signed worker SHA-256 | `c94ce76fce24267de79c5fab3dff42c5fddb50b203e65305262e1f0ae771e531` / `a6c084f63444655ac829cef0589e6bcec62a2820b28becabfdb1a2780919079a` |
| Config SHA-256 | `2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899` |
| Tokenizer SHA-256 | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| Processor SHA-256 | `022f906548d5a89d3a18c61c0a87de760db8c81d618437501baa3dcdc14d4ecd` |
| Template SHA-256 | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| Index SHA-256 | `f7e35f06cb23dba09dea6197545111c092459f001715beead877c5f84959ccd4` |

The original prepared folder retained the vision tensors. The installed
`hybrid_prepare --upgrade-images` helper generated processor metadata and the
manifest in place using existing shards. It completed without downloading or
copying weights. A private backup of the pre-upgrade metadata was retained.

## Workload and controls

- Only one model worker may run at a time; the canceled baseline is stopped.
- Startup is a functional observation on the user's machine, without a quiet
  host or cold-cache screen. Do not infer a controlled speedup from it.
- Candidate first-start includes the app's automatic image checks if no matching
  private check receipt exists. Keep this boundary separate from native readiness.
- Playground verification uses a private synthetic 224 × 224 RGB PNG, not user
  content. Image attachment, request completion and an image-grounded answer are
  the functional gates. Record only aggregate results and hashes.
- Fixture SHA-256: `ba81a6229409d4f1b280d3c3d3971a5d7b84894229e22f20bc779ed3a7fa4e1f`.
- Existing 300-second readiness and image-check deadlines remain unchanged.
- Full-model decoder numerical parity, broad visual quality and image throughput
  are outside this bounded installation verification.

## Attempt history and correctness

| Trial | Observation | Disposition |
| --- | --- | --- |
| Baseline installed start | Active single-core hashing, low RSS; user canceled before readiness | Inconclusive startup timing; no automatic crash or timeout established |
| Candidate release builds | Rust 1.98.1 locked builds passed | Attempt 02 and 03 workers built |
| Initial test registration | Plain Rust `#[test]` conflicted with the engine's custom test framework | Invalid setup; replaced with `#[stark_test]` |
| Attempt 02 native controls | Two known-vector/order and corrupt/missing shard tests passed | Bounded component correctness |
| Attempt 02 Playground control | Attach button enabled; PNG accepted; completed answer correctly identified both shapes, colors and locations | Bounded visual smoke, not numerical parity |
| Attempt 02 local check receipt | All six local image checks true; receipt SHA-256 `dff21b00ef99dfcdf1efa915c470ace62419e291c45f572e014d670e5903cc2b` | `locally_tested`, not reference parity |
| Initial Python tests | Five unrelated synthetic download tests hit the host's <4 GiB disk guard | Invalid setup; mocked fixture headroom, retaining explicit rejection overrides |
| Python helper setup | Three synthetic import checks lacked the required template helper | Invalid setup; used the existing verified helper on PATH |
| Attempt 03 native controls | Four targeted engine tests passed: SHA order/vectors, corrupt/missing shards, private/manifest-bound cache, same-size mutation with restored mtime | Cache must invalidate on ctime changes |
| Attempt 03 Python / UI controller controls | 25 preparation tests and 139 controller cases passed | Preparation/cache publication and retained capability/event handling |
| Attempt 03 cache seeding | Existing 55 shards hashed in 11.340 seconds; no model download/copy | One-time preparation; excluded from ordinary startup boundary |
| Attempt 03 UI activation | Installed signed app; computer-use tool reported locked Mac | Timing/visual verification pending unlock |
| Attempt 03 native startup | Two serial installed-worker starts advertised readiness and image input; no generation requests sent | Native lifecycle passed; app local checks excluded |

## Performance, installation and limits

| Boundary | Baseline | Candidate | Conclusion |
| --- | --- | --- | --- |
| Complete native/app startup | Canceled; not measured | Attempt 02 Ready first observed at 31.753 s; still loading at 5.163 s | Single UI observation; rejected latency, no exact completion time |
| Playground attachment and image request | Disabled for the text-mode artifact | Attempt 02 passed shape/color/location smoke | Functional execution only |
| Attempt 03 native startup | Historical user estimate ~8 s, not reproduced here | 11.875 s, 10.478 s | Spawn-to-native-readiness, two observations, no quiet/cold-cache screen; app checks and UI publication excluded |
| Attempt 03 complete Playground startup | Not measured on matching baseline | Pending unlock | Target: return toward prior startup latency |
| Peak memory / generation throughput | Not measured | Not measured | No inference-performance claim |

The attempt 02 app used Developer ID hardened signing with a secure timestamp,
build `20261110003125478391`; it was a development build based on dirty source
`b0c0d9fb3eb2754879c69bb7883835638201a0fc`. The build manifest preserves its
per-file source and executable hashes privately. This is not a release.

Append attempt 03 installed hashes, observed startup boundaries and the final
functional result after execution. No PR was requested.

Attempt 03 is signed with the same Developer ID team, development build
`20261110010236089137`. Its private app manifest binds the dirty source tree and
the worker hashes above. The preparation cache initially had SHA-256
`3b30919341b36f6cc36e2083be9c2163b0dc7b1635590b4fab3d80018070e46b`;
all 55 per-file digests and open-descriptor signatures matched the current
manifest and files. Native loads can atomically rewrite this disposable cache.

The private aggregate native-startup receipt has SHA-256
`905edd42a29b4c4d9bf23ccc032ebcfc032f94845d89bb0d0ff04cb56bddc0b0`.
Each run used the installed signed generator with the same model path,
`--vision-staging-root` (fresh private directory), and `--prefix-cache`; it
closed stdin and exited after receiving readiness, without submitting a prompt.
The two native timing observations are not full-app timings or a controlled
old/new speed comparison.
