# Qwen3.5 MoE default-image text parity

## Attempt QMOE-20261109-01

Registered before model execution. Base source: `5f702450`. This attempt begins
IMGPAR-08/09 from the [default-image worklist](../../IMAGE_DEFAULT_PARITY_PLAN.md).
It qualifies only the existing affine v4 artifact named below. The official
BF16 v5 payload is not present, and is not downloaded by this attempt.

| Field | Fixed scope |
| --- | --- |
| Source | `mlx-community/Qwen3.5-35B-A3B-4bit@1e20fd8d42056f870933bf98ca6211024744f7ec` |
| Converted weights | `2f762b2629aaefc2fea6eb37ff6caaa4747fdc23a040e36f331fa4a056c6cad7`, 20,394,001,296 bytes |
| Converted config | `1fe4929c5df2c9c5dae321e82f5cf3e181c05136e385fb9a016436ba3146c525` |
| Target | Affine MoE v4 decoder, BF16 tower; production source-order affine projections, BF16 activation boundaries, F32 DeltaNet recurrence |
| Correctness gate | Image-loaded cached text equals fresh text exactly for matched sampler/seed; expected prefix counts and image invalidation must agree |
| Bounded workload | Synthetic text, context 2,048, 1,040 input tokens, at most four output tokens; greedy and seeded sampling; image transaction emits one token |
| State | Attention KV/positions plus a 1,024-token non-final prefill snapshot of DeltaNet recurrence and convolution history; expert banks are immutable, route scratch is overwritten |
| MTP | Explicit zero drafts: the selected affine source has no predictor tensors |
| Privacy | Requests and responses stay in memory; private receipts contain aggregates and hashes only |

The affine source index has 2,090 tensor names and zero MTP tensor names, despite
`mtp_num_hidden_layers=1`. This is a source artifact limitation, not proof that
the Qwen3.5 MoE family has no MTP. The official BF16 source at
`59d61f3ce65a6d9863b86d2e96597125219dc754` declares a head and its index contains
785 MTP tensor names among 1,811 total. Fresh metadata inspection matches index
SHA-256 `d8d0b7ca4e61ae107e3e87a3ff21136b3ac7c789e64bb24267227ca804e04205`.
Of those official predictor names, 768 belong to routed experts, four to the
shared expert, and six to attention; the fusion matrix is present. The expert
predictor graph needs its own implementation and qualification. The retained
affine index SHA-256 is
`56f02123353b7fe444b287a46a779a836cb939860502908da4a79e3b43931cdb`.

The loaded image decoder has 30 DeltaNet layers. Each extra prefix snapshot owns
`32 × 128 × 128 × 4` recurrence bytes plus `8,192 × 3 × 4` convolution-history
bytes per layer: 65,863,680 bytes (62.8125 MiB) total. This is admitted state
payload accounting; it excludes the already retained attention KV capacity,
allocator overhead, activations and process RSS. Expert bank weights are
immutable; routing decisions/activation scratch are recomputed for the suffix,
so there is no routed-expert cache to snapshot.

## Performance controls and remaining scope

Model loads and GPU work are serialized with the other family workstreams.
Loading, hashing, and correctness checks are excluded from inference timing.
No performance acceptance is declared before a valid matched loading-mode arm
exists. The ordinary text loader selects `NumericalProfile::Legacy`, while the
image loader selects `Qwen35AffineBF16`; treating those as identical graphs
would incorrectly attribute arithmetic-policy changes to image residency.
A coordinated internal tower-omitted loader arm must preserve full admission,
decoder weights and arithmetic before measuring residency overhead. A separate
prior-text comparison remains necessary to detect regressions against the
previous product path. Neither comparison inherits the small dense BF16 result.

## Attempt QMOE-20261109-02: corrected scope

The first recipe used a human-readable cache scope. The worker correctly
rejected all six scoped text requests with `invalid_generation_request`, because
the contract requires exactly 32 lowercase hexadecimal characters. This was an
invalid cache test, not an engine restoration failure. Two unscoped fresh text
references and the real image transaction executed successfully. Preserve the
first private receipt SHA-256
`0c1679089bf16df513e48fa29bfb0f236afb12f6e2db330b0ae6a4b73df86f86`.

The corrected recipe uses two valid opaque scopes. It reuses only the first
receipt's successful fresh-reference output hashes, and rechecks worker,
artifact/config and synthetic-input identities before comparing results.
All workload settings and exactness criteria remain unchanged. No earlier
failed scoped request contributes cache evidence.

## Corrected results and disposition

The seven corrected cases passed on the complete 35B affine-v4 artifact. All
six text cases exactly match the appropriate fresh greedy/seeded output and
text hashes from QMOE-20261109-01. Fresh controls used the same executable and
image-loaded artifact with prefix caching disabled. Successful generation,+rather than equal error replies, is required by every passing text case.

| Corrected case | Expected / actual reused tokens | Exact matched fresh output |
| --- | --- | --- |
| Cold greedy text | 0 / 0 | Yes |
| Repeated greedy text | 1,024 / 1,024 | Yes |
| Seeded stochastic text on the same prefix | 1,024 / 1,024 | Yes |
| Actual synthetic image, one output token | 0 / 0 | Separate image operation; not compared to a text request |
| Text following the image | 0 / 0 | Yes |
| Repeated text after recovery | 1,024 / 1,024 | Yes |
| Different valid host scope | 0 / 0 | Yes |

Readiness reports image schema 3, family v4, enabled prefix caching on the
1,024-token grid, and `mtp.supported=false`, `max_drafts=0`. The image execution
is real, through private PNG staging and `prepare_image`/`generate_image`; it
does not stand in for full 35B decoder/image-logit reference parity. MTP was
explicitly zero for every text request, and the source has no head to execute.
The worker still reports the image artifact's reference qualification as
unverified; these cache checks do not promote that independent qualification.

The release executable was built from `5f702450` using Rust 1.98.1 with a distinct
worktree target directory. Worker SHA-256:
`c636d35720ea5acbf98036051b8f137c3e6bb145719aa3d5968ac0f86aba33d2`.
Cargo lock SHA-256:
`85c73ea6bdb1ece383aca9502820095be4dba8818d6d7e348caca938507e18d6`.
Converter lock SHA-256:
`df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5`.
The corrected receipt is
`f6048cdb3147ab9bfdaa4bad7e11d6b3e1681b0ca939d959ca976566f2bba501`.
The private driver uses the existing prefix validator's worker, tokenizer,
synthetic PNG and hashing helpers. Python 3.13.13 ran through locked `uv`.
Driver SHA-256:
`e620a3881d44a97345bd46deb24046856595f5321de1cd17c82553822534a591`.
The 1,040-token synthetic input hash is
`8d9b86530275a508eab883f952a901f4a7768a7e80aa583172424a94dfa5fe94`;
each successful text case sampled two output tokens within the four-token
budget. The output and generated text hashes remain in the private receipts.
Device: Apple M5 Max, 40 GPU cores, 128 GiB physical RAM, macOS 26.7.2
(`25G332`). Repository policy and diff whitespace checks passed.

The implementation already merged in #110 works for these bounded MoE-v4
cache controls; this attempt changes no engine code. It does not qualify all
cache edge cases: shrinking/divergent/context-changing prompts, reset, error,
cancellation, restart and host delivery/revocation paths still need this
artifact's broader matrix. Shared model-free lifecycle tests are separate
evidence, and were not rerun by this documentation/qualification attempt.

Request durations are diagnostic only. Corrected cold text ranged from
30.90 to 89.52 seconds, while the three prefix-reused requests were 0.87–0.92
seconds. Load cost was approximately 70–74 seconds. No alternating loading-mode
comparison, host thermal/paging eligibility screen, timing acceptance bound or
peak-RSS measurement was performed, so these observations establish neither
default-image speed parity nor a qualified benchmark gain. Further performance
work must include the prior qualified text graph as well as the matched
tower-residency control described above.

Full 35B decoder numerical parity, broad task quality, all formats and contexts,
text-speed qualification, BF16-v5 execution, expert-head support and
installed-app behavior remain open. No whole-family completion is claimed.
