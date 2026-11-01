# Qwen3.8 exact MTP sampled generation optimization

Report date: 2026-10-30. The predeclared throughput runs and their route-hash,
count, and worker-identity checks are complete. Regression, static,
lifecycle, and final execution-source identity checks passed.

## Long-run results

The 16-pair top-k 20 run pools **78.33 tok/s with MTP on** against 67.46 tok/s
off, a **16.11%** gain. All 16 pairs favor MTP. Per-request MTP rates range
from 71.54 to 83.28 tok/s; 8 of 16 requests reach 77 tok/s.

The eight-pair nucleus run pools 71.01 tok/s on against 70.60 off, a **0.58%**
gain. MTP wins 3 of 8 pairs. No on request reaches 77 tok/s, and median request
wall time is slower with MTP on. The measured average clears 77 only for the
top-k mode; it does not provide a per-request 77 tok/s floor. The 77–80 tok/s
target is not qualified across sampling modes.

| Sampling mode | Pairs | Off tok/s | On tok/s | Pooled change | MTP wins | On requests at or above 77 tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-k 20, top-p 0.95 | 16 | 67.4633 | 78.3329 | +16.11% | 16 / 16 | 8 / 16 |
| Nucleus, top-p 0.95 | 8 | 70.5978 | 71.0097 | +0.58% | 3 / 8 | 0 / 8 |

These two modes use separate measured requests and are not combined into one
throughput figure.

## Scope and controls

The checkpoint is `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at revision
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`, with 55 shards totaling
106,905,155,192 bytes. The same converted Lily q4-xl weights, including the
existing one-layer MTP head and its quantized readout, are used throughout;
there is no weight conversion or requantization. All 55 shard stat identities
match the earlier complete SHA-256 verification. Target execution uses BF16
activations, F32 reductions/logits/GDN recurrent state, and Q8 attention K/V.

All four short top-k windows use the same Qwen3.8 Lily q4-xl checkpoint at
revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, 4,096 input tokens,
256 requested samples, temperature 1, top-k 20, top-p 0.95, seed 42, and
two MTP drafts. Each report contains four paired blocks, one warmup per route,
and eight measured requests. Rates pool 1,020 decode advances per route
(`sampled_tokens - 1`) divided by the route's summed `decode_seconds`; the
prefill draw is excluded. Each route returned all 256 samples on every request.

The off and on route hashes match the corresponding preserved strict-verifier
hashes in all four reports. Within-route determinism holds. The on and off
hashes differ from each other, as allowed by the exact-rejection sampler's
separate draft randomness. Hashes are identities, not generation payloads.
The off route hash is `621ca90417af420aaf9defdc957f7cc1f57f318c08d908be764834abb382b0d0`;
the on route hash is `3e3ab0caeb1b7dcc58c1b8dac0c0cf27810756fe538b7548754d315baadb149c`.

| Short window | Source commit | Off tok/s | On tok/s | Pooled change | Worker SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline | `4fa5c02a` | 70.4748 | 78.0180 | +10.70% | `ec4afa5cdb579b47ae64f1e1c3b8b31eab537ac5b873b68f6dd06e17f1ff0a23` |
| Profile build, profiling env removed | `a7c9e10c` | 70.6542 | 74.3966 | +5.30% | `33960639b8060b9d56ebaf5d2ba4ebec20264bcb9e1547ddbeec811ad8621c23` |
| Batched sparse QSA | `8bec97e7` | 68.1585 | 79.8312 | +17.13% | `200a1e88c9920c1178f73ae08acd78faafedeb4a275290b44edf64312f6435d8` |
| Fused GDN projection/convolution/scatter | `a1a2a5f4` | 66.6554 | 80.4283 | +20.66% | `81945ba24640c044ca43dd91b76577d361f063c290a7a905181da8183c4b6f06` |

These are separate short windows on named builds, not a controlled attribution
of each source change. The profile-disabled window is a valid uninstrumented
throughput measurement for its named build: the profile output environment
variable was removed, CPU profiling counters stayed disabled, and GPU stage
markers return before adding timestamp work. The profile-enabled MTP stage
capture described below is diagnostic only and does not qualify throughput.
The GDN window's 80.43 tok/s is a four-pair observation. The long runs above
provide more paired evidence and show that results differ by sampling mode.

| Build | Full source commit | Source manifest SHA-256 |
| --- | --- | --- |
| Profile build | `a7c9e10cc81517d5c398eb1dba5a8a8753713eb5` | `9171a2046c60da154c5ceccc38dd954d9015f9f4122d494464e8c483c2167ef0` |
| QSA build | `8bec97e72b8f398a4a4123773596f38e07dbd41d` | `b8d0102c65f02e45834399050e2281cbade40ab55b66f7e0e20da8312da8cc57` |
| GDN build | `a1a2a5f4168235680164e0bffb6dd874df4771d2` | `8aa44a9e1703b695245e7b9c01775bb65351a1c13284d6fc930156f8ee30c887` |

The baseline identity receipt records its source commit and worker hash but no
source-manifest digest. No baseline manifest digest is inferred here.

## Sampling semantics and paired protocol

MTP is optional and disabled by default. The off and on routes use the same
loaded checkpoint; off skips execution of the head. On drafts from its actual
filtered distribution `q`, accepts proposals using `min(1, p/q)`, resamples a
rejection from normalized `max(p-q, 0)`, and draws a target bonus after a fully
accepted group. This preserves the ordinary target distribution. Draft
randomness can change the seeded trajectory, so only within-route output hashes
are expected to repeat; cross-route hashes need not match.

Each benchmark case loads one release worker, runs one warmup on each route,
then executes paired routes in balanced randomized ABBA order against the same
4,096-token input prefix, sampling settings, and loaded model. Top-k uses
temperature 1, top-k 20, top-p 0.95; nucleus omits top-k and uses top-p 0.95.
Both use seed 42, two drafts, 256 requested samples, and context length 4,352.
Pooled decode throughput is `sum(sampled_tokens - 1) / sum(decode_seconds)`;
the prefill draw is excluded because it occurs before decode. Every measured
request must return all 256 samples and acknowledge the ready worker’s exact
MTP sampling protocol. The benchmark removes `STARK_QWEN38_PROFILE_OUTPUT`
before worker launch. The profile-disabled worker remains valid throughput
evidence for its named build; the timestamped profile run is diagnostic.

The host is an Apple M5 Max with 40 GPU cores and 128 GiB, macOS 26.7.1
(`25G241`), on AC power with Automatic energy preferences. Desktop GPU activity
was not controlled. The reported windows do not establish performance under
all ordinary desktop activity.

## Exactness and component evidence

The retained attention change batches up to four sparse queries through the
existing score, select, split and combine kernels. Independent query scratch
replaces the previous per-row reuse of ordinary decode scratch. Per-query
causal visibility and the ordinary split/head-group plan are retained; dense
queries still execute ordinary decode. At three verification rows this removes
96 dispatches and 96 dependency barriers across the 12 attention layers.

The retained GDN change replaces separate projection, split, capture and
convolution dispatches with one guarded Q4 projection/convolution/scatter
kernel. It shares raw weight loads across independent token dots, keeps the
16-lane packed-dot order and BF16 projection boundary, and evaluates the
original runtime convolution loop. It records projected inputs and writes the
alternate window while retaining the original for rollback. The existing F32
register scan and output projection follow. This removes 108 dispatches across
36 GDN layers; unsupported layouts retain the previous path.

The new `stark_packed_rows.metal` and `stark_gdn_projection_rows.metal`
extensions retain Apache-2.0 licensing and derive their arithmetic from the
vendored `quant.metal` at upstream revision
`e22d58fa7cc488493fb1b5d810a198bd29730aeb`. The original projection and QSA
shader sources and the exact sampling implementation are unchanged.

The QSA control compares 140 full target readouts (34,764,800 F32 values) with
zero unequal values, zero top-1 changes, and zero measured full-distribution
KL/TV. A separate GDN production control reports the same counts and exact
result for the combined fused-GDN and batched sparse-QSA source. These controls
cover the tested prefix, verifier-width, acceptance, rollback, and early-drop
cases. They are finite controls, not an exhaustive enumeration of possible
model histories. The final post-long-run regression suite is recorded below.

An isolated GDN projection/convolution/scatter candidate has a separate exact
control: 128 component cases and 448 rollback-prefix cases compare 34,062,272
bytes with zero unequal bytes. Its source identity is based on `c4a5b008` and
marks production sources unmodified. A separate synthetic streaming receipt
reports mean GPU speedups of 1.19x, 1.33x, and 1.31x at 2, 3, and 4 rows,
respectively. These candidate-component measurements do not establish the
effect of the GDN MTP worker change or an end-to-end MTP gain. The initial
component identity was written before timing execution and still labels
performance validation "not run"; later standalone exact and streaming-timing
receipts passed. Treat those later receipts as separate component evidence,
not as production GDN MTP performance.

The draft head has known catch-up-shape differences: maximum measured full
vocabulary TV is 0.004025, with no top-1 mismatches in the recorded control.
The sampler uses the actual filtered draft distribution `q`, so these
differences affect acceptance efficiency, not the target distribution.

## MTP stage profile

The private profile analysis covers two 64-sample on requests and two off
requests. Across the on requests, 50 rounds propose 96 tokens, accept 76, and
verify 146 rows. Aggregate host wall scopes total 1.5385 seconds: draft
0.1468 seconds (9.5%), verify 1.3355 seconds (86.8%), finish/rollback
0.0395 seconds (2.6%), and target sampling 0.0168 seconds (1.1%). These are
elapsed scopes that include synchronous GPU waits; they are not CPU-active
time.

The verified GPU command duration is 1.2603 seconds. Its profiled phase shares
sum to 100%: GDN 22.0%, expert gate/up plus shared down 21.0%, HC 17.2%, QSA
16.7%, expert down/combine 12.2%, vocabulary head 6.1%, MoE routing/shared
setup 4.3%, PLE 0.5%, and embedding
under 0.1%. Timestamp collection affects scheduling. These phase percentages
locate work inside verifier GPU commands; they are not production throughput
measurements or causal speedup estimates.

## Shelved expert-union experiment

The private union-v3 experiment passes exact component controls for 768
projection cases (12,550,080 compared bytes) and eight complete-MoE cases
(102,400 compared bytes), with zero unequal bytes. It is not a full-model or
generation control, and production sources were not modified.

Synthetic timing is sensitive to routed-expert overlap. Full-rotated overlap
was only 1.018x paired geometric-mean speedup (20/30 pairs faster); partial
overlap was 0.879x (9/30 faster); disjoint experts were 0.672x (0/30 faster).
The recorded recommendation is not to promote this experiment. It remains a
private component-qualified result.

## Longer paired runs

Both runs use the same frozen GDN worker (`81945ba2…4c6f06`), checkpoint,
4,096-token input, 256 requested samples, temperature 1, top-p 0.95, seed 42,
and two drafts. They use separate sampling modes and are not pooled together.
Every measured request returns 256 samples. Each on/off route hash matches
its corresponding preserved strict-verifier hash.

| Sampling | Off route hash | On route hash |
| --- | --- | --- |
| Top-k 20 | `621ca90417af420aaf9defdc957f7cc1f57f318c08d908be764834abb382b0d0` | `3e3ab0caeb1b7dcc58c1b8dac0c0cf27810756fe538b7548754d315baadb149c` |
| Nucleus | `9c33d3a4ef1fef97562e00f7f88b5dc287aacadb16b50c4739aec28dbea1b614` | `64c3861cfdfd7142f2910823bda99d63af853aba57ec7c3eccac9ca0190b62c6` |

| Sampling | Pairs | Off tok/s | On tok/s | Pooled change | MTP wins | Advances per route |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-k 20 | 16 | 67.4633 | 78.3329 | +16.11% | 16 / 16 | 4,080 |
| Nucleus, top-p 0.95 | 8 | 70.5978 | 71.0097 | +0.58% | 3 / 8 | 2,040 |

Top-k per-request off rates have min/median/max 60.43/68.60/72.00 tok/s;
on rates are 71.54/78.30/83.28. Eight of 16 on requests reach 77 tok/s; seven
reach 80. The pooled 78.33 tok/s clears 77 for this sampled mode, but it is not
a per-request floor: half the on requests are below 77.

Nucleus per-request off rates have min/median/max 66.98/71.05/72.90 tok/s;
on rates are 70.19/71.11/71.50. Paired gains have min/median/max
-1.92/-0.43/+6.47%. No on request reaches 77 tok/s. Median request wall time
increases from 5.692 seconds off to 5.763 seconds on; this mode's small pooled
decode gain does not show a consistent request-level improvement.

Across the two modes there are 48 measured requests and four warmups, or 6,120
decode advances per route in total. This is a count only; do not calculate a
combined throughput because the sampling modes are different. The observed
range includes one mode with a strong, consistent paired gain and one with a
small gain, three of eight wins, and slower median request wall time. The
77–80 tok/s target is therefore not qualified across sampling modes.

The first eight top-k pairs pool 66.91 tok/s off and 78.08 on (+16.70%);
the last eight pool 68.03 off and 78.59 on (+15.52%). The average stays in the
target range in both halves. The earlier four-pair GDN window's +20.66% and
the long window's +16.11% describe their individual measurements.

## Final validation

The final regression run used source commit
`a1a2a5f4168235680164e0bffb6dd874df4771d2` and confirms the benchmark worker
SHA-256 remained
`81945ba24640c044ca43dd91b76577d361f063c290a7a905181da8183c4b6f06`. It
passed 68 generator tests, 39 scorer tests, two profiler tests, one ignored
GDN numerical/rollback test (128 component cases and 448 rollback prefixes),
666 native tests, seven C++ tests, three documentation tests, and both Clippy
checks with warnings denied. Native testing had 47 ignored tests. Static
validation passed 17 Python tests, repository policy, Rust formatting, Ruff,
and Ruff formatting. Lifecycle validation passed 23 generation requests,
including bounded-tail cases, cancellation recovery, invalid-request recovery,
and greedy on/off hash equality.

The final seal confirms the frozen executable, all 55 shard stat identities
and host power preferences are unchanged. Of the measured 912-file inventory,
911 files remain byte-identical; the only difference is the subsequent
`UPSTREAM.md` provenance update. Execution sources are unchanged. The final
seal has SHA-256
`39d8bd979c1d0bba0c8556607be5d7f302721582393ef4662ea64e7197fc5fab`.
The aggregate regression, static, and lifecycle receipts have SHA-256 values
`3fb8b01902a3f641166a32bb5f11577aefaa787514788fe824f6adc273e75ab7`,
`548d1fab2a132f6ad8a1bec60d32aa186174187fe9642661653fc70e7ba60175`, and
`4b0015ee866d439c26905a2f25e0895fd2e45c3f8bd2be3f73e67abc80762d87`,
respectively.

## Private evidence identities

All paths below are ignored runtime artifacts and must remain private. Hashes
identify the exact aggregate receipts used for this study.

| Artifact | SHA-256 |
| --- | --- |
| Baseline top-k two-draft report | `390cc1592ed938a43b70bed776f657f657a56fe7baa8523369a0d3f2cf8c27a7` |
| Profile-disabled top-k two-draft report | `0766c55b024f68994f07e09af713f2e4de4de6e092e7cf7ad4e404dd6872ee44` |
| QSA top-k two-draft report | `46d83dbc8e1f0172899420604e7ae7c7ed4bedc574d693c048eee4fd69ef145f` |
| GDN top-k two-draft report | `8638c6636270023a421dfb4fd1f7c66d0dc6a881912fb8c723878de9f5515ac3` |
| Long top-k 16-pair report | `500bde7e251a339f4622df72739accfc25a6ef87493eee06df548392ea082a3a` |
| Long top-k aggregate analysis | `17fd0b85bc5fad757b66abd942e8fa0296ee639b11045857754a992b1f1acafa` |
| Long nucleus eight-pair report | `65adb0414857ad608c36a49a59539727068a5f44fc38c513d80c3d4335649207` |
| Long nucleus aggregate analysis | `7517138e21f6a2b8a028a579a1c9edf5f884c27608cc95e1451e422ec4b217a0` |
| Strict nucleus replay control | `7d46da77ca58d1c8d9a35caae4c233763e90c585ee3aa5adc40e85fbe99afab2` |
| Initial build identity | `f1025abf2fabd1d00d16d55d228afb48a381fe37b1e55b4337194bbdb0491e1c` |
| Profile build identity | `3b8c982db85cf2a36b0c4bcb6d2e12f2f6b962bb59eca06a4da0322c7ce9abbd` |
| QSA build identity | `01e834394ebd2a4f8f699f3d6fff4cd5be7fd67ab17e6d659c5a87711cddf757` |
| GDN build identity | `a6e2a4a86930a57f74fd30b6a8974a285f2c3d13daebf8ea08dcc6d82a818afc` |
| QSA full-model control | `134b961ad91673fcb41a910596a640b36594c2103ad1c59dbd34696badbad35b` |
| GDN fused full-model control | `e7b6a84d3c2f76f7cf6e805f6e8c3f34d389b8f666ba0f1e35f0e32191cc7ed5` |
| Preserved strict-verifier top-k report | `e3e47e130567fe5d3b495bc0cbe48be47716a6af3da0d7222da3dfd70f2de624` |
| Strict-verifier exact control | `53760ec81365890781ddb37d44944f8c83130a5f639c5591cf9cc6b84dd18b98` |
| GDN component identity | `0904c6372aa456995aa53f8042313d064b57e649c4961c0503a8dba0d891a5da` |
| GDN component exact controls | `389f682662cb56dae03879f3e3ebbf7c9d10d245d2aa5f45b64213f01054c968` |
| GDN component numerical control | `b0113757b488ad823a21000fa28765c5d3e9818c7c014abaa37cd6e42d2d09d0` |
| GDN component streaming timing | `a6dd42ad2347d0185dc0801909c267cc710b9e2c0cbcbc72bfdd00dd4c97a43a` |
| MTP stage analysis | `b13950ec02e1a6525d5364851618e48394bb56b410855df513967bb79c3136fa` |
| Checkpoint weight identity | `3d2c0973e28b4b41cdaf2c3af86a13495cbcebffe56911f0833a0d798eccbc99` |
| Checkpoint configuration | `2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899` |
| Checkpoint source receipt | `7f5c685f35fc4c5d1509f6379feb1a965cb625f7a1ce76a0b5fe69983561dd30` |
| Public-code corpus | `daecdf178c6429a721f9e1b5ae379c664adb8f8af55b381535355636ec469955` |
| Tokenized 4K input identity | `fd1222c74a823b3a6cf5ed3ffd3dd22cb627afde17c21cef8b2150ac9e14b322` |
| Expert-union v3 exact controls and timings | `976d94b6b0a60f615f4ed44cfc8a6334d9e3e64a61dc624931fda45178750b88` |
| Expert-union v3 build record | `9e19d9ebd43f4b984b64ca90c517099e5d6dac0d355399f2c7a6ebd8cfe2889f` |
| Final regression validation | `3fb8b01902a3f641166a32bb5f11577aefaa787514788fe824f6adc273e75ab7` |
| Final static validation | `548d1fab2a132f6ad8a1bec60d32aa186174187fe9642661653fc70e7ba60175` |
| Final real-worker lifecycle control | `4b0015ee866d439c26905a2f25e0895fd2e45c3f8bd2be3f73e67abc80762d87` |
| Final execution-source and identity seal | `39d8bd979c1d0bba0c8556607be5d7f302721582393ef4662ea64e7197fc5fab` |

## Reproduction

The aggregate receipts and profile output are private ignored runtime artifacts;
only hashes and aggregates are included here. To reproduce, use the pinned
checkpoint revision above and materialize the benchmark corpus from the pinned
source file at the [pinned upstream revision](https://github.com/fabiogreter/lily-qwen3.8-flash-next/blob/e22d58fa7cc488493fb1b5d810a198bd29730aeb/src/qwen4exp/model.rs). Verify that
corpus SHA-256 is
`daecdf178c6429a721f9e1b5ae379c664adb8f8af55b381535355636ec469955` and that
the tokenized 4K input identity is
`fd1222c74a823b3a6cf5ed3ffd3dd22cb627afde17c21cef8b2150ac9e14b322`. The
worker used for both long runs has SHA-256
`81945ba24640c044ca43dd91b76577d361f063c290a7a905181da8183c4b6f06`.

Build the generator separately from optimized scoring, using the recorded
native source revision and pinned toolchains. From the repository root:

```sh
cargo +1.98.1 build --release --locked -p stark-generate
mkdir -p runtime/mtp-reproduction
chmod 700 runtime/mtp-reproduction
uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_qwen38_mtp.py \
  --model models/qwen38-lily-q4-xl \
  --worker target/release/stark-generate \
  --corpus runtime/pinned-lily-model.rs \
  --input-tokens 4096 --tokens 256 --blocks 16 --drafts 2 \
  --top-k 20 --top-p 0.95 --seed 42 \
  --output runtime/mtp-reproduction/top20.json

uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_qwen38_mtp.py \
  --model models/qwen38-lily-q4-xl \
  --worker target/release/stark-generate \
  --corpus runtime/pinned-lily-model.rs \
  --input-tokens 4096 --tokens 256 --blocks 8 --drafts 2 \
  --top-k 0 --top-p 0.95 --seed 42 \
  --output runtime/mtp-reproduction/nucleus.json
```

The resulting timings will vary by host activity. Compare route hashes and
counts to the pinned evidence above; do not use a reproduction timing as a
replacement for the recorded source-bound result. Private artifacts use
mode-0700 directories and mode-0600 aggregate reports. No prompts, output
text, token vectors, logits or decision probabilities are persisted.
