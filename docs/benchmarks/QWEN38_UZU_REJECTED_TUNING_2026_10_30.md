# Qwen3.8 Uzu rejected tuning trials, 2026-10-30

The Q4 gathered-matrix, Q8 hyperconnection (HC), and execution-only WAR
barrier trials, and the later selective GEMV decode trial, are rejected for
production. Their sampled-generation
observations do not establish a meaningful throughput gain. The Q4 and Q8
component gains remain source-bound research results. Their private patches,
sources and aggregate logs are preserved.

This record covers the new standard Uzu execution graph described in
[Qwen3.8 Uzu execution](../QWEN38_UZU_EXECUTION.md). The earlier
[converted-format loading and sampler study](QWEN38_UZU_SPEED_2026_10_30.md)
measures a different executor and remains unchanged.

## Workload and precision

The four initial full-model cases use the same converted artifact and unchanged
source Q4/Q8 codes with BF16 affine parameters. There is no speculation, MTP,
re-quantization or cache-precision change. Activations are BF16; reductions,
router, vocabulary logits and GDN recurrent state use F32. Attention K/V
retains signed Q8 codes and F16 scales. HC normalization is rounded to BF16
before the standard projection. Standard GEMM retains F32 split-K partials
until its final output cast.

The serialized text workload uses 4,096 input tokens, a 4,352-token context,
256 sampled outputs, temperature 1, top-p 0.95, top-k 20 and seed 42. Each
worker runs three requests; the first is warmup. The reported warm rate is
the median of the remaining two observations. Decode rates count the 255
continuation steps after the first sample. All requests finish at the length
limit, and every output digest matches across these four cases. No token
vectors or model content are published.

The host is an M5 Max with 40 GPU cores and 128 GiB unified memory. Automatic
AC Energy Mode is confirmed by the session and current preference readback;
there is no power-mode mutation during these runs. Clocks are not recorded
per run. These are separate worker-load windows, with no same-loaded ABBA
comparison, confidence interval or control for frequency drift. These runs
do not qualify throughput during normal desktop multitasking.

| Full-model worker | Warm decode tok/s | Change from baseline | Warm prefill tok/s |
| --- | ---: | ---: | ---: |
| Lexical-cutoff router baseline | 43.967673 | — | 136.234950 |
| Q4 gathered tile trial | 44.103186 | +0.3082% | 131.986374 |
| Q8 HC tile trial | 44.301812 | +0.7600% | 134.282371 |
| WAR-only barrier trial | 43.645546 | −0.7326% | 135.909632 |

Passing request/repeat checks establishes the sampled replay result for the
named workers. These short windows do not establish a stable speedup or the
77–80 tok/s target.

## Q4 gathered projections

The narrow selector trial changes independent output-row assignment for
the target MoE gate/up and down projections, using four SIMD groups with
four and two results per group respectively. It preserves the K reduction,
quantization, global expert-row addressing and full-bank affine strides.

Synthetic component controls rotate selected experts, use 32 warm and 256
measured alternating AB/BA pairs, and compare every BF16 output bit. The
selected gate/up geometry has a pooled GPU speedup of 5.1457%; down has
11.0287%. All 256 output pairs are exact in each case. The gate/up order
subgroups show +62.65% when the baseline runs first and −31.47% when the
candidate runs first. Down likewise shows +23.93% and −1.08%. Those large
order effects limit interpretation of the pooled component gain.

Full-model warm decode changes by only +0.3082%, while prefill falls by
3.1186%. The measured-shape selector and its admission tests are removed.
The generic gathered policy, independent numerical controls and private
benchmark helpers remain.

## Q8 HC projections

The trial admits only unsigned Q8/group64 ScaleBias GroupOutput matrices
with BF16 parameters/input/output, full-precision activations, one input row,
and no output transform, on Apple10 or newer with at least 32 GPU cores.
HC down (N320/K10240) and injection (N4/K10240) use two SIMD groups with one
result per group; up (N10240/K320) uses eight groups with two results.
Reduction lanes32, group lanes8 and split-K1 remain unchanged.

Synthetic controls retain 48 distinct projection triplets, with 32 warm and
256 measured alternating pairs per geometry. Every selected geometry has
256 exact BF16 output pairs. The component records show:

| Projection | Hot-one pooled GPU speedup | Rotating-48 pooled GPU speedup |
| --- | ---: | ---: |
| HC down | 38.0210% | 25.5598% |
| HC up | 23.0916% | 17.9272% |
| HC injection | 41.5746% | 38.7504% |

Rotating down/up measurements retain substantial first/second-arm effects;
their synthetic timing does not predict full-model throughput. The
full-model warm decode observation is only +0.7600%. The special selector,
its two guard tests and its extra compiled Q8 geometries are removed.

## Execution-only WAR barrier

This private trial changes a pure write-after-read dependency to a Metal4
execution barrier with visibility `None`. Any dependency on pending writes
retains the existing `Device` barrier. An execution-only barrier clears read
history while preserving every pending write until a later device flush.
Buffer byte ranges and command-buffer boundary barriers remain unchanged.

The installed SDK explicitly defines `None` as an execution barrier without
a cache flush. That API contract motivates the experiment; it does not
establish an inference speedup. The sampled-generation window is 0.7326%
slower than the lexical-cutoff baseline. The patch remains private and is
not adopted into the production command encoder.

## Evidence identities

The request records share these identities. `source_manifest_sha256` names
the model source manifest in the request receipt; it is not a runtime source
digest. Worker hashes distinguish the executable trials.

| Identity | SHA-256 |
| --- | --- |
| Converted configuration | `cd9fef82e5016e282dc1920600776a1fee5c3560b86bc7787b883b83c732410e` |
| Conversion receipt | `fe495b0be28cb114d042e98f9e1451be6118ac6d28b14b7b73bfae688db2f302` |
| Model source manifest | `4bce8ad08a02057e549cdbab0cc401c28849b9b4e4fd306c7fff9649a7a2823d` |
| Input tokens | `fd1222c74a823b3a6cf5ed3ffd3dd22cb627afde17c21cef8b2150ac9e14b322` |
| Sampled output | `873717fe8f8ced6ad2ca7f0dc0c1bb0e74d65943d170752ee4009b02ce82165f` |

| Worker | Executable SHA-256 |
| --- | --- |
| Lexical-cutoff baseline | `1d5ae11058ca0e1280105b03dc071366d668626a367285fa8fdd2a5bedc99255` |
| Q4 trial | `6b74b8e9a2a5a35458ee9c361a6d3c2a8c7668b07f06d7ac4cb5016101189262` |
| Q8 HC trial | `d3f801cebb3dc78ca478f448863121cfc1fe998c31a7524cc3da16d9699d6e16` |
| WAR-only trial | `90abe32524029b6a3ecdbfaea4ee28573cf82b47fba182b1cb533b1bd4119706` |

These private aggregate records are retained with their original contents:

| Artifact under ignored `runtime/` | SHA-256 |
| --- | --- |
| `uzu-lexical-router-top20-4k-v1.json` | `b5930a348d54d08037468b5065fc9f2403df05ec93235ad4abd807000e989f32` |
| `uzu-q4-tiles-top20-4k-v1.json` | `f0741e20c42c3cbafbc82c655c5e550178171a34c3724477e8be65f7235d7b2a` |
| `uzu-q8-hc-top20-4k-v1.json` | `1949c6167d3887e7558592096bbbc8c238c16f3230d581821c9192e8d36d14d7` |
| `uzu-war-only-top20-4k-v1.json` | `6f357d56d247fd4d7a6aa61f89e95a7119b3c1c2b15fdd24e77032644dea4faa` |
| `qwen38-gather-tiles/results-v2.log` | `d390319d0b957268efecd4245038827cb375af5854e153e6f8669e591db571b1` |
| `qwen38-hc-q8-tiles/results-v1.log` | `f5a7721a0e3e897a4c4cfb7088b34347ba1d2c938865ae1e9508628effc67428` |
| `qwen38-hc-q8-policy-trial/q8-hc-policy.patch` | `b7f9c7d44e808dcdd7a888e95fabb06dca3aad2d331b947ec5fb60df8daf8b9d` |
| `uzu-war-only-barrier.patch` | `e4025f477035955082ca2c9cdc5aacfc4e48694c78cb46d6cd7f645c6938b566` |

The F32 split-K correctness fix, generic policies and other independently
validated changes are preserved. Removing these experiments does not
revert their fixes or rewrite earlier validation evidence.

## Selective GEMV decode and scheduling

The later `trial-v2.patch` used normal GEMV geometry with selective integer
code conversion and rowwise load scheduling. It enabled rowwise scheduling
for HC down/up and both options for gathered MoE down. Source quantization,
canonical GroupOutput affine strides and the no-speculation workload were
unchanged. Synthetic exact controls motivated a full-model trial; they did
not predict sampled-generation parity.

The production trial is fully reversed. The complete patch, its admission
tests and shader/header changes are retained privately for reproduction.
The name “exact GEMV” in private filenames records the experiment's original
hypothesis; the complete inference path did not satisfy that hypothesis.

One private worker then compared false and true modes against a single
loaded model. The controller used 32 input tokens, a 288-token context,
256 sampled outputs, the sampling settings above, two warmup requests and
four measured ABBA quads. Each measured arm has eight requests. Complete
requests are serialized; selection counters are checked for the preceding
window. Both arms produce 256 outputs ending at the length limit and repeat
their own output digest exactly. The arms produce different digests, so
the cross-arm parity gate fails.

| Same-loaded observation | Value |
| --- | ---: |
| False-mode pooled decode tok/s | 48.091909 |
| True-mode pooled decode tok/s | 48.182943 |
| Pooled rate difference | +0.1893% |
| Approximate 95% paired-quad interval | −7.2095% to +8.0694% |

The interval uses Student's t approximation on four ABBA quad log-duration
ratios, with three degrees of freedom and independent normal quad-error
assumptions. It is a short-window diagnostic. The failed parity gate means
these timings are not accepted performance evidence. This result supports
neither retention nor the 77–80 tok/s target.

The separate 4K windows are preserved as raw observations:

| Worker/load window | Warm decode tok/s | Warm prefill tok/s |
| --- | ---: | ---: |
| Fresh standard Uzu baseline | 44.694533 | 184.743942 |
| Selective GEMV trial | 48.076069 | 220.148070 |
| Fresh earlier production executor | 77.823437 | 2285.970729 |

Each window has one warmup and two measured requests. The standard Uzu
baseline and GEMV trial have different continuation digests and separate
model loads. Their rates do not establish a speedup. The earlier production
executor is the previous Lily-backed route with its own sampled continuation;
its 77.823437 tok/s observation is an executor comparison, not qualification
of the standard Uzu implementation. None of these windows establishes
performance during normal desktop multitasking.

| Preserved identity | SHA-256 |
| --- | --- |
| Root trial worker | `bf6d447bf7ebcf0602c1acd38726bed2fef820cc6469419ae88d36c40b8f9965` |
| Private same-loaded worker | `d11f8f468921be89d8a233084f1136a2b0bf7093cf33c99ab8f13f77d3168f93` |
| Earlier production worker | `3fca1cb56dc0ab2d8ab9b870a2c5e0ae9a92289900e88cf70768f237f3445544` |
| Root source seal, `qwen38-gemv-exact-seals-v1/workspace.json` | `dc04a9666801b1bbe1a696133d9ba1c2d2bf4aab885e23bd5ca031d419b30342` |
| Private source seal, `qwen38-gemv-exact-seals-v1/private.json` | `34c77e2bdc0271f817373f78b07158587988ba3124ecc8976d697caa30ea8c12` |
| Root `trial-v2.patch` | `859855a7b6fac4e2e06776fa9dad262d473561eb7ebe3c3d141b803a25b65f72` |
| Private same-loaded hook patch | `bf5db10c3bb7ab35f17fa5fa9e8ea620a9e1ca9f069d3e55683129ede168e511` |
| Same-loaded false-mode output | `2b8df86d606e47fd9801f424528e6d673e5827dace09924d5cf85b63212a8e3d` |
| Same-loaded true-mode output | `f1707075d1e4ed73d0cdb5877aa188197f60d411bf32e0617cacbfc86cc4be1a` |
| 4K trial output | `f3c41acc89fed511fa93f2484c16e09397a05a01197c2bfa45d500df964567e4` |

| Private aggregate record under `runtime/` | SHA-256 |
| --- | --- |
| `uzu-exact-gemv-same-loaded-32-v1.json` | `9b96d068f996f78382c616b02f07f5a7ab27f40c5adfab5832c7feeb61f9e66a` |
| `uzu-exact-gemv-top20-4k-v1.json` | `cfc15bbce02a0ed4b477863898bd29f3a7fec2ee37b541c2a20ed2db230af16d` |
| `uzu-lexical-router-top20-4k-current-v2.json` | `dc89271a64020c6c12cc508eb3f1e1eb05c5657461bed998c137d7c37ab13c23` |
| `old-production-top20-4k-current-v1.json` | `9b689da72f36f79b0d3532d2b941271406880a20adcf02938f0af8272fa4447d` |

The separate [35+3 full-vocabulary observation record](../archive/runs/2026-10-30-qwen38-experimental-full-vocabulary.md)
covers another worker built from this experimental source. That study does
not qualify the restored production source. Fresh source/binary seals and
the matching 35-prefix study for the retained Uzu path remain pending.

## Other component experiments

Wider K lanes and fused gate/up projection were also evaluated privately.
They retain the standard canonical weights and backend primitives. Neither
experiment is adopted. Wider lanes passed independent synthetic dot
controls, but most Q8 cases and the Q4/32 case regressed. Gate/up fusion
passed its synthetic projection/activation controls; normal routed Q4
geometry regressed, alternate geometry did not isolate a fusion benefit,
and shared Q4/Q8 did not establish a gain. No full-model generation speedup
was measured for either experiment.

| Private component evidence | SHA-256 |
| --- | --- |
| `full-vocab-control/wide-lane-component-v1.log` | `4875cec44675b9898e161684550de38bedb33f830c48aeeff8bb493b8b72952e` |
| `qwen38-gemv-wide-lanes-v1/source-freeze-v1.json` | `27b323715189b8212fbbf19341df7d8b115958f9b77a9e67cbb7e85de87cd768` |
| `full-vocab-control/fused-gate-up-component-v1.log` | `63e43a429cbbc5ae52daf411cdbfdaafe07f1732e5cf308f42d41b1b222c7396` |
| `qwen38-gemv-gate-up-v1/source-freeze-v1.json` | `6c11dcfeff56f18bc0f771f951b847ffeda0e1e0e038d8a2bc48e3966fb6060c` |
| Fused executed harness, `qwen38-gemv-gate-up-v1/bench_v2.rs` | `a1d93e144b36584dc32c5bdd10a81acb5f25c2f4b43addbd43d7ae595d4f453d` |
