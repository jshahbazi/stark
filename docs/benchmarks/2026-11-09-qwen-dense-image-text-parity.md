# Qwen dense image-loaded text parity controls

This record belongs to attempts `20261109-QDIP-02/05/06`. The initial recipes
were registered before execution. The corrected QDIP-06 artifact/recipe was
fixed in its private driver before launch; its public entry was updated after
launch. It is a bounded control study for the
[dense implementation attempt](../archive/runs/2026-11-09-qwen-dense-image-default-parity.md),
not a family-wide performance qualification. The corrected 27B control fails
the bounded text speed gate; functional MTP is restored, performance is unfinished.

## Two separate comparisons

1. **Resident tower control:** the same admitted headless Qwen image artifact,
   identical decoder and numerical profile, with the tower omitted or resident.
   The validation-only loader keeps complete artifact admission and hashes;
   normal product loading always allocates the tower. The ignored native test
   uses ABBA loads, one warmup and three measured fresh states per load, 192
   input tokens, 32 output tokens, greedy sampling, context 4,096 and MTP off.
   It times iterator first sample and decode, excluding state/stream setup and
   load time. It cannot establish whole-worker request parity or prior-export
   parity.
2. **Current text versus default image graph:** exact Jundot 27B source revision
   `7dd276b77fcb27c982ebd8d8a434539345713fea`, complete native text control
   versus the new complete image/head export. One candidate release worker
   executes both arms, ABBA loading order, one warmup and three measured fresh
   requests per load. Both use the same prompt/token hash, context 2,048, eight
   output tokens, greedy seed 42, three MTP drafts, full original draft readout
   and no retained cache scope. Record actual proposals, output hashes,
   first-token wall time, useful decode rate and whole-request wall time.

The second comparison includes the existing loader-selected graph differences:
Legacy text arithmetic versus the image reference activation, normalization,
attention and affine materialization policy. Those differences are part of the
default-image performance requirement. The first comparison isolates tower
residency and cannot excuse a regression from those changed graph policies.

The fixed decode bound for the short worker comparison is
`max(5%, 2 * baseline_range / baseline_median)`, calculated from the six measured
baseline rates. A loss beyond that bound is a failed bounded parity observation.
Same-mode outputs must match; cross-mode hashes are reported separately. Load
time, warmups and private artifacts stay separate from request timing. GPU jobs
are exclusive and serialized; no thermal, background-load or paging qualification
is claimed. This short workload cannot qualify larger prompts, seeded sampling,
other draft depths, other artifacts or installed app behavior.

## Artifact identities

| Arm | Identity |
| --- | --- |
| Current 27B full text control | Model SHA-256 `bdde7b725452290d9c0d1fb7faf6571a0c792ed78690069e0d0819c02306e8cd`, 17,078,038,512 bytes; configuration `67a36cc0c4ac748fb23da0d8c8093fbcc77e0cd18e521a65f320e7b89de7d89d` |
| Image-loaded 27B with head | Model SHA-256 `6c64740ac1d09365b0099f868114e5c495bac0d1366c27e842c87a9f3a5db647`, 17,999,537,280 bytes; configuration `b4bb4776365308b3e157e21e5fe7777e5650f328879a3f0a4b9981ef978fa2da` |
| Machine | Apple M5 Max, 40 GPU cores, 128 GiB physical memory; macOS 26.7.2 |

## Results

### QDIP-02: headless 0.8B tower residency control

All 16 fresh requests per artifact, including four excluded warmups, produced
matching token hashes and lengths across tower-off/on loads. Six requests per
arm contribute to each median below. These are native iterator measurements;
state/stream construction, worker handling and loading are outside the timer.

| Exact artifact | Tower off decode | Tower resident decode | Observed decode change | First sample off/on |
| --- | --- | --- | --- | --- |
| Official Qwen3.5-0.8B BF16, `2fc06364715b967f1860aea9cf38778875588b17` | 60.1967 tok/s | 60.1963 tok/s | -0.0006% | 0.27605 / 0.27754 s |
| `mlx-community/Qwen3.5-0.8B-MLX-4bit`, `5d894f8cc4ef3e6c88537bf3746ed262f549da6a` | 59.9266 tok/s | 59.9866 tok/s | +0.1001% | 0.72233 / 0.72306 s |

No material decode cost from the resident tower was observed in this short control. It
does not compare either image graph with a historical text export or qualify
other sizes, quantizations, contexts, worker overhead or text MTP.

| Identity | BF16 | Affine 4-bit |
| --- | --- | --- |
| Model SHA-256 | `7c5960e7225b3b36844fa397a85c5e20d281d732e382475062f552c0737c79d4` | `7e127dfb8c78f22b94edd35e52717f50e3d70398fe3668abd2290d088b5b1343` |
| Configuration SHA-256 | `3f0cecb7cb3633bc271eabf69b06ac9f74b377b9ec17b441f5ceac331a71f42a` | `f1c1b857d6cbd935bab28145a0dbaf2b1cbfdea0e93e7b0fd697ee3ad090de98` |
| Private receipt SHA-256 | `ff67fe93fa519d0332c071db9225f3a071dc2fd09d7a977aa13d1235984c5b63` | `16239297524d3b715aa796b479cd8e1923502c0d5c9a536c25179caeaf46b163` |

The pinned-nightly release test executable uses `cpu,metal` and has SHA-256
`708ff0fd9416493e6f368317d2eb6e58731be9b22cba86abc66d55d69f580b90`.
Its underlying production source reproduces the release worker hash listed
below. This control does not replace the separate whole-worker comparison.

The initial QDIP-05 control selected a retained, removed longer-chain diagnostic
configuration containing the obsolete `max_drafts: 7` field. Strict native
loading rejected it before inference. This was an invalid setup, not a measured
text regression. The rejected attempt receipt is
`91c6a6878f0fdf3babe3f15b71585864e50366b16a34acca175ed023c3d77ce1`.

QDIP-06 corrects that baseline before collecting any timing samples. It creates
a private text control with unchanged packed weights/tokenizer, validates exact
decoder and full original-head configuration equality against the current
conversion plan for the pinned source, and removes the separately optional
coarse-readout setting. The model payload is hardlinked, not duplicated or
re-quantized. Its byte configuration is separately identified above; this is a
new control, not an assertion that the historical qualified byte configuration
was reproduced. The workload, loading order, repetitions and acceptance formula
are unchanged from registration.

### QDIP-06: the image reference graph regresses text performance

Six measured requests per arm, following one excluded warmup per load:

| Metric | Current text control | Image-loaded reference graph |
| --- | --- | --- |
| Useful decode median | 61.091 tok/s | 3.394 tok/s |
| Useful decode range | 60.627–61.483 tok/s | 3.364–3.404 tok/s |
| First-token median | 0.07882 s | 2.86318 s |
| Whole-request wall median | 0.19405 s | 4.92582 s |
| Input / selected context | 16 / 2,048 tokens | 16 / 2,048 tokens |
| Measured output length | 8 tokens | 8 tokens |
| Actual proposed / accepted drafts per request | 5 / 5 | 5 / 5 |
| Greedy output hashes | Identical across all requests and both arms | Identical |

The observed decode loss is **94.44%**, outside the fixed five-percent bound.
Both arms executed real three-draft MTP with the complete original readout.
This is a short unscreened observation on one pinned checkpoint; it is enough
to reject a claim of restored text speed, and does not establish a new
host-qualified family benchmark. It does not measure MTP-off 27B parity.

The release worker SHA-256 is
`e41bc8b07681cacb9c6c66d352dd5805cf64baa2285b8287d27edde4a1867d67`.
The private corrected receipt SHA-256 is
`8692ee0f4c8e540fc0db5401500041b4c7a05dbe73a8cae76b1e0628f299241c`.
It records exact configuration/weight/input/output hashes, loading order,
excluded warmups, all timings and actual draft statistics. It contains no
prompts, responses, token vectors or probabilities.

The next implementation must preserve the fast text arithmetic and supported
verification path while the image tower is resident, or numerically qualify a
replacement that meets the same speed gate. The functional serial verifier
cannot close IMGPAR-03. The historical 51.483 tok/s report uses a different,
longer and host-screened protocol; it is retained unchanged and is not compared
as a matched arm here.
