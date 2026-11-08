# Model experiment register

This register tracks changes intended to make a model correct, faster or less memory hungry, including unsuccessful attempts. Each experiment needs a stable ID, a model-specific result and a source-bound record. A passing primitive, a successful conversion and a qualified complete request are different evidence levels.

## Recorded work

| Family / profile | Findings and attempt IDs | Latest qualified state |
| --- | --- | --- |
| OLMo 3 7B Instruct BF16 and affine W4/group-64 | [OLMo/Gemma retrospective, OG-001 onward](archive/runs/2026-11-08-olmo3-gemma4-findings.md) | Native text generation/scoring qualified on pinned fixtures. Q4: 14/14 numerical gates, 13/14 full-vocabulary exact cases. Parallel Q/K RMS improved paired request latency 19.46% over C2. Latest RMS reuse adds no material gain. |
| Gemma 4 E2B BF16 and affine W4/group-64 text | [OLMo/Gemma retrospective, OG-001 onward](archive/runs/2026-11-08-olmo3-gemma4-findings.md) | Q4: 14/14 numerical gates and 14/14 full-vocabulary exact cases, plus a fully matching bounded cached trajectory. Parallel Q/K RMS improved paired request latency 3.07% over C2; subsequent source fusions improved paired work rate 2.57% over C3 and reached MLX parity on pp512/tg128. |
| Gemma 4 E2B BF16 images | [Image findings and limitations](archive/runs/2026-11-08-olmo3-gemma4-findings.md#native-support-and-initial-qualification) | Three canonical image cases, worker lifecycle and cold/cached client checks pass. Installed desktop UI execution and image performance remain unmeasured. |
| GLM-5.3-Flash and Qwen3.8-Flash-Next | [Existing hybrid studies](benchmarks/README.md#optimization-records); [transfer assessment](archive/runs/2026-11-08-olmo3-gemma4-findings.md#transfer-from-qwen-and-glm) | Separate studies exist. Their complete experiment histories have not been backfilled into this register. Hybrid performance gains do not qualify a dense-model port. |

The initial backfill covers the entire OLMo/Gemma chat and its linked evidence through engine `5aeaab96` and benchmark publication `294b2286`. It is not a repository-wide inventory of every past model experiment. The dated retrospective preserves superseded outcomes; current conversion and feature scope remains in the [support guide](OLMO3_GEMMA4.md).

## Recording rules

1. Assign an ID before changing code or starting a trial. Continue the `OG-` sequence for this investigation; use a distinct family/investigation prefix for other work. Never reuse an ID. Record a materially different candidate or reference protocol separately.
2. State the hypothesis and exact model/checkpoint, source revision, quantization, activation/accumulator/KV precision, device and relevant geometry. Separate each affected model's result, even when the implementation is shared.
3. Record the baseline and candidate source revisions or complete patch hashes, executable hashes, reference/toolchain locks, fixture hashes, acceptance gates and timing boundary. Bind private evidence by hash; keep its contents ignored.
4. Record correctness before claiming speed. Test the relevant conversion, prefill, populated-cache decode, longer trajectory, worker recovery and image path. State what was not tested. If calibration or the oracle changes, explain why and preserve the earlier failure under its original protocol.
5. Publish the full accepted comparison, including unsuccessful blocks and rejection reasons. Freeze all arms and serialize GPU jobs. Use balanced independent loads and paired statistics; distinguish kernel timing, screens and qualified end-to-end results.
6. Record the disposition even when no code ships. Keep failed or contaminated attempts, reverted patches and superseded source-bound records. Document the evidence required before retrying an unsuccessful idea.
7. Update this register and the relevant support/benchmark indexes when a result changes the current qualified state. Keep dated validation under `archive/runs/` and measured performance reports under `benchmarks/`.

Use these dispositions consistently:

| Disposition | Meaning |
| --- | --- |
| Retained | Shipped to the branch after the recorded qualification. State whether the evidence establishes correctness, performance or both. |
| Rejected | Tested and failed a stated gate, or the investigation disproved its premise. |
| Not promoted | A candidate/control was tried, but its benefit or coverage did not justify shipping. |
| Invalid run | The run cannot support its intended claim because identity, reference, harness or host conditions failed. This does not by itself reject the implementation. |
| Investigated | Source/design review only; no native end-to-end result. |
| Deferred | Proposed work remains unimplemented or unqualified. |

An aggregate gain for several changes must stay attributed to that bundle until an ablation measures the individual changes. A model can pass correctness and show no material performance gain.

## New experiment template

Copy this section into a dated record and link it from the register. Use one result row per model/profile. For a long investigation, a stable-ID table may summarize attempts, with detailed evidence in linked records.

```markdown
## <ID>: <specific change>

Date / branch / owner:
Hypothesis:
Disposition: <retained / rejected / not promoted / invalid run / investigated / deferred>

### Scope and identity

- Model repository and immutable revision:
- Native artifact/config/conversion hashes:
- Weights / activations / accumulators / KV precision:
- Device, OS, attention geometry and context limits:
- Baseline source and executable hashes:
- Candidate source revision or complete patch hash; executable hashes:
- Reference revision, dependency lock and toolchain:
- Fixture/protocol hashes and private evidence location:
- Upstream methods, source revision, license and adaptation scope:

### Change and controls

Exact implementation change and expected mechanism:
Correctness gates fixed before testing:
Control/negative-control cases and excluded paths:
Timing boundary, sampling, warmups, loads, order and host guards:

### Results

| Model / profile | Correctness and coverage | Performance and uncertainty | Disposition |
| --- | --- | --- | --- |
| ... | ... | ... | ... |

Accepted and rejected run counts; rejection reasons:
Aggregate receipt/report links and hashes:
Unmeasured cases and limits:

### Decision

Why retained/reverted/deferred:
What evidence would justify retrying:
Supersedes / superseded by:
```

Never include private prompts, responses, token vectors, logits/probabilities, credentials or exception dumps. Publish aggregate counts/errors/timings and hashes only. Experiments do not authorize changing the runtime's serialization or consecutive-cache contract.
