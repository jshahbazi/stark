# Experimental GLM register KDA prefill

`STARK_GLM53_KDA_REGISTER_PREFILL=1` opts the native GLM chunk-prefill path
into a multi-token KDA operator. It uses the same loaded GGUF tensors and keeps
activations, convolution histories, recurrent state, accumulators and outputs
in F32. There is no BF16 cast or new persistent cache. Decode retains its
existing one-token implementation.

Admission requires a 128-channel KDA head, one to 128 input tokens, a compiled
Safe-math Metal pipeline with SIMD width 32, sufficient threadgroup capacity,
and a valid bounded row plan. Other head dimensions and unsupported pipelines
use the existing chronological token loop. The row plan covers every value
row once, carries at most 128 rows and two heads per group, and adjusts to the
pipeline's thread limit. With 64 heads, 40 reported GPU cores and a thread
capacity of at least 832, it produces 80 groups of at most 104 rows. The plan
does not change an individual row's arithmetic. Opt-in aggregate profiling
records `glm.kda.prefill.register.tokens`, `.groups`, and `.rows`, so a
qualification run can verify that the route was admitted and record its plan.

The operator encodes three kernels and a sticky-status bridge:

1. Preparation convolves each token against the incoming history and earlier
   tokens in the same chunk. It retains chronological convolution sums and
   the existing selected q/k norm policy. It commits each convolution history
   once after all outputs are prepared.
2. Eight lanes share a value row. Each lane owns sixteen F32 state elements in
   local `float4` values across the entire submitted token span. Eight-token
   staging tiles have bounded tails. State is loaded once and stored once per
   chunk. Compiler register allocation and performance require measurement on
   the actual target; source-level local storage alone does not prove an
   absence of spills.
3. The finish kernel retains the existing selected per-head RMS policy and
   applies the existing norm and output gate.

The recurrent prediction and output dot products pass a running F32 sum
through eight segments. The owning lane applies its sixteen keys in order,
then broadcasts the running sum to the subgroup. Eight broadcasts per dot
retain all 128 chronological additions. Decay, delta and state update use the
existing source expressions, including its runtime dimension-dependent output
scale. This restores the source arithmetic schedule while retaining local
state and the lower state traffic. It introduces a serial arithmetic
dependency; its performance requires measurement.

The new path uses the existing `parallel_norm_admitted` decision for each KDA
instance. When selected, it compiles the exact existing norm helper: four
SIMD partial sums followed by their SIMD sum, with chronological fallback
when any value is nonfinite or exceeds magnitude `1e15`. A risky q or k
selects the ordered fallback for both; output RMS has its own fallback.
Otherwise, preparation and finish use the existing ordered sums. There is no
new norm selector or independent environment interpretation.

Source-bound synthetic trajectories now require bit equality for output,
convolution history and recurrent state against the corresponding existing
GPU operator, including ordered/parallel norm policies, both backing layouts,
and finite extreme vectors. Those checks and full-model readout/generation
qualification must pass before admitting a default or claiming an integrated
gain. Restoring source expressions is not a substitute for executing these
checks. The earlier XOR-tree candidate failed a 513-token readout comparison;
its diagnostic results and source hashes are retained in the
[declined-candidate record](../archive/runs/2026-11-03-glm-kda-segment-tree-declined.md).

Both `[head,value,key]` and `[head,key,value]` persistent state layouts retain
their existing physical representation. Prefix snapshots therefore preserve
their parameter/layout identity and exact F32 bits. A graph never clears
shared status from the CPU. Cancellation before submission drops an unwritten
graph; submitted graphs retain every bound resource until their draining
owner completes, before model reset or snapshot. Errors remain sticky until
exclusive model reset.

Additional graph scratch is exactly `tokens * (5 * inner + heads)` F32 values:
prepared q/k/v/decay/beta and raw recurrent outputs. Fixed recurrence
threadgroup scratch is 28,736 bytes. Persistent state and snapshot size do not
grow.

The synthetic tests in `glm_kda_prefill_tests.rs` cover nonzero incoming
history/state, convolution widths 2/4/32, both layouts, selected norm policies,
finite extreme inputs, one-token and tile tails, 128-token chunks, continued
chunks, exact restore and 8+1 continuation within the new path,
unsubmitted and submitted graph ownership, and nonfinite controls whose
effect a gate might otherwise mask. Recorded validation results belong in a
dated run record; the existence of these tests is not evidence they passed.

## Source provenance

Algorithmic references were inspected at the immutable mlx-stream revision
[`ac3b772b61bce66450862bca697dbb742c35453f`](https://github.com/davidtai/mlx-stream/tree/ac3b772b61bce66450862bca697dbb742c35453f):

- [`src/arch/glm5_next/kda_percore.zig`](https://github.com/davidtai/mlx-stream/blob/ac3b772b61bce66450862bca697dbb742c35453f/src/arch/glm5_next/kda_percore.zig)
  describes the row plan and launch admission. mlx-stream's host code is
  Copyright (c) 2026 David Tai under the MIT License, with its repository
  [license](https://github.com/davidtai/mlx-stream/blob/ac3b772b61bce66450862bca697dbb742c35453f/LICENSE)
  and [notice](https://github.com/davidtai/mlx-stream/blob/ac3b772b61bce66450862bca697dbb742c35453f/NOTICE).
- [`src/arch/glm5_next/kernels/kda_percore.metal`](https://github.com/davidtai/mlx-stream/blob/ac3b772b61bce66450862bca697dbb742c35453f/src/arch/glm5_next/kernels/kda_percore.metal)
  credits oMLX 0.7.0, `omlx/patches/glm53_kda_recurrence.py`, Copyright oMLX
  contributors, Apache License 2.0, via mlx-serve branch
  `glm5next/kda-prefill-40c` revision `b07dc2f5`.

Stark's implementation is independently written against its existing scalar
contract. It does not copy the generated upstream Metal text, MLX transport,
or BF16 input policy. The shared algorithmic choices are segmented register
state, causal token ordering, bounded staging, and a row plan near one or two
threadgroups per GPU core. Upstream attribution remains in the host and
kernel source headers.
