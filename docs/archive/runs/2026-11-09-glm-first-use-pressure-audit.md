# 20261109-GLMMEM-13: first-use pressure source audit

State: source/header/receipt audit and one focused diagnostic proposal only.
No new engine build, Metal execution or full-model run is granted by this record.
Predecessors: [GLMVIS-10](2026-11-09-glm-functional-pageout-screen.md) and
[GLMMEM-11](2026-11-09-glm-mapping-lifecycle-diagnostic.md).

## Actual failure boundary

The functional worker reached readiness in 0.354363667 seconds. The next
ordinary-text request did not complete. No internal layer, library, pipeline
or dispatch marker was captured, so the failure cannot be placed within those
steps from its receipt.

| Wired observation | Bytes |
| --- | ---: |
| Host baseline | 4,960,813,056 |
| Host at exact compressor breach | 6,690,865,152 |
| Host growth at breach | 1,730,052,096 |
| Maximum sampled host wired | 9,055,518,720 |
| Maximum sampled host wired growth | 4,094,705,664 |
| Worker `proc_pid_rusage` wired at breach / maximum | 0 / 0 |

The worker's wired field is not a complete GPU/driver physical inventory.
Compiler-service peak memory was not captured. Neither the later small service
RSS nor `mincore` backing residency attributes the real pressure to a compiler,
copy-on-write, file mapping or driver allocation. The small blit-only control
did not exercise native shader library/pipeline creation.

The initial pressure audit receipt is
`75d663f0fdc21a4468453eb469c348362723d484c698b5f613a39fd5ab787ea1`.
Its Q8-only next-control proposal is superseded by the first-use audit below.
The mere presence of 346 Q8 tensors did not establish their execution order.

## Source-grounded first-use order

Corrected audit receipt:
`17f02cba7e625973f8d346b173c1213e0ed76e75a8054c5ddb567467cd396989`.
It pins the actual GLM/weight graph source and four GLM shader files, and reads
only the 42,155-byte second-shard header for the first projection identity.

1. Before readiness, `weight_metal::initialize` creates its safe-math source
   library and the Packed(0) reduction pipeline. Other format pipelines are lazy.
2. On first advance, `MetalDecode::new` allocates/uploads controls and attention
   state. `Control::new` does not itself initialize the separate GLM executor.
3. Layer zero's first `Control::pre` encodes the weight-library RMS family, then
   `glm_control_project` calls `glm_metal::executor` for the first time.
4. That executor creates its base and transposed KDA libraries, two KDA PSOs,
   35 required fixed-name graph PSOs and two additional fused-control PSO
   attempts. The transposed library is compiled even with fused-state opt-in
   unset. These calls precede the first trunk packed-weight projection.
5. Control mixing/input/normalization and KDA initialization follow. The first
   packed projection is `blk.0.attn_q.weight`, kind 13 (Q5_K), GGUF dimensions
   `[4096,8192]`, runtime rows 8,192 / columns 4,096. With the experimental
   selectors cleared, it uses `hybrid_packed_block_matvec`.

The packed pipeline constants describe format/block sizes or affine layout,
plus fixed schedule choices. Actual rows, columns and context are runtime
arguments. GLM graph PSOs use fixed function names. This provides no evidence
of an unbounded per-dimension specialization cache. Format/library compilation
can still have an unmeasured peak. The old receipt did not seal inherited
`STARK_HYBRID_ORDERED_RMS`, so its exact first RMS function name is conditional;
either RMS variant precedes the separate GLM initialization.

## One proposed small discriminator

Use one new ignored, phase-instrumented control with the existing native
`Control::pre` and only the actual first Q5_K block kernel. A synthetic Control
with hidden 256, four streams and 20 iterations uses about 96 KiB of weights.
The projection has 13 rows by 256 columns and 2,288 packed bytes. No GGUF/model
file is mapped. Both compilation and data geometry remain bounded.

Record readiness-equivalent weight-library/Packed0 completion, Control buffer
allocation, before/after Control encoding and separate GLM library/PSO creation,
Q5 pipeline/dispatch encoding, first submit completion, bounded warm repeats
and owner exit/drain. Encode Control and Q5 in the same graph before submitting
so the first-use ordering is preserved. Check finite/nonempty output and an
independent decoded-byte F64 Q5 projection oracle.

The proposed supervisor must timestamp host counters, worker usage, Metal
allocated bytes and **all** `MTLCompilerService` PID/executable/RSS totals,
peaks and newly observed PIDs throughout cleanup. It must never kill a service.
Proposed narrow ceilings are 1 GiB each for worker RSS, footprint, host wired
growth and service-RSS growth; compression growth below 256 MiB; 64 GiB entry
and 32 GiB runtime available floors; zero swap; and a 60-second deadline.
These limits and post-drain service monitoring require independent review.

Pressure before pipeline/dispatch completion in this no-file control would
narrow the affected stage without a full-model load. A clean result covers
only this first-use path. Persistent OS shader-cache state is unknown, and
the test build's additional shader fixtures need separate source/binary seals.
Other formats and later operators remain outside this control. No format
sweep, larger input, retry or full-model promotion is authorized.
