# GLM-5.3-Flash prefill and KDA optimization follow-up, 2026-11-01

Status: **qualification in progress**. This follow-up records component and
full-model evidence. A reproducible improvement in total request time has not
been established.
The completed v39 screened result remains unchanged in the [native optimization
record](GLM53_NATIVE_OPTIMIZATION_2026_11_01.md).

The [aggregate JSON](data/glm53-prefill-kda-2026-11-01.json) contains only
aggregate measurements, hashes and status. Runtime receipts remain ignored.
Prompts, completions, token IDs, logits, probabilities and exception payloads
are not published. This report has no entry in the benchmark index until an
end-to-end result is available.

## Baseline and provenance

The work starts from clean, pushed source revision `9cc4fe90`. Comparisons use
the immutable v39 generator and scorer workers from
`runtime/glm53-optimization-20261101/build-receipt-v39.json`: generator
SHA-256 `0191ff9aff0eac1022a433604ca3d71de929fb557c4945c49201cfdad00db44c`,
scorer SHA-256
`5a4af2f26271a265938eebec8f7a48ed9ae8b7c480cb95b3bc0c6fc0e3a2cc58`.
The v39 screened reference was 34.800 useful tok/s on the 14-token, 512-context,
32-output greedy request; the matched MLX control was 34.382 useful tok/s.
These are historical reference rates for the follow-up. A first combined
queued-prefill/KDA candidate now has a provisional screened run below; further
ablation and qualification remain pending.

## KDA recurrence component

The v40 KDA norm trial is opt-in and default-off via
`STARK_GLM53_KDA_PARALLEL_NORM=1`; admission also requires fused KDA state,
transposed recurrent state and the qualified dimension-128 geometry. The
selected safe-math path parallelizes RMS reductions within the existing KDA
recurrence kernel. Three KDA numerical oracle tests passed.

The same-process paired Safe ABBA component trial measured median recurrence
time per layer at 58.108 microseconds for the serial norm and 50.933
microseconds for the parallel norm, a 12.35% component reduction. It used 16
warm samples, 64 heads and dimension 128. This is a recurrence component
measurement only; it does not include the full model, encoder, other layers,
prefill, or request scheduling and is not an end-to-end throughput claim.

## Queued prefill design contract

Queued prefill has passed implementation fixtures, but v43 keeps it disabled
because same-queue and one-pending policies were within noise; it remains
optional research. Its correctness contract is to encode
the next prompt token while the prior submitted graph runs, keep every resource
used by that graph alive through completion, preserve same-queue dependency
ordering for directly bound tracked resources, and read CPU-visible KDA
failure/status only after all queued writers complete. Apple documents
automatic synchronization of conflicting tracked-resource accesses submitted
to a command queue in [Resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization).
Its [command-structure guide](https://developer.apple.com/documentation/Metal/setting-up-a-command-structure)
describes command-buffer ordering and that queue-created command buffers hold
strong references to their resources. The implementation will use these
documented behaviors only within that same-queue, tracked-resource scope.

The first combined v41 queued-prefill/KDA candidate measured 34.948 useful
tok/s, but four paging exclusions and the lack of a rebuilt scorer left it
provisional. The v43 short candidate uses parallel KDA norm, fused IQ gate/up
and one-pending prefill overlap; queued prefill is off. Against a fresh sealed
v39 control at 34.706 useful tok/s, v43 measured 34.901 useful tok/s and
36.026 all-sampled tok/s; prefill measured 33.047 tok/s. It had four warmups,
four accepted measurements and three paging-excluded attempts. Input-token,
configuration, tokenizer, template and checkpoint identity hashes match the
fresh control, and the decoded-text hashes match. The point estimate is not a
final gain claim. The first fusion-off run failed quiet admission after
unrelated compiler activity; its single measured row is not used. The complete
retry below retains its own exclusions and timing conditions.

The v43 generator is bound to `generator_sha256`
`33e3e9b9af86c339b197efcc93e67a940c50defa7beb7f0fba1777fafa80f951` in
`runtime/glm53-followup-20261101/build-receipt-v43.json`. The scorer was also
rebuilt. Its 128-point next-token scoring comparison against the archived
original F32 worker passed: maximum normalized log-probability difference
`1.4185903996377647e-5`, RMS `5.898910284575336e-6`. This is limited numerical
agreement, not a model-quality evaluation or comparison with an MLX teacher.

The v41 paired-prefill fixture passed five cases, the queued-graph suite passed
four tests with one ignored, and KDA runtime admission and graph tests passed.
These checks cover synchronous fallback, cancellation/reset, draining a second
writer after failure, tracked-resource admission, same-queue state dependencies,
and selection of the safe parallel-norm path. They do not establish end-to-end
quality or performance.

An unscreened v42 phase probe accounted for about 98.6% of warm prefill time
in GPU command execution (about 0.3833 of 0.3889 seconds). This is a ratio of
command timers to phase duration, not a GPU utilization or clock measurement.
It used different activity conditions from quiet-admission throughput samples;
its rates must remain separate. A v43 diagnostic confirmed actual fusion on
41 layers per token: 574 calls during 14 prompt tokens and 1,271 during the
31 decode steps.

The selected overlap/KDA recipe still needs repeated screened end-to-end
comparisons against sealed v39 workers, with identical checkpoint files, input
token IDs, context, decode length, sampling and host admission. Pool
oracles and paired-worker checks must remain separately identified from the
throughput result. The task is pending those runs and final receipts.

## Completed ablation and longer prompt

The complete same-v43-binary fusion-off retry measured 34.973 useful decode
tok/s, 36.101 all-sampled tok/s and 30.769 prefill tok/s. Four warmups and four
accepted measurements were retained; six attempts were excluded for paging.
The output hashes match the fused run. These results do not show a full-model
benefit from fusion, and prefill timing varies materially between these runs.
The experimental wrapper therefore leaves both fusion schedules disabled.
Parallel KDA norm and one-pending prefill remain explicitly selected by that
wrapper; ordinary engine defaults remain unchanged.

The unscreened 96-input-token, eight-output-token run also passed exact native
replay and repeated the sealed-v39 output hash. Prefill was 33.215 versus
33.332 tok/s; useful decode was 31.734 versus 32.832 tok/s; median wall time
was 3.1814 versus 3.1372 seconds. This longer prompt shows no improvement.
It is a separate workload from the 14-token screened comparison.

The rebuilt hybrid suite passed 187 tests with 69 ignored. Five prefill fixture
tests passed, along with both 94-position source-oracle pools and the selected
KDA runtime graph check. The paired generator/scorer fixture passed isolation,
cache, recovery and reference checks; maximum scorer logit error was
`1.33514404296875e-5`. Generator library tests passed 92 with two ignored;
scorer tests passed 21. Both worker Clippy checks and repository policy checks
passed. These are targeted checks; the complete engine suite was not run.

## Other trials

The IQ constant-table correctness oracle passed. The corrected v41 component
receipt measured IQ2 at 232.52 to 229.95 microseconds and IQ3 at 58.91 to
58.61 microseconds; these small shifts were not reliable gains and neither
path was promoted. The initial v40 receipt copied hint labels across schedules
and its timings are unusable. The v42 fused IQ gate/up component measured
145.984 to 131.703 microseconds per 8 experts (9.78% component reduction),
with two correctness tests passed and bitwise-equal gate, up and activation
values. This component result does not establish an integrated-model gain.
The typed-half K-quant oracle passed, but gains were small or inconsistent and
the Q5 output shape regressed; that trial is not promoted, and its trial
kernels are now test-only. A smaller fused tile has a separate default-off
`STARK_GLM53_IQ_FUSED_GATE_UP_R2=1` flag, requiring the parent fusion flag.
An E288 component with scattered IDs measured R4 at 118.957 to 106.807
microseconds and R2 at 119.764 to 105.993 microseconds against their respective
paired baselines. R2's advantage over R4 is small; neither proves an integrated
gain. R2 runtime parity, sticky failure/recovery, parent-flag admission and
queued mmap-owner lifetime checks passed. Full-model R2 timing remains pending.

All changes remain experimental and default-off. No production default or
checkpoint format changes are part of this follow-up.
