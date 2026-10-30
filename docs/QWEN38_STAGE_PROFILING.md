# Qwen3.8 stage profiling

Set `STARK_UZU_STAGE_PROFILE_OUTPUT` to a new private file path before starting a worker to collect bounded stage timings for the standard Uzu Qwen3.8 route. Profiling is off by default.

The Metal backend creates the file with `create_new` and mode `0600`. It never overwrites an existing file. It profiles at most the first four single-token Qwen passes in that process; batched prefill does not activate profiling. A fresh worker and a fresh path are required for another sample. A single-token prefill suffix is eligible, so use a prompt whose prefill batches contain multiple tokens when collecting decode-only samples.

Each completed profiled command contributes one JSONL record containing fixed stage names, call counts, summed GPU time, CPU encoding time, and whole-command GPU time. Stages cover embedding, HC read/inject, GDN, QSA, MoE, PLE, final HC, head and standard sampling. Records contain no model inputs, generated output, token IDs, logits, decision probabilities, or exception text. Runtime artifacts remain private and ignored.

Profiling uses a bounded Metal 4 timestamp heap with precise checkpoints. The heap and fixed category metadata remain owned through command completion, and counters are resolved only after the existing GPU wait succeeds. An extra completion owner retains the heap if a pending command is abandoned. Unsupported timers and profiler setup, resolution or output failures disable further profiling without selecting another inference backend.

Precise checkpoints can split encoders and change scheduling. The first profiled pass can also include CPU pipeline creation. Use the resulting stage ranking to choose optimization experiments. Compare profiled and unprofiled wall time, and qualify performance separately with profiling disabled. Profiled throughput is diagnostic and must not be reported as production throughput.

The existing whole-command GPU duration comes from Metal commit feedback. CPU encoding timings exclude the timestamp calls themselves. Stage totals do not include work after the last checkpoint; whole-command time includes that tail. Pure aggregation tests cover category totals, malformed and nonmonotonic timestamps, missing samples, and unavailable frequency. GPU timing support and measurement overhead require a controlled run on the target Mac.
