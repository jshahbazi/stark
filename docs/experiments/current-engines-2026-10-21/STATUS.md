# Fresh comparison status

## Admission

All six configurations completed the untimed admission pass: optimized Stark, pinned current-comparison MLX-LM, llama.cpp b11747, TensorFold 1.1.1 with and without drafting, and vLLM Metal 0.31.0. All six performance fixtures per configuration met their output limits. The admission observations are not comparative performance evidence.

The Stark executable and both Qwen source/repack model hashes match the newer comparison's retained provenance. The downloaded llama.cpp and TensorFold release archives/executables and borrowed GGUF also match the prior pinned cohort. New Python environments were installed from unchanged locks in a task-owned runtime. No engine/model code was patched.

## Timed attempt 1: interrupted, not accepted

The first attempt acquired four loads in block zero before interruption. Optimized Stark passed its load's guards. MLX-LM, llama.cpp and TensorFold default were excluded for external CPU interference. Recorded peaks included Docker Desktop's `com.apple.Virtualization.VirtualMachine` above 103% CPU, `Codex (Renderer)` above 201%, and `PerfPowerServices` above 141%. Their measurement intervals had no pageout/swapout increase or thermal violation; those facts do not override the CPU exclusions.

The computer-control tool blocked access to the Codex application, so its window could not be hidden automatically. No alternative control path was used. The benchmark was stopped through its SIGTERM handler, and its owned native workers were confirmed absent. Unrelated Docker, application and model-service processes were left running.

The aggregate report is retained in `interrupted-attempt-1.json`; `timing_accepted` is false and `interrupted-attempt-1-summary.json` has `allow_public_table: false`. No cells from this attempt may replace website values or be pooled with a later attempt. The website remains on the newer original Stark/MLX-LM result; the historical multi-engine marketing presentation was withdrawn.

The one permitted full retry is pending a quieter host and the requested manual pause of Docker work/hiding of Codex. No inference that current Stark beats the remaining engines is established by this interrupted attempt.
