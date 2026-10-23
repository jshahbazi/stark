# Final retry amendment

This amendment is frozen before the third timing attempt for the 2026-10-21 current-engine comparison resume. It records a deviation from the original two-campaign cap because campaigns 1 and 2 were independently invalidated by host maintenance and interference, after manual Codex hiding where applicable. No rates from either campaign were examined, reused, or pooled.

The final retry uses the same six configurations, source commit, executable and model hashes, dependency locks, fixture set, input counts, cache controls, and CPU, GPU, thermal, sleep, and pageout/swapout guards already recorded in the resume protocol. It consists of one complete run of all 24 planned loads: four loads for each of six configurations. The whole-campaign acceptance gate remains mandatory.

Before starting, root must recover sufficient owned cache headroom and establish 120 contiguous quiet seconds. The preflight timeout is 600 seconds. The same per-process and total CPU limits, known-model CPU limit, device GPU limit, nominal thermal requirement, monitor-gap rule, sleep checks, fixed-work checks, cleanup requirements, and wall/monotonic discrepancy gate remain in force. No gate is relaxed.

Campaign 2 evidence is preserved in `campaign-2.json` and `campaign-2-summary.json`. Stark and MLX-LM passed their first loads, TensorFold drafting-on passed its first load, and llama.cpp was excluded after mobile asset daemon interference. These are diagnostic observations only; no performance claim follows from them.

If the final retry fails or is invalidated, there are zero further automatic retries. Preserve its full receipt and summary, publish no replacement all-engine table, and report the failure disposition. This amendment authorizes continuing the user-requested clean run once the preflight conditions are met; it does not authorize model or GPU work outside that run.
