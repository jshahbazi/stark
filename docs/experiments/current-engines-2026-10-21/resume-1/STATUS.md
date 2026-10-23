# Resume campaign 1 status

Updated 2026-10-22. The active campaign binding is `runtime/current-engine-comparison-2026-10-21/resume-1/config.json`, SHA-256 `9408dc2ef3cd7bcc2c7c565e818924e902c286cfdf8b1eb843e42c8e3e087bdb`. The parent `config.json` remains unchanged as the source-bound configuration for the previous attempt.

Fresh admission completed successfully for all six configurations: optimized Stark, MLX-LM, llama.cpp b11747, TensorFold 1.1.1 with drafting enabled and disabled, and vLLM Metal 0.32.0. The Stark executable is the verified exact-main bundle from source commit `97f15f38201cc0b465a81c45e26a4b5b135d903c`, with executable SHA-256 `1cd922689dca98d41efc0727dab32fe472e6e7d199546130c1e555c8a151e517`. MetalRT exact-model support remains unverified and is recorded as an explicit limitation.

Campaign 1 began timing and completed the first Stark load. Its load and quiet guards passed, with quiet windows of 73 seconds before load and 99 seconds after warmup. The first MLX-LM load was excluded for interference: Codex Renderer reached 196% CPU and total external CPU reached 238%, and duetexpertd briefly reached 100.257% CPU at the measurement/teardown boundary. No pageouts, swapouts, or thermal violations were observed. The owned driver stopped cleanly with SIGTERM (PID 54552); the campaign receipt records `status: interrupted` and `timing_accepted: false`. A subsequent process check found no remaining task-owned runtime processes.

The aggregate receipt is preserved in `campaign-1.json`, and the existing summarizer generated `campaign-1-summary.json`. The summary reports `timing_accepted: false`, `allow_public_table: false`, and `diagnostic_only_reason: timing_or_completeness_gate_failed`. No cells from this attempt may be reused, and no public all-engine table or ranking may be produced from it.

This campaign is retained as diagnostic evidence only. A fresh complete retry awaits the user's ready signal and manual hiding of Codex. The computer-control tool blocks controlling Codex itself. No benchmark worker remains active while waiting.

Campaign 2 is also retained as diagnostic evidence only. Stark and MLX-LM completed their first loads, TensorFold drafting-on completed its first load, and llama.cpp was excluded after a mobile asset daemon deletion caused CPU interference. The owned driver stopped cleanly with SIGTERM (PID 58508). The aggregate receipt is preserved in `campaign-2.json`; `campaign-2-summary.json` reports `timing_accepted: false` and `allow_public_table: false`. No campaign-2 rates were examined or pooled, and no performance claim is made.

Campaign 2 did not change code or guards. The final retry is governed by `RETRY_AMENDMENT.md`: one complete 24-load run after host maintenance, owned cache headroom, and a 120-second contiguous quiet preflight. No automatic retry follows a failure.

## User pause

The user explicitly paused testing on 2026-10-22 during final campaign 3. The owned driver (PID 61917) received SIGTERM and exited through its cleanup handler. The receipt is `status: interrupted`, `timing_accepted: false`; 14 load outcomes were retained (11 passed, 3 excluded). No automatic retry or continued testing is scheduled. The partial campaign is diagnostic evidence only and must not replace public benchmark claims. Runtime environments and borrowed models are retained for an explicitly requested future resume.

## User-directed completed observed run

Following the user's just run it and try again instructions, campaign 4 completed all 24 loads and 528 measurements without quiet-window waiting. All 432 performance samples met fixed work; all 24 loads retained host-interference exclusions. The timing_accepted flag is false for this mode. [RESULTS.md](RESULTS.md) reports observed values, quality sanity outcomes and limitations. The public site remains unchanged. All task-owned workers and sleep assertions are stopped; no automatic retry is scheduled.
