# Flash Next prefill grid on Pi coding sessions

**20261117-FLASHSPEED-06 · November 17, 2026 · Default decision study**

The [bounded native screen](2026-11-17-flashnext-speed-experiments.md) found that 2,048-token chunks can preserve more of a growing prompt than the current 4,096-token chunks. This study tests whether that improves actual coding-agent work with MTP enabled on the pinned Lily Q4-XL artifact.

## Prespecified comparison

Use the installed Pi CLI with its Stark stream guard and the real Stark Desktop HTTP backend and Rust/Metal worker. Keep one worker resident. Give Pi a fresh profile and task checkout per session. Use the four fixed three-turn synthetic Python tasks, with new fixtures for both launches. Run each task twice per grid in opposite order, eight pairs and sixteen sessions. Keep the image tower and draft head loaded, use text requests, cap each completion at 1,024 tokens and fix the worker context to 40,960. Request two exact drafts using Stark's MTP setting. Use temperature 0.6, top-k 20, top-p 0.95 and seed 42 for each request.

Pi uses its file tools in each fixture and an independent grader evaluates behavior after every user turn. The experiment uses the local Stark provider and serial sessions. Record native first-token time, reused input, actual verifier work, worker wall time and Pi's full turn time separately. Clear the prefix before each fresh session. Predict each subsequent hit from the actual rendered input IDs, immediately previous native request, context and scope. Persist timing, counts, check outcomes and digests.

For promotion on this pinned generation route, require exact cache hit accounting, positive MTP work for sustained outputs, no new acceptance failures among baseline passes, and at least six successful pairs. Require at least 10% median paired improvement on eligible follow-up native prefill with six faster pairs where the sessions have comparable work, at least 5% median total accepted Pi turn-time improvement, and at most a 5% cold-prefill regression under matched work. Keep unpaired observations and the reasons they cannot prove a gain. Treat the four tasks as workload coverage and their repeated runs as order controls. Shared worker launches and repeated fixtures do not produce independent population samples. If observed host variation or different trajectories prevent matched cold/warm comparisons, leave the default decision unqualified. An actual production switch also needs lifecycle and image regressions and a separately verified artifact-scoped selector.

The initial 4,096-grid pilot reached the 1,024-token cap during a source edit and ended with a length error. The other first-turn pilot also lacked a clean completion. Both had real native requests and correct MTP/cache acknowledgements. The scored run starts all eight pairs fresh with a 2,048-token completion budget on both grids. It records the Pi finish reason and independently grades completed file writes even if the client ends with a length or API error. Both pilot cases are setup evidence.

A second setup pass exposed another comparison issue. Pi renders its working directory into the system prompt, so differing sandbox paths would change input tokens even for identical tasks. The scored run presents fresh fixtures at the same path and archives each workspace after Pi exits. The task instructions reflect the available file tools and independent grader. A model-free guarded streaming preflight exercised the actual installed Pi write, edit and read tools under the same sandbox. All three succeeded and the expected file contents were verified. The second setup pass is excluded from scoring.

In an initial fixed-path run, Pi continued reading and editing the same ledger after the first implementation already passed its checks. The harness's native request cap raised an exception inside the worker call. Stark closed the worker, invalidating the remaining user turns. That run is a setup receipt. The final scored schedule uses only read, edit and write, asks for one read and one edit per turn, and cancels through Pi at eight tool completions. It records capped turns as failures. Each grid receives the same task and limit.

## Execution identity

The scored run uses Pi 1.5.2, Node.js 26.13.0 and Stark's guarded streaming extension. The real source Desktop HTTP backend renders and parses Pi's tool calls. An ephemeral authenticated loopback server runs for each worker launch. Pi's profile and fixtures are disposable. The operating-system sandbox permits file writes to those two paths and allows network connections to the benchmark server's port. The detached Python grader uses another sandbox, discards candidate stdout/stderr, and runs cumulative fixtures independently of the agent. Its reference implementations passed all 12 task/phase calibration checks.

The native executable is the same Rust/Metal release worker from the [kernel study](2026-11-17-flashnext-speed-experiments.md), SHA-256 `bdf851bedf895ab09859cbdc40cab583a9976ddaf06f01d98ad3872dcd8f7dbe`. The same pinned Lily Q4-XL artifact, `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, loads its vision tower and MTP head. The host is an M5 Max with 128 GiB on AC, macOS 26.7.2 (25G332). Host counters observe system-wide paging during each session. They do not measure GPU clocks. Timing excludes the fixed startup warmup and reports actual Pi turn duration separately from session duration, which includes independent grading and Pi lifecycle.

The measured scope is text-only Pi RPC with its three file tools, synthetic Python projects and the pinned Lily checkpoint. The same source model's vision tower is resident during those requests.

## Results

**Keep 4,096 as the default.** The 2,048 grid has a repeatable advantage on growing Pi prompts. The full-session acceptance sample and cold-prompt coverage fall short of the prespecified promotion gate. The implementation remains available as an experimental selector for the pinned worker.

All eight counterbalanced pairs completed, for 16 actual Pi sessions, 48 user turns and 152 native requests. Every request performed positive two-draft MTP work. For each of the 76 paired native requests, the ordered input and emitted-token digests, output cap, finish, token counts and all four MTP work counters matched. All 24 paired user-turn grades and both grids' executed tool counts matched. The expected cache boundary agreed on every request.

| Measure | 4,096 grid | 2,048 grid |
| --- | ---: | ---: |
| Fully accepted three-turn sessions | 2/8 | 2/8 |
| Independently accepted user turns | 14/24 | 14/24 |
| Cumulative behavior checks passed | 222/252 | 222/252 |
| Native requests with cache hits | 2/76 | 40/76 |
| Input tokens reused | 8,192 | 86,016 |
| Input tokens processed fresh | 185,516 | 107,692 |
| Sum of native first-token times | 116.53 s | 78.54 s |
| Sum of actual Pi turn times | 250.65 s | 220.82 s |

- **Growing prompts.** In 38 exact-work requests, 2,048 reused an additional 2,048 tokens. It reached the first native token sooner in all 38. The median request reduction was 57.1%, and the median paired reduction after summing these requests per session was 54.5%, with eight session-level wins.
- **Pi wall time.** Summing the three turns per session, 2,048 reduced median paired time by 12.1% across all eight pairs of identical work, with seven wins. Among the 14 pairs of accepted user turns, the median reduction was 5.0%, with ten wins. Those accepted turns included eight first turns, where the median change was +0.8%, and six follow-ups, where 2,048 won every comparison. Both fully accepted pairs used the catalog task and finished 6.5% and 23.4% sooner.
- **Fresh prompts and host.** The 36 exact-work requests fresh in both arms ranged from 1,491 to 2,205 input tokens. Request-level median first-token time was 5.6% higher on 2,048. Summing fresh requests within each session gave a 0.2% median change and four wins per grid. There were zero matched fresh requests at 4,096 or longer. All prompts ranged from 1,491 to 5,103 tokens. System-wide paging was visible in both arms, including 134,460 swapout pages during candidate sessions. The host measurements cannot establish a hardware-controlled cold difference.

The ledger, schedule and inventory tasks missed the same late requirements on both grids. Passing catalog twice provides two accepted pairs, short of the required six. The strongest finding is faster repeated prefill under identical coding-agent work. The default decision needs longer cold controls, additional independently accepted task families, and the artifact-scoped lifecycle checks described above.

The later [fresh 8K/12K/32K follow-up](2026-11-17-flashnext-fresh-coding-grid.md) filled the long-cold timing gap for those exact lengths. With identical outputs and MTP work, 2,048 was slower on all 18 pairs and exceeded the 5% cold criterion at all three lengths.

[Benchmark runner](../archive/experiments/2026-11-17-flashnext-agent-ab/benchmark.py) · [Fixture and independent grader](../archive/experiments/2026-11-17-flashnext-agent-ab/fixtures.py) · [Aggregate results and source hashes](../benchmarks/data/flashnext-coding-agent-grid-2026-11-17.json)
