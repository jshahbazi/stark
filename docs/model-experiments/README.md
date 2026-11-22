# Model change and experiment ledger

[Look up research and speed changes by model](../MODEL_RESEARCH.md), including benchmark comparisons and branch-only attempts. The [source index](source-index.md) groups individual reports by family and topic.

- [20261121-INKDECODE-01–10 · Inkling 2 Mini exact decode toward 60 tok/s](2026-11-21-inkling-mini-decode-60.md) measures exact short verifier blocks, native DSpark depth, vector NVFP4 singleton decode, SIMD expert selection, short-run expert weight sharing and lookup eligibility for the original checkpoint on M5 Max.

- [20261121-INKPROFILE-04 · Default exact one-row router for Inkling 2 Mini](2026-11-21-inkling-mini-decode-default.md) records the flags-unset selection, pinned parity and paired decode comparison against the original source router.

- [20261121-INKPREFILL-14 · Default fast prefill for Inkling 2 Mini](2026-11-21-inkling-mini-prefill-default.md) records default kernel selection, flags-unset parity and the cold 4K worker comparison against the prior opt-in configuration.

- [20261121-INKPREFILL-11–13 · Inkling 2 Mini tensor MXFP8 prefill](2026-11-21-inkling-mini-mxfp8-tensor.md) reaches 1,058 greedy and 1,054 seeded sampled tok/s at cold 4K in eight exact worker pairs per mode. The compensated tensor path passed direct BF16 audit in the opt-in study. The wider NVFP4 tile with lane-scale sharing was rejected.

- [20261121-INKPREFILL-10 · Inkling 2 Mini wider NVFP4 run tiles](2026-11-21-inkling-mini-nvfp4-run-tiles.md) compares exact grouped matrix tile sizes and MXFP8 fusion on the stock 4K schedule, and advances the 64 × 32 tile to worker timing.

- [20261121-INKPROFILE-03 · Inkling 2 Mini NVFP4 guard and singleton router](2026-11-21-inkling-mini-nvfp4-router.md) measures occupancy and replay on stock expert routes, then screens shared scale metadata and an exact one-row router against the original worker.

- [20261121-INKPROFILE-02 · Inkling 2 Mini stock 4K prefill and singleton decode](2026-11-21-inkling-mini-stock-profile.md) measures source-aligned prefill by stage, isolates the singleton router projection cost from the same 4K cache, and selects the next controlled kernel screens.

- [20261121-INKPREFILL-06–09 · Inkling 2 Mini packed prefill](2026-11-21-inkling-mini-packed-prefill.md) measures contiguous activations, eight-route MXFP8, and the source-shaped Metal graph and numerical gates on the M5 Max.

- [20261121-INKPREFILL-05 · Inkling 2 Mini 4K prefill budget](2026-11-21-inkling-mini-prefill-budget.md) measures the 128-row bottlenecks, tests wider groups and several matrix and staging changes, and qualifies batched controller RMS with direct expert bindings.

- [20261121-INKPREFILL-04 · Faster native Inkling 2 Mini prefill](2026-11-21-inkling-mini-faster-prefill.md) records the M5 Max move to 128-row commands, batched controllers, grouped MXFP8, eight-row BF16 weight reuse and partial-tile expert routing.

- [20261121-INKMATCH-01 · Inkling 2 Mini matched 512-row prefill](2026-11-21-inkling-mini-matched-prefill.md) locates the first full-model mismatch and verifies source-shaped dense projections, local attention and router logits at the first visit.

- [20261121-INKPROFILE-01 · Inkling 2 Mini 4K prefill stage profile](2026-11-21-inkling-mini-prefill-profile.md) attributes M5 Max GPU time on the grouped prefill graph. The instrumented and uninstrumented paths produce bit-exact continuation logits.

- [20261121-INKLING-14 and 20261121-INKLINGGROUP-02 · Inkling 2 Mini grouped original NVFP4](2026-11-21-inkling-mini-grouped-nvfp4.md) preserve BF16 expert boundaries, native routing and stable GPU buckets at 64 and 128 rows. The grouped component also passes the speed gate with all 128 experts active.

- [20261121-INKLING-13 · Original Inkling 2 Mini in the native worker](2026-11-21-inkling-mini-worker.md) exercises prepared-artifact loading, streamed chat, images after DSpark-enabled text, and an API tool-result round trip.

- [20261121-INKFULL-01 and INKPREFILL-01/02 · Inkling 2 Mini full target and prefill](2026-11-21-inkling-mini-full-target.md) measure complete serial target stages, pinned traces and exact 4K native parity with the 64-row grouped path.

- [20261121-INKLING-12 · Inkling 2 Mini original DSpark](2026-11-21-inkling-mini-dspark.md) verifies the target's serial selection and draft-state restoration, and tests first-mask proposal fidelity against the original draft head.

- [20261121-INKLINGVISION-01 · Inkling 2 Mini native vision tower](2026-11-21-inkling-mini-vision.md) passes both real-weight image projection controls against the pinned Ollama runtime on the M5 Max.

- [20261121-INKLING-07 · Inkling 2 Mini short-prefix projection](2026-11-21-inkling-mini-short-projection.md) isolates one BF16 gate-projection word, passes connected short and 4K parity, and clears the paired M5 Max budget regression gate.

- [20261121-INKLING-04–06 · Inkling 2 Mini live Ollama parity](2026-11-21-inkling-mini-pinned-oracle.md) fixes the first BF16 mismatch and passes the two-visit 4K connected gate against the pinned runtime. One short global expert row remains outside tolerance.

- [20261121-INKLING-01–03 · Inkling 2 Mini native decode slice](2026-11-21-inkling-mini-native-slice.md) records a 27.35 ms 44-visit 4K budget, exact-ingress NVFP4 parity and a connected numerical failure after BF16 ingress differences.

- [20261120-INKLING-01–04 · Inkling 2 Mini first milestone](2026-11-20-inkling-mini-milestone-1.md) measures the original and Unsloth image-loaded serial references. On the M5 Max, all 26 delayed-scale NVFP4 projection cases passed. The 26.65 ms weighted stage estimate supports a bounded native serial prototype.

- [20261120-LILYSCHED-01: Lily Q4-XL default MTP schedule](2026-11-20-lily-mtp-schedule-default.md): on the pinned image-loaded Lily Q4-XL artifact only, an MTP-enabled text request that names no schedule now runs the adaptive policy with prompt lookup; `"mtp_policy": "fixed"`, `"mtp_lookup": false` and `STARK_LILY_MTP_ADAPTIVE=0` opt out, the worker's `default_drafts` stays zero and the prefix-reuse key is unchanged. On the real model the default path equals the previous worker with explicit selectors hash for hash, the opt-out and the kill switch equal the previous fixed schedule, all eight frozen baseline arms reproduce with the kill switch, and the prefix validator passes 77 of 77 checks. Seeded sampled output of such requests changes; greedy output does not. The app requests two drafts, so its default is the adaptive schedule at a limit of two. Uncommitted and not installed at registration; the app end to end and memory were not exercised. Follow-up, 2026-11-20: on 24 natural requests through the chat template (coding, planning, prose; greedy and sampled) the default is level with two fixed drafts, 0.989 to 1.017 of it in every cell and 1.15 to 1.39 times MTP off, so it holds where the two earlier default changes lost and gains nothing there.

- [20261119-LILYADAPT-01: Lily MTP schedule, matched-window gate for a default](2026-11-19-lily-mtp-schedule-default-gate.md): 648 requests in fourteen sessions on prose and code, greedy and sampled, at 4,096, 16,384 and 32,768 input tokens, each arm on the same windows and seeds, judged by a rule written before the results. Geometric-mean rate against MTP off: adaptive with lookup 1.218, adaptive 1.215, one fixed draft 1.193, two fixed drafts 1.155. Two fixed drafts, the app's default until now, are slower than MTP off on prose at 4,096 tokens (0.956 greedy, 0.952 sampled). Adaptive with lookup is selected; at the app's limit of two it scores 1.193 against 1.133 on its own eight cells. A supplement of sampled long-context windows, registered before its results, changed the outcome from "no change". One artifact, one unscreened host, raw continuation, one sampled setting.

- [20261119-LILYPREFIX-01: Lily MTP scheduling selectors under generation prefix reuse](2026-11-19-lily-mtp-selector-prefix-reuse.md): the model-bound prefix validator gains opt-in selector cases and, for a build whose default schedule is not fixed, default-schedule and opt-out cases. On the pinned Lily Q4-XL artifact every prefix hit equals the same request without reuse, reuse counts follow the existing rule and the schedule is not part of the key: 67 of 67 checks with explicit selectors on the previous worker, 77 of 77 on the new-default worker, six of six invalidation paths each time. 8,192 context tokens, 16-token outputs, two drafts; midpoint restores past 6,144 tokens and longer outputs are not covered.

- [20261119-LILYMERGE-01: merged Lily MTP changes, quiet before/after timing](2026-11-19-lily-mtp-merged-quiet-timing.md): GDN replay rollback, the host round-trip changes, opt-in adaptive depth and prompt lookup, and the fixed-corpus benchmark measured together on one uncommitted working tree against the preserved base worker, in base, candidate, candidate, base order with no other project process running. Every default arm reproduces the base output hash; MTP-off is unchanged (75.5 against 75.6 greedy, 76.4 against 76.1 sampled) and MTP arms are 11.3 to 18.3 percent faster with non-overlapping ranges (greedy two drafts 105.9 against 90.3 tok/s, sampled two drafts 87.4 against 77.9). On the fixed-corpus benchmark at 4k the best candidate cells are 122.6 tok/s on code (adaptive with lookup) and 87.1 on prose (one draft); the adaptive planner trailed the best fixed depth because its cost table predated the merge. A same-day follow-up retuned the table and settling behaviour: defaults again reproduce every hash at +9.2 to +17.8 percent, and adaptive on 4k prose rises from 77.4 to 83.9 tok/s against 86.2 to 87.3 at one fixed draft and 76.2 to 76.7 off; sampled adaptive is not established either way. Small samples, host not screened, not a matched engine comparison; not committed or installed.

- [20261119-SUSHI-02: Lily MTP round profile](2026-11-19-lily-mtp-round-profile.md): full-model timing and instrumented stage attribution on the pinned Lily Q4-XL artifact at 4,096 input tokens, MTP off and one to three drafts, greedy and sampled. Verify is 86 percent of a sampled two-draft round; each extra verified row costs about 5.7 ms of GPU time (routed experts 2.6, hyper-connections 1.15, GDN 0.55 to 1.65, attention 0.54) against a 13 to 14 ms serial step, and host work is 2.5 to 3.8 ms per round. Diagnostic only, one unscreened session; six follow-up attempts are opened and recorded separately.

- [20261119-SUSHI-01: Sushi and Stark on Qwen3.8-Flash-Next, source and record comparison](../benchmarks/2026-11-19-sushi-flashnext-lily-assessment.md): a reading of both projects' sources and published records, with nothing built or measured. Sushi's serial decode on its 4bpw pack is recorded below Stark's Lily path; its higher headline rate comes from speculative rounds whose extra verified rows cost about 3 ms against a derived 9 ms in Stark, adaptive depth to eight drafts with prompt lookup, less host readback, and greedy fixed-corpus benchmark conditions. No matched comparison exists and no Stark change is qualified by it.

- [20261119-LILYROWS-01: shared-weight hyper-connection reads in MTP verify](2026-11-19-lily-hc-verify-rows.md): a shared-weight exact HC down kernel for verify rows is byte-identical and passed the eight-arm output-hash gate; its component GPU time per pass of 97 HC reads was 1.087/1.363/1.681 ms against 1.185/1.571/1.899 ms at 2/3/4 rows, and the in-model effect was inconclusive (two instrumented sessions disagreed by up to 30 percent at 3 rows). The shared-weight up half is not byte-exact under fast math and was rejected. All timings are unscreened shared-host screening. Not adopted on risk against benefit; the code is not on the integration working tree and its patch is [archived](../archive/experiments/2026-11-19-lily-verify-rows/README.md).

- [20261119-LILYROWS-02: Lily GDN verify replay rollback](2026-11-19-lily-gdn-verify-replay.md): the verify recurrence writes only each GDN layer's final state into a double-buffered shadow, and a partial accept replays the committed rows from the pre-verify state, removing 113,246,208 bytes of per-row state writes per extra row. Kernel byte tests, the real-model control (verify, rollback and host-serial phases zero unequal) and all eight output hashes pass. Unscreened shared-host screening: in-model recurrence 1.651/1.711/1.846 ms against 1.850/2.078/2.386 ms at 2/3/4 rows, end-to-end +1.6 to +3.4 percent on MTP arms. A follow-up established the cause of the row-3-and-4 GDN input-projection slope (per-row input re-read, BF16 widening and input sum in every output channel) and tried three restructurings; the only faster one was byte-identical in 288 synthetic cases but changed 3,476,470 real-model verify values and every MTP arm's output hash, so nothing from the follow-up is adopted. The replay rollback is adopted on the integration working tree: merged with LILYHOST-01, LILYDEPTH-01 and LILYBENCH-01, uncommitted, not installed, final quiet timing pending.

- [20261119-LILYROWS-03: share routed experts across verify rows](2026-11-19-lily-moe-verify-rows.md): cross-row expert overlap is 17 to 30 percent of reads, but exact sharing kernels, byte-identical and passing the real-model control, are not faster in unscreened shared-host component screening at 3 rows (gate/up 4.13 against 4.21 ms, a tie; down/combine 3.39 against 2.72 ms, a regression). The per-row slope is arithmetic-bound, not DRAM-bound. Rejected; the code is not on the integration working tree and its patch is [archived](../archive/experiments/2026-11-19-lily-verify-rows/README.md).

- [20261119-LILYHOST-01: fewer host round trips in Lily MTP rounds](2026-11-19-lily-mtp-host-roundtrips.md): a vectorized greedy sampler, a greedy draft chain on the device in one submission that reads back only proposal ids, encode-ahead of draft and verify passes, borrowed verify rows with rows after the first rejection unread, and polled waits. Every baseline output hash is reproduced and the real-model control, extended with a device-versus-host greedy section, passes. Unscreened shared-host interleaved screening of the final source against a base rebuild: s2 85.21 against 78.46, g2 102.91 against 91.14, s3 73.24 against 68.16 and g3 94.89 against 89.53 tok/s. Adopted on the integration working tree: uncommitted, not installed, final quiet timing pending.

- [20261119-LILYDEPTH-01: Lily MTP adaptive depth and prompt-lookup drafts](2026-11-19-lily-mtp-depth-policy.md): opt-in `mtp_policy: "adaptive"` (a deterministic expected-value planner over per-position acceptance and a static round-cost table from SUSHI-02) and `mtp_lookup: true` (4-gram prompt-lookup point-mass drafts with a head cache-row advance). Default requests reproduce all eight output hashes, greedy adaptive and lookup equal the greedy baseline, and sampled requests repeat exactly. Unscreened shared-host screening: on the fixed-corpus ladder adaptive matches the best fixed depth on code and gives 71.1 tok/s on 4k prose against 66.9 at two drafts and 77.0 with MTP off; greedy lookup on copy-heavy input gives 107.4 against 92.7 tok/s at two head drafts with 638 of 642 lookup drafts accepted; sampled lookup is not useful at temperature 1. Adopted default-off on the integration working tree with its cost table unchanged (it predates the host round-trip change): uncommitted, not installed, final quiet timing pending.

- [20261119-LILYBENCH-01: Lily fixed-corpus benchmark under Sushi's bench conditions](2026-11-19-lily-fixed-corpus-benchmark.md): `scripts/benchmark_lily_fixed_corpus.py` ports the corpus generator, prose chunker and request ladder of Sushi's release bench (MIT; notice in `LICENSE.sushi-reference`) and passes 29 model-free tests. One unscreened shared-host screening run on the base worker: 48 of 48 samples valid, greedy output hashes identical across depths; best fixed depth 92.2 tok/s on 4k code and 62.1 on 4k prose beside Sushi's published 146.2 and 79.5, which is not a matched comparison. The tool is merged into the integration working tree, uncommitted.

- [20261118-D13-01/02: regular Decider v1.3 admission and smoke timing](2026-11-18-decider-v13.md): the pinned 27B BF16 image checkpoint receives a separate Decision-only source/catalog profile and a headless untied-embedding conversion route. Focused model-free controls pass, conversion and native load succeed, and one 201-token M5 Max request processes 4.001 input tok/s; reference parity, images, peak memory and repeatability remain unqualified.

- [20261118-ELEPHANT-01: pinned Elephant 1 Decision admission](2026-11-18-elephant-1.md): source-aware Qwen3.5 block-attention and 256-slot OneJev readout implementation, catalog/install admission and separate image-retaining runtime path. Pinned metadata preflight passed; full 9B numerical, lifecycle, memory and performance qualification remains pending.

- [20261117-QWEN38-CATALOG-01: official dense catalog admission](2026-11-17-qwen38-dense-catalog.md): the complete official `Qwen/Qwen3.8-27B` is pinned in the existing dense Qwen3.5/Qwen3.8 family, with reviewed image and XML tool-template identities. Metadata and model-free admission pass; full-model conversion and native qualification remain pending.

- [20261117-Q38OFF-01/02 · official dense Qwen3.8 template admission](2026-11-17-qwen38-official-template-admission.md): the official 27B BF16 image conversion completed, but native admission rejected its raw/normalized chat-template pair. Binding both reviewed hashes and fixing desktop load diagnostics lets the exact converted artifact pass native load, eight-step text execution and the full automatic image check. Performance, source numerical parity, cache/MTP qualification and installed-app status remain separate.

- [20261117-FLASHSPEED-11–16 · Lily midpoint speed and release](2026-11-17-lily-midpoint-speed-and-release.md): the exact pinned retained-vision Lily Q4-XL artifact enables midpoint reuse by default with an environment kill switch. At 12K, normal append-only follow-ups were 40–46% faster to first token and fresh-plus-follow-up cycles improved 19.4%; full-model, Metal and all four lifecycle checks passed.

- [20261117-FLASHSPEED-11–14 · Lily midpoint integration](2026-11-17-lily-midpoint-integration.md) isolates the default-on, pinned-artifact midpoint runtime and tests. The pinned Lily Q4-XL benchmark matched logits and two-draft MTP work. Median Pi turn time was 15.3% lower across eight matched pairs. The owner accepted the observed 12K tradeoff for the integration decision.

- [20261117-LILYSTART-01–03: Lily startup mode and memory admission](2026-11-17-lily-startup-mode-memory.md): default non-Decider starts select Chat; expected startup failures retain recovery guidance; exact pinned Lily separates selected-residency headroom from its unchanged full planning estimate. Native readiness and fresh automatic image checks pass; no new native precision or performance qualification.

- [20261116-DFHTTP-01: Decider HTTP and latency diagnosis](2026-11-16-decider-http-latency.md): authenticated native decisions and signed local installation passed text/image HTTP controls. Three-length profiling identifies native forward time as the main cost; no native speedup claim.

- [20261116-DF13APP-01–02: Decider Flash app integration](2026-11-16-decider-flash-app.md): Decision-only setup and one-image workflow; signed relocated bundles match 5/5 reference choices, including three image cases. Installed GUI image decision/removal/Stop passed after fixing Add from Source defaults; performance remains separate.

- [20261116-DF13-01–03: Decider Flash v1.3](2026-11-16-decider-flash-v13.md): dedicated native BF16 text/image scoring; final artifact matches 15/15 reference decisions and passes lifecycle checks. Tower-residency timing is inconclusive; desktop integration remains separate.

- [20261116-PIAUTO-01: automatic API worker selection](../archive/runs/2026-11-16-automatic-api-worker.md): Decision Playground to Pi generation and back, typed preparation errors, one-worker ownership, signed local installation and bounded Qwen3-4B checks; no timing or broader artifact qualification.

- [20261116-LILY-REUSE-02: independent review and clean PR preparation](../archive/runs/2026-11-16-lily-projection-pr-review.md): no kernel blocker; a corpus-copy privacy issue in the bundled benchmark CLI is fixed with one RAM snapshot and 27 passing model-free tests. A main-based branch excludes inherited app/catalog and retired MTP-policy edits, preserves all benchmark data, and matches all 1,213 validated Rust/Metal/build-input files. Fresh isolated release build and 240-case physical component replay pass; original model results keep their build identities and the old corpus-copy behavior receives a dated correction.

- [20261116-LILY-REUSE-01: adopt verifier projection reuse](2026-11-16-lily-projection-reuse.md): group-64 Q4/BF16 and Q8 verifier weight reuse is now the normal branch path after the user accepted the modest long-context gain. Final component controls pass 240 cases / 10,436,400 bytes exactly; 56 native controls cover depth comparisons, prefixes, images, cancellation, and identical full coding/planning/book replays. [Run record](../archive/runs/2026-11-16-lily-projection-reuse.md) preserves host-wrapper resource deferral and corrected harness setup separately. Two drafts remain recommended; earlier evidence and installed build unchanged.

- [20261115-LILY-LONGCTX-01: very-long-context InfiniteBench performance case](../benchmarks/2026-11-15-lily-long-context-infinitebench.md): one pinned, unmodified whole-book summary request renders to 88,398 tokens; context 98,304/output ceiling 4,096. Sixteen measured responses plus four full-context warmups on sealed baseline/candidate workers, two drafts fixed and images loaded, preserve all output/work hashes. Paired decode improves 5.40% (8/8, both halves positive), complete request time 2.39% (7/8). All host observations and paging increments remain included; planner success-only metadata limit is explicit. Additional timing/trajectory evidence, not a benchmark quality score or promotion. Prior data is unchanged; no native source/build/install/default change.

- [20261115-LILY-VERIFY-01/02/03/04: small-row verification on natural long responses](../benchmarks/2026-11-15-lily-small-row-verifier.md): coarse and finer stage attribution, an unbuilt head-only candidate rejected on its 5–6% ceiling, and one packed-weight projection-reuse prototype. The public-dispatch component control passes 60 cases / 9,375,336 bytes exactly. The uninstrumented image-loaded comparison retains 32 measured responses plus eight warmups, two drafts fixed, with identical hashes/lengths/MTP work. Coding improves 1.84% useful decode (6/8 wins), planning 4.54% (8/8); both fail the 10% practical gate. Candidate archived and rejected for promotion, with initial invalid-selector request and unbuilt source variants retained. Production source, two-draft recommendation and installed build unchanged.

- [20261115-LILY-LONG-01: natural coding response](../benchmarks/2026-11-15-lily-long-coding-response.md): eight measured responses plus four warmups, before two drafts versus candidate automatic one, same pinned image-loaded artifact. One loses all four paired decode/request comparisons (-11.35% useful decode, +38.06% time) and emits 22.72% more tokens. Both unique programs pass syntax; execution is untested because runtime isolation was not qualified. Initial zero-worker stdin preflight is preserved. This counterexample contributed to reverting the automatic one-draft policy; current Lily recommendation is two. No native code or installed-build change.
- [20261115-LILY-SLACK-LONG-01: natural planning response](../benchmarks/2026-11-15-lily-long-planning-response.md): eight measured responses plus four warmups on the same before/candidate pair. +1.02% paired useful decode is inside observed noise; -3.83% paired request time includes 2.86% fewer tokens and unequal background load. Both arms finish naturally; no business-plan quality grading. This result is inconclusive and does not counter the coding result or restore the automatic one-draft policy. No native code or installed-build change.

- [20261115-LILY-MTP1-04/05/06: pinned Lily sampling-specific recommendation experiment](../benchmarks/2026-11-15-lily-sampling-mtp-policy.md): the exact-receipt automatic one-draft candidate was later reverted; current Lily recommendation is two for all sampling profiles. Its held-out top-k request latency improved 14.66% (8/8 paired wins) in the retained 64-request, four-session confirmation, with 16 warmups and four setup requests; seven image-loaded prefix transitions matched fresh outputs and image clearing. These historical results remain workload-specific and are preserved in the [restoration record](../archive/runs/2026-11-15-lily-mtp-two-draft-default-restored.md) and [aggregates](../benchmarks/data/lily-sampling-mtp-policy-2026-11-15.json).

- [20261115-LILY-MTP1-01/02/03: pinned Lily MTP recommendation study](../benchmarks/2026-11-15-flashnext-lily-one-draft-default.md): the source-scoped one-draft candidate was reverted after the artifact-wide preregistered gate failed. The primary latency improvement was 3.58% against a 5% threshold and the long-nucleus guard lost 6.01% useful decode rate; held-out top-k favored one by 20.03% decode and 14.45% latency. Two drafts remain recommended; explicit one-draft support, scoped positive evidence, and passing selector/image/prefix checks are retained. Not installed or released.

- [20261115-LCPP1-01/02/03: llama.cpp 1.0.0 assessment and Flash Next Lily baseline](../benchmarks/2026-11-15-llama-v1-flashnext-lily-assessment.md): active Lily source audit and ranked stop/defer decisions; fresh image-resident text/MTP measurements under explicitly accepted concurrent host activity. Preserves initial readiness/image-check failure and missing-host-observation warmup failure. No optimization candidate or promotion; public timings remain separate from historical isolated measurements and image numerical qualification.

- [20261113-PIGUI-05: persistent Pi restart](../archive/runs/2026-11-13-pi-persistent-default.md): installed signed build, preserved normal Pi configuration, actual Keychain-backed authentication and native Qwen3-4B generation across two fresh default-profile client processes.
- [20261113-PIGUI-01–04: Pi Connections delivery](2026-11-13-pi-connections.md): signed app/candidate, retained nvm/fd and pre-PR recovery/build findings, manual GUI setup and fresh bundled-profile Qwen3-4B checks.
- [20261113-PI-01–08: guarded Pi manual qualification](2026-11-13-pi-manual-qualification.md): missing-usage/DONE guard, retained failed/superseded harness recipes, and final 21-check installed-Pi/Qwen3-4B backend qualification; manual RPC scope only.

- [20261112-PHI-01: untied Phi3 output head](2026-11-12-phi3-untied-embeddings.md): untied `phi3` plans a separate output embedding and passes synthetic MLX-LM, lifecycle and prefix-reuse controls; actual Phi-3.5-mini execution failed (head dimension 96 has no Metal attention kernel) and that geometry is now refused in preflight. No real untied checkpoint qualified.
- [20261112-CATALOG-01: shared config normalization](2026-11-12-catalog-config-normalization.md): one reviewed rewrite pass before every in-scope family planner, shared by catalogue admission and conversion; unknown keys and real RoPE variants still fail closed. Header admission only; converted artifacts unqualified.
- [20261110-CATALOG-01: catalogue family admission](2026-11-10-catalog-family-admission.md): shared header planners, interleaved discovery and unsupported reasons. Pre-PR review repaired image-to-text fallback after vision header failure. Full checkpoint qualification remains separate.

- [20261110-TOOL-QUALITY-01: bounded tool output and continuation replay](2026-11-10-tool-output-quality-replay.md): five installed artifacts checked; all passed synthetic tool continuation, four passed ordinary-chat exact match; 17×19 arithmetic controls isolated no demonstrated parser failure.
- [20261110-TOOL-QUALITY-02: restored Qwen3 0.6B artifact replay](2026-11-10-tool-quality-qwen06-replay.md): pinned-source recreation failed the tiny-tool control; replay of actual host input fit context and ended at EOS with no calls. Model/workload limitation supported, no parser repair established; temporary import removed.
- [20261110-G4INSTALL-01: Gemma installation repair](2026-11-10-gemma-install-repair.md): bounded HF config-link authorization, E4B conversion/app image smoke, and content-free failure diagnostics.

Current continuation: [Gemma3 cache and Gemma4 assistant implementation](2026-11-09-gemma-next-image-parity.md)
records attempts `20261109-G3IP-01` and `20261109-G4AP-01`; execution is pending.

Use this directory to track attempts to make a model work correctly or run faster.
Record unsuccessful, invalid and resource-deferred attempts alongside successful
ones. A later passing run does not erase the earlier result.

This ledger connects a change to its evidence and shipping decision. Numerical
receipts, benchmark aggregates, source patches and detailed run records retain
their existing canonical locations under `docs/archive/`, `docs/benchmarks/` and
`provenance/`. Private model payloads and evaluation data stay ignored.

## Starting records

| Record | Scope |
| --- | --- |
| [20261112-DENSE-MLX-01/02/03: Qwen2 and Mistral MLX affine import](2026-11-12-dense-mlx-qwen2-mistral.md) | Planner gates opened for uniform 4/8-bit MLX; nine live repositories pass real preflight and header planning (re-checked after the main merge). Qwen2.5 0.5B 4-bit fails the stock-MLX logit/greedy gates (0/5), inside MLX's own F16/BF16 envelope; prefix reuse exact. The pinned Mistral 7B v0.3 4-bit export fails conversion (named chat template); a template-substituted copy fails 1/5 logit cases and passes greedy checks. Not qualified as BF16. |
| [20261112-DENSE-MLX-04/05/06: native F16 dense profile](2026-11-12-dense-mlx-qwen2-mistral.md#20261112-dense-mlx-04-qwen25-05b-instruct-mlx-4-bit-native-f16-profile) | Engine F16 activation/KV profile for plain full-attention Llama/Qwen2/Mistral graphs. The F16 Qwen2.5 0.5B 4-bit artifact passes all three gates (5/5 logits at 0.37-0.61%, greedy exact, prefix exact). The F16 Mistral diagnostic copy passes logits 5/5 and workers; its prefix aggregate fails on the template rule as before, and the published revision still fails conversion. BF16 outputs are bit-identical across builds. No timing ([run record](../archive/runs/2026-11-12-dense-f16-native-qualification.md)). |
| [20261112-DENSE-MLX-08: system-role fold and prefix reuse for the published Mistral export](2026-11-12-dense-mlx-qwen2-mistral.md#20261112-dense-mlx-08-system-role-fold-for-the-published-mistral-export-and-its-prefix-reuse-gate) | Conversion records `fold_into_first_user` for templates that refuse a system message. System prompts then render on the published export, with weights and config unchanged and workers identical. The prefix-reuse gate passes with the opt-in `--session-kind user-assistant` (32/32 exact; follow-ups reuse 2,048). Tool histories untested; not promoted ([run record](../archive/runs/2026-11-12-install-auto-f16.md#system-role-fold)). |
| [20261112-DENSE-MLX-07: published Mistral export through the app install](2026-11-12-dense-mlx-qwen2-mistral.md#20261112-dense-mlx-07-published-mistral-7b-v03-mlx-4-bit-export-through-the-app-install-route) | The named chat-template list converts with its `default` entry. The desktop local install selects F16 and reproduces the `-05` weights, config and tokenizer; workers are identical to `-05`; tools are `unavailable`. Prefix reuse is not run: the harness cannot render on this template (system and tool roles). Not qualified ([run record](../archive/runs/2026-11-12-install-auto-f16.md#published-mistral-export-named-template-list)). |
| [20261112-FNI-01–07: general Flash Next image admission](../archive/runs/2026-11-12-flashnext-general-images.md) | Shared Add Source admission, actual KQuant lifecycle and image fixtures, and contract/harness follow-up; compatibility and artifact-specific qualification remain bounded to the recorded evidence. |
| [20261110-Q38IMPORT-01: external-table affine import](2026-11-10-qwen38-external-import.md) | Live iQ-MLX catalog and preflight, complete header comparison, synthetic import and manifest replay. Full checkpoint execution deferred for storage. |
| [20261110-LILY-SIZE-01: retained image resolution](2026-11-10-lily-image-size.md) | Retained Lily 16 MP processor ceiling, negotiated app limits, 65,536-row component controls and 2.4 MP native execution; signed local build installed, full-ceiling model qualification pending |
| [20261110-FAMILY-RECHECK-01: independent pre-PR review](2026-11-10-family-double-check.md) | Isolated cache invalidation controls, accepted-temperature overflow handling and scratch-pool high-water admission; revised validation tracked separately |
| [20261110-DENSE-01: capped attention and dense sampling](2026-11-10-dense-attention-sampling-correctness.md) | Gemma 2 soft cap, recoverable sampling validation, dense MTP filter semantics, nonfinite readout errors and optional snapshot copying; component controls and artifact limits tracked separately |
| [20261110-CONVERT-01: family conversion regressions](../archive/runs/2026-11-10-family-converter-correctness.md) | Safe dense config annotations, Qwen3.5 norm convention and size-specific metadata; RNJ/Polaris remain planned pending YaRN support |
| [20261110-IMAGE-01: image loading and request admission](2026-11-10-family-image-correctness.md) | Single component scan with mutation checks, skipped PNG ancillary inflation, primary MPO decoding and tower workspace accounting; artifact qualification pending |
| [20261110-PREFIX-01: text-only snapshot routing](2026-11-10-text-prefix-snapshot-capability.md) | Route snapshot-only ordinary text through existing exact state capture/restore; artifact qualification pending |
| [20261109-LAUNCH-RECHECK-01: image-resident Lily text controls](2026-11-09-model-launch-double-check.md) | Same ddalcu payload with explicitly prepared image metadata passes seeded three-draft history, 4,096-token reuse and real image-to-text recovery; original folder is not image-admitted, paging prevents speed claims |
| [20261109-QMEM-01: F32 Qwen scalar attention workspace](2026-11-09-qwen-attention-memory.md) | Independent lazy scalar scores/probabilities and matching allocation inventory; allocation/stride and synthetic full-forward controls passed, production speed unmeasured |
| [20261109-LILY-STARTUP-01/02/03/04: retained image startup](../benchmarks/2026-11-09-lily-image-startup.md) | Initial startup canceled; accelerated full scan passed image smoke but latency rejected; digest reuse/UI cleanup installed, native startup 10.5–11.9 s, routine UI Ready by 14.774 s; final image attachment and copy-removal checks passed |
| [20261109-QDIP-01: dense default-image text parity](../archive/runs/2026-11-09-qwen-dense-image-default-parity.md) | Preserve the supported affine dense text MTP head with the image tower resident; source-bound cache, conversion and performance qualification is tracked separately |
| [20261109-QDIP-02/05/06: dense image-loaded text controls](../benchmarks/2026-11-09-qwen-dense-image-text-parity.md) | Separates tower residency from changed target arithmetic; corrected short 27B comparison fails text speed parity despite exact output hashes and restored MTP |
| [20261109-QDIP-07: shared-weight 27B fast text](../archive/runs/2026-11-09-qwen27-shared-weight-fast-text.md) | Exact Jundot target/head payload and configuration equality; normal text regains Legacy graph and batched MTP with images resident. Paired 64-token MTP0/3 medians pass bounded parity; all 14 nonperiodic near-capacity cache/lifecycle controls pass. Other artifacts retain separate gates. |
| [20261109-QNIP-01/02/03/04/05: remaining Qwen image parity](../archive/runs/2026-11-09-qwen-remaining-image-text-parity.md) | Three headless exports pass short speed gates; BF16/affine 0.8B each pass 30 lifecycle controls. Headed BF16 MTP0 passes 34/28 lifecycle checks plus 48/26 geometry/image-recovery checks, ordinary hybrid restoration and strict short speed gates. Test-only reference FC fixes fusion; broader head projections and materialized SiLU still fail full-head gates. MoE cache remains unqualified after foreign/pageout deferrals and a functional-only hard 64 GiB available-memory stop; frozen source rules out full recurrent-state histories. Tiny affine operator measures command-heap growth and exact split/reuse, without full-model attribution or policy adoption. Retained ddalcu has a complete source/header peak plan and remains held; BF16 MoE full payload is absent. |
| [QNIP bounded prefill / source-rounded head continuation](../archive/runs/2026-11-09-qwen-moe-bounded-prefill-memory-plan.md) | Physical 256 preserves the logical 1,024 grid; two CPU controls, four sigmoid controls and 16 synthetic state/cache cases pass. Actual MoE 1,026-token cold/reuse/changed-tail controls pass 4/4 and match the frozen worker's cold output. Short 64-token image/text comparison passes fixed 5% checks with zero observed paging/compression/swap; full 1,792-token lifecycle remains pending. Source-rounded head still fails stage zero hidden absolute error (0.09375 > 0.0625), with two subsequent single-row stages exact; no head arithmetic adoption. |
| [QNIP pinned wide projection / production head continuation](../archive/runs/2026-11-09-qwen-remaining-image-text-parity.md) | Pinned MLX reduction and source operators pass the exact 0.8B BF16 unchanged head gate without test selectors. Production selection requires Legacy, qualified dense geometry and full-BF16 target/head metadata; target/MTP0 remain unchanged. Actual 1,026-token lifecycle passes 28 text / 34 image controls. Short 62-of-64-token loading-mode comparisons pass fixed 5% checks for MTP0 and MTP3, with zero observed paging/compression/swap; MTP3 is 49.03% slower on this small model. Initial timing overlapping package provisioning is retained/excluded. Other sizes/layouts remain unqualified. |
| [20261108-LVMTP-01: retained vision text MTP](../archive/runs/2026-11-08-lily-vision-text-mtp.md) | Lily vision/text MTP parity passed bounded output, 4,096-token prefix and image-transition controls; short paired text speed matched; memory recipe updated |
| [Image-support retrospective](2026-11-08-image-support-retrospective.md) | Conversation-wide findings, model coverage, validation mistakes, review, delivery and remaining work |
| [Qwen image attempts](2026-11-08-qwen-images.md) | BF16 and affine dense Qwen, affine and BF16-source MoE, retained Flash-Next/Lily numerical repairs and resource controls |
| [Gemma image attempts](2026-11-08-gemma-images.md) | Gemma 3 processor/convolution fixes; Gemma 4 efficient, unified, MoE and larger-tower attempts; decoder admission and retained compatibility |
| [20261109-G4IP-01: Gemma 4 initial image-loaded text controls](2026-11-09-gemma4-image-text-parity.md) | Ring snapshot components and short loading-mode measurements; original worker cache qualification withdrawn after full-prompt replay was found |
| [20261109-G4IP-02: Gemma 4 warmed cache correction](2026-11-09-gemma4-image-text-parity.md#20261109-g4ip-02-warmed-alternating-cache-timing) | Suffix-dispatch fix, actual-prefill invariant, nonperiodic/capacity regression and eligible warmed alternating controls; separate assistant conversion/execution remains open |
| [20261108-VTP-01/02/04: vision artifact text prefix reuse](../archive/runs/2026-11-08-vision-text-prefix-reuse.md) | First recipe missed reuse coverage with exact outputs; stable-prefix and final restore-fence workers passed all 32 numerical/lifecycle cases on official Qwen3.5-0.8B BF16 |
| [20261108-VTP-03: paired vision text prefix timing](../benchmarks/2026-11-08-qwen35-vision-text-prefix.md) | Four observations per arm; bounded first-token benefit, cache-cold cost and snapshot payload; other artifacts and host-screened performance remain unqualified |
| [20261108-VTP-05: quick Lily text sanity check](../archive/runs/2026-11-08-vision-text-prefix-reuse.md) | Retrospective short unscreened text comparison; identical outputs and no apparent regression; separate from vision qualification |

The `IMG-20261108-QI-*`, `IMG-20261108-GI-*` and `IMG-20261108-XI-*` IDs register
historical attempts on 2026-11-08. The registration date is not a claim about
when each experiment ran. Original dates and identities remain in the linked
records. This image batch covers the image-support conversation and its
explicit prerequisites. The context/resource-budget attempts are indexed below.
These batches are not an exhaustive backfill of all Stark research.

Existing performance work remains indexed in the
[benchmark index](../benchmarks/README.md). The
[dense Qwen3.8 approach ledger](../benchmarks/2026-11-06-qwen38-27b-approaches.md),
[Flash-Next projection experiment](../benchmarks/2026-11-07-flashnext-moe-projections.md),
[sustained-throughput diagnosis](../benchmarks/2026-11-07-flashnext-sustained-diagnosis.md)
and [cross-backend idle-wake study](../benchmarks/2026-11-08-cross-backend-idle-wake.md)
are related records from other work. Their findings must retain their own
checkpoint, host-screening and source boundaries; they are not new measurements
from this retrospective.

## Record an attempt

1. Before changing code or running a comparison, copy the
   [attempt template](TEMPLATE.md) into a dated Markdown record. Choose a unique
   `YYYYMMDD-AREA-NN` or `WORKSTREAM-YYYYMMDD-AREA-NNN` ID
   (for example, `20261108-CTX-01` or `IMG-20261108-QI-001`)
   and list the model/checkpoint variants it actually
   targets. Keep source-bound validation records under `docs/archive/runs/` and
   timed studies under `docs/benchmarks/`; the ledger links their canonical
   evidence. State the observed problem, hypothesis, baseline, intended change,
   fixed acceptance limits and stop conditions. Use `not recorded` for missing
   historical details; do not reconstruct them from a different run.
2. Bind the checkpoint revision and artifact hashes, baseline and candidate
   commits/executables, source references, tools, device/OS and precision policy.
   Record conversion, kernel, cache, sampler and MTP choices that can affect the
   result. Separate image/tensor/component checks from full-model execution.
3. Run the smallest useful correctness controls first. For a speed claim, retain
   a matching correctness gate and a controlled end-to-end timing comparison.
   Preserve prompt/input lengths, output budget, warm/cold/reuse mode, seed,
   sampling, draft depth, repetitions, paired order and host/resource screening.
   Publish aggregates and hashes, never prompts, generated text, token vectors,
   probabilities, credentials or raw exception dumps.
4. Append what happened, including every failed gate, rejected arm, invalid setup,
   guard stop and known counterexample. Give each changed candidate or materially
   changed measurement recipe its own attempt or follow-up ID. Keep prior
   receipts immutable and link the predecessor.
5. Record the **outcome** and **promotion decision** separately, with reasons and
   the next test. A component pass, synthetic kernel speedup or unscreened timing
   observation does not establish a production full-model gain. Name each exact
   artifact whose qualification remains missing.
6. Add the entry to this ledger and the [source index by model](source-index.md), and link its ID from the implementation PR. For a performance attempt, record its scoped change or timing limitation in the [family speed history](../MODEL_RESEARCH.md#record-the-next-result). Update
   the current model-support/reference page only when the evidence supports a
   capability change. When a new comparison supersedes a benchmark, update its
   current/superseded index without changing the historical measurements.
7. For a performance attempt, read the
   [performance playbook](../PERFORMANCE_PLAYBOOK.md) before choosing the
   change, and update it in the same pull request when the result changes what
   a lever returned, adds a failed approach, adds or settles an open idea, or
   changes a gate or measurement rule. The playbook links here; it does not
   replace the record.

For a shared infrastructure change, one record may contain a per-artifact
matrix. Different checkpoints, quantizations, precision policies, MTP depths,
workloads or source revisions need separate results; a shared code path or
passing unit suite does not qualify every artifact.

This is a repository documentation workflow, not an automatic experiment
collector or evidence validator. Reviewers must check the linked receipts and
the claimed scope. A record does not authorize a large model run, a download,
publication or promotion by itself.

## Outcome and promotion vocabulary

| Outcome | Meaning |
| --- | --- |
| Planned / untested | Proposed or implemented route without the named execution evidence |
| Passed | The stated gate passed for the exact recorded inputs and build; name whether it is structural, component, full-model, lifecycle or timing evidence |
| Failed | A valid attempt missed its fixed gate; retain the failing metric and diagnostic limits |
| Invalid setup | The intended experiment was not run with matching inputs/tools/builds; results do not decide model correctness or performance |
| Resource-deferred | Admission or a resource guard prevented completion; this is neither a numerical pass nor a numerical failure |
| Inconclusive | Execution completed but controls, coverage or timing eligibility do not support the intended conclusion |

Use a separate decision such as `promoted within named scope`, `default-off`,
`diagnostic only`, `rejected`, `deferred` or `awaiting qualification`. An
intentionally rejected mutation is a passing **admission negative control**, not
a failed implementation. A counterexample can remain unresolved even when a
bounded feature gate passes. Never turn `implemented`, `merged`, `released` or
`CI passed` into a substitute for model qualification.

## Required evidence boundaries

- A checkpoint name, parameter count or catalog row does not identify the actual
  artifact, quantization, processor/template or selected native path.
- Full logit parity, greedy agreement, image-feature similarity, image-answer
  correctness and lifecycle safety answer different questions.
- A BF16 tower with an affine decoder differs from a quantized tower. A typed
  text-only conversion differs from a retained format containing vision.
- Equal output hashes do not establish throughput. Faster component timing does
  not establish faster requests. Memory admission caps and sampled RSS do not
  establish a full-model working-set guarantee.
- Prefix restoration, cancellation, reset, scope ownership and MTP must be
  assessed for every state-affecting route, following
  [model onboarding](../MODEL_ONBOARDING.md).

Backfill older work by linking its existing canonical record first. Mark missing
comparators or receipts explicitly; do not silently fill gaps with current
results from a different build.

## Evidence and status rules

| Dimension | Required distinction |
| --- | --- |
| Attempt state | Proposed, attempted, completed, blocked or superseded. A proposal is not an executed trial. |
| Result | Passed, failed, inconclusive or not measured, with the criterion and evidence scope stated. “Passed” alone is insufficient. |
| Evidence scope | Source review, model-free test, native smoke, numerical qualification, matched performance study or deployment check. |
| Correctness | Exact reference/artifact, aggregate error/hash agreement and tested operator/request scope. Equal errors are not output parity. |
| Performance | Same checkpoint/quantization, workload/options and timing boundary; repeats/order, host eligibility, cache state, dispersion, memory and excluded attempts. A short smoke is not a speed result. |
| Disposition | Experiment only, retained optional, adopted on branch, merged, released, installed, rejected or superseded; attach commit/PR/build evidence. CI success is not deployment. |
| Provenance | Source and dependency/vendor revisions, patch/manifest and executable/model hashes. A malformed or missing identity stays unresolved. |
| Safety/lifecycle | Context/resource admission, cancellation, error recovery, prefix reuse and clearing, scratch lifetime, and known limits. |

Existing studies also include the [Qwen3.8 optimization history](../benchmarks/QWEN38_OPTIMIZATION_HISTORY.md); its findings retain their original scope. The [performance playbook](../PERFORMANCE_PLAYBOOK.md) summarizes the Flash-Next performance attempts by lever for use on other families; it is a guide to these records and qualifies nothing.

Use totals only within their recorded scope; overlapping reruns must not be summed. Report sampled tokens versus emitted/useful tokens explicitly. Do not use a directory name, available artifact, implemented backend or successful EOS as proof of broad model quality or speed.

## Context/resource-budget attempt index

All rows below link to the [complete findings and per-model matrix](../archive/runs/2026-11-08-context-resource-budget-findings.md). `CTX-NN` expands to `20261108-CTX-NN`. The implementation's runtime revision `9a3e7216` merged through [PR #99](https://github.com/jshahbazi/stark/pull/99) as `ea098a24`. This records runtime adoption; it does not establish release, installation or broader artifact qualification. This ledger and its workflow are a separate documentation addition.

| ID | Scope and attempt | Recorded outcome | Disposition |
| --- | --- | --- | --- |
| CTX-01 | Shared cap audit across model entry points | Source confirmed independent pre-tokenization policies. | Diagnosis recorded; coordinated change in PR. |
| CTX-02 | User-facing fixed resource setting | User rejected the interface requirement. | Superseded by automatic selection; developer override retained. |
| CTX-03 | Shared framing/template/content budget | Model-free large-input, sandbox and recovery controls passed. | Implemented in PR. |
| CTX-04 | Finite Auto and output ceiling removal | Qwen3.5 emitted 4,097 sampled tokens; short HTTP Auto/retry worked. Early identity has CTX-19's limitation. | Implemented; bounded native evidence. |
| CTX-05 | Explicit output requirement for unknown capacity | Did not satisfy automatic behavior requested by the user. | Superseded; legacy update guidance retained. |
| CTX-06 | Unknown-capacity memory planning, including Dense MTP | Synthetic ShortConv HTTP/CLI passed; allowance not filled. Dense MTP coverage is model-free. | Implemented; trained-artifact qualification pending. |
| CTX-07 | macOS memory probe prefix compatibility | Native smoke exposed rejection of a valid older statistics prefix. | Corrected and retested. |
| CTX-08 | Current-main integration/vision variants | Incomplete matches and unrelated branch content required correction. | Clean replay and integration fix; no new model qualification. |
| CTX-09 | Finite generic/GLM/Qwen/Flint/Lily cache admission | Source review found growth gap; native Qwen3.5 smoke and family inventory tests passed. Other families lack runs here. | Implemented; per-artifact limits retained. |
| CTX-10 | Unicode/concurrent HTTP parsing admission | Initial three-times-raw estimate was insufficient. | Replaced by shared twelve-times-raw estimate and atomic reservations. |
| CTX-11 | Hybrid inventory failure recovery | Error classification could unload a healthy model. | Corrected; model-free retry tests passed. |
| CTX-12 | Per-buffer and shader-index bounds | RAM-only fit was insufficient. | Checked geometry/device limits implemented and tested. |
| CTX-13 | Scoring retained-pool admission | Generation's pool-lifetime assumption was insufficient. Dense Qwen3 and hybrid Qwen3.5 exact-fill smokes passed. | Implemented; long scoring workload unqualified. |
| CTX-14 | Scoring overcharge refinement | Draft sampler/per-chunk charges were unnecessarily restrictive. | One final readback charge and final-pool allowance retained. |
| CTX-15 | Private app parsing and draining | Large frames lacked current parsing-headroom admission. | Corrected; worker, Reset, Stop and retry tests passed. |
| CTX-16 | Native/hybrid image decoder admission | Short qualification check did not cover full requested decoder span. | Implemented; model-free recovery coverage only. |
| CTX-17 | Selected sparse virtual cache capacity | Short physical span did not prove full virtual reservation valid. | Separate device/index and predictor-bound checks implemented. |
| CTX-18 | Buffered callback performance | No matched timing study. | Open measurement; no speedup claimed. |
| CTX-19 | Early Qwen3.5 tokenizer provenance | Recorded SHA-256 has 63 hexadecimal characters. | Unresolved; recover original receipt, do not substitute later hash. |

## 2026-11-08 Qwen Beginner memory admission

[Source-bound record](../archive/runs/2026-11-08-qwen-guided-memory.md).

| ID | Scope and attempt | Recorded outcome | Disposition |
| --- | --- | --- | --- |
| 20261108-MEM-01 | Exact Lily Q4-XL and MLX-Serve mixed text artifacts on M5 Max 128 GiB; reconcile Beginner and native memory admission | Final trials passed 9/9 requests per artifact through near-32K, MTP on/off and reuse/fresh-scope hash agreement. Physical footprint stayed below 74 GiB, without additional swap. Earlier admission, compile, test, telemetry and staging failures are retained. | Adopted on branch; signed build installed; Coding + Balanced recommendation and actual Lily startup verified. |
| 20261108-MEM-02 | [Fresh-install recipe-bound planning](../archive/runs/2026-11-08-fresh-guided-memory.md), independent of an installed copy | Corrected empty-registry planning; 73 focused tests and full converter suite (6,791 passed, 96 skipped) cover hardware/resource boundaries and source/artifact drift. No model run. | Adopted on PR #104; signed build installed and opened without loading a model. |

## 2026-11-09 request resource ceilings and installation

| ID | Scope and attempt | Recorded outcome | Disposition |
| --- | --- | --- | --- |
| 20261109-ADM-01 | [Initial Playground correction](2026-11-09-resource-admission.md): remove fixed content/generation fractions and correct Lily resident accounting | Model-free checks and short Qwen native controls passed; original private request was not replayed. | Adopted at `b9e904fe`; installation was pending. |
| 20261109-ADM-02 | [Follow-up ceiling audit](2026-11-09-resource-admission.md): Decision fraction, stale Auto allowance and progress cutoff | Native boundary, Auto growth and late-progress controls passed. Python startup failures reproduced by the fixture's two-second deadline; six normal isolated repeats passed. | Adopted at `de60be71`; included in installed build. |
| 20261109-ADM-03 | [Signed installation and actual Flash Next control](2026-11-09-resource-admission.md) | Two short Auto requests admitted the declared 262,144 context; bounded follow-up passed on the same worker. No swap growth/guard trigger; all 459 installed overlay hashes matched. Full-window execution untested. | Signed development build installed and opened; Ready observed. |
| 20261109-ADM-04 | [Non-Lily resident-memory accounting](2026-11-09-non-lily-admission.md): generic/dense-MTP/vision decoder and GLM/Qwen/Flint state ownership | Final-source review, full Metal suite and bounded Qwen/toy controls passed; full hybrid/vision artifacts unqualified. | Adopted on branch at `08fb0075`; PR authorized; no new installation. |

## Public record boundary

Publish aggregate results and hashes only. Keep prompts, completions, token vectors, probabilities, credentials, exception dumps, model files and private evaluation/runtime artifacts out of the ledger. An evidence link can point to an ignored private receipt by a stable identifier/hash without publishing its contents. Preserve upstream licenses and distinguish unmodified source revisions from local patches.
