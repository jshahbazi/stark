# Inkling 2 Mini standard benchmark against MLX-Serve, 2026-11-23

Attempt `20261123-INKENGINE-06`, 2026-11-23, 19:10:52 to 19:50:35 local. The first standard result under [`docs/BENCHMARK.md`](../BENCHMARK.md) for any family: three repeats in the block layout with natural-text quality attached, run after optimization stopped to confirm the standing of the committed build against MLX-Serve. Every row is server level through the real worker or server. Record in the [experiment ledger](../model-experiments/README.md); every difference between the two engines and what was tried is on the [differences page](../model-experiments/inkling-mlx-serve-differences.md).

**Result.** On the default artifact pair (Stark 8-bit against MLX-Serve's author 8-bit pack) Stark is ahead on three standard rows (about 4K, about 8K, repeated-sentence prefill), behind on one (novel prose, 0.86) and unresolved on five (initial coding, sustained coding, about 512, about 16K, exact 8,800-ID prefill), where the three blocks disagree. Level to ahead except novel prose; not a sweep. On the original pair Stark is ahead on three rows (initial coding, sustained coding, about 4K), behind on two (novel prose 0.96, exact 8,800-ID prefill 0.98) and unresolved on four. Both Stark arms pass the quality bar, and the tool reports all four arms as standard results.

## Run identity

| Field | Value |
| --- | --- |
| Run directory (private, ignored) | `runtime/benchmark/inkling-2-mini/20261123-191052-inkling-2-mini` |
| Manifest SHA-256 | `7c049a5d63dd30ecdaad177af9922ef49df4eb334590f2de42d41cbb45f2060c` |
| Aggregate `summary.json` SHA-256 | `cf1d2632f8f03128c998b9278f010944b8ecf29d5f80f938e10205ec2e6b32c0` |
| Profile | `inkling-2-mini`, `scripts/benchmark_profiles/inkling-2-mini.toml` SHA-256 `a8540d22e9b4eb7a28edfc072822647e260225f0b8e8e136206bc30174f037c5` |
| Harness | `scripts/benchmark_standard.py` SHA-256 `3ed848df648763dee8c795a58c588b6cf35e7fd2ea3d7ccee1546050c48d79b4`; template helper `17183f9ec9fc…` |
| Repository | `b2a5a88f` with 12 modified paths at the start: the then-uncommitted prefix-reuse grid change, committed unchanged as `7e73e180`, and documentation drafts |
| Command | The coordinator's private `runtime/session-tools/timed.sh`: `benchmark_standard.py run --profile inkling-2-mini --repeats 3 --sessions probe,prefill` with the two Stark artifacts and the two MLX-Serve packs named below, inside one lock hold that also ran the image-parity timing of [INKAFFINE8-04](../model-experiments/2026-11-23-inkling-mini-affine8-default.md) |
| Layout, repeats, sessions | Block layout, 3 blocks with arm order reversed on odd blocks, sessions probe and prefill, one warmup per probe |
| Burst | Not run: out of scope by owner decision (one user is the goal) |
| Deviations | None; the tool reports the run eligible and every arm as a standard result |
| Status | `complete`, no arm failures |

## Arms

| Arm | Engine and binary SHA-256 | Artifact | Match class |
| --- | --- | --- | --- |
| `stark-affine8` (default) | `stark-generate` `f86d94b877b3034db82843480669112ecd97306ad09ffbdf7ff8c5addabcc2e3` | Stark 8-bit dense artifact derived from `thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`; target 18,052,013,648 bytes, SHA-256 `c9e14848…acddd`; 8-bit affine attention, embedding and head, original expert bytes; fingerprint `835107eca3b1…` | `affine8` |
| `stark-original` | Same worker binary | The original checkpoint at the same revision, prepared; target 19,526,019,168 bytes; fingerprint `52a50c387b87…` | `original` |
| `mlx-author-8bit` (reference) | MLX-Serve `v26.11.4-pre-release.1`, `05983e1278d6253d3648ddaa2514ec2a2f73ff3f617d41412d18334759991392` | `ddalcu/Inkling-2-Mini-MLX-8bit` at `cb7fa052a9bbafbbeb91750edc67046e7bacd4f1`, four shards; fingerprint `fae180d4f7fd…` | `affine8` |
| `mlx-original` (reference arm of the speed column) | Same MLX-Serve binary | The original checkpoint; fingerprint `dbf0f8e77613…` | `original` |

**What the Stark build contains.** The worker was built at about 18:55 from the code committed as `7e73e180`: everything in `4314ac51` (unquantized NVFP4 expert inputs, the history depth planner with the hold rule, wide prefill commands, BF16 prefill attention, the gathered NVFP4 prefill kernel) plus the unit-test attributes (`b2a5a88f`) and the prefix-reuse grid advertisement (`7e73e180`). Relative to engine-matrix run 5 (`4314ac51`), the arithmetic is unchanged: every hash in the coordinator's final gate run (private `runtime/final-check/`, summary SHA-256 prefix `39cdcb18a3170aae`) equals the same gate's hash on the `4314ac51` build, for example state `4d9c63e146e2` and seed logits `db19030a40f1` on the original and `4d6c9ca36a00` and `1d6268c2e1b3` on the 8-bit artifact. The same gate run passed the format check, 168 worker tests, the planner (3 of 3) and schedule-validation (6 of 6) unit tests, the exact reference on `g4096-02` and `s4096-00`, drafting on equals off at 73, 4,150, 8,800 and 16,000 tokens on both artifacts, a repeated replay with identical hashes, and the worker smoke on both artifacts (readiness reports reuse chunk 2,048, tail 4, default drafts 3).

**Client.** llmprobe 0.6.18, the pinned bundle with the aggregate receipt patch (`bin/dist/llmprobe.mjs` `5013dcfadcee…`); Node.js 24.22.3.

## Conditions

- **Reasoning:** `off` requested on every arm. Both engines kept thinking under both llmprobe disable methods, so "off" means effort level 0 on the same render, not thinking disabled.
- **Matched rendering:** input token counts were identical across arms in every scenario (71 initial coding, 72 sustained coding, 40 novel prose, 498 at about 512, 4,019, 8,252 and 16,335 on the context ladder, 8,768 repeated-sentence prefill, 7,514 to 7,600 unseen-text prefill, 8,800 exact IDs, identical across arms).
- **Drafting:** each engine's default. Stark requested the engine default and the worker reported default drafts 3 with the history planner; MLX-Serve's draft was on by default and not forced.
- **Decode length:** MLX-Serve honours `ignore_eos` and `min_tokens` and decodes 192 tokens; the Stark gateway does not, so Stark decode rows follow the model's own stop and cover different amounts of work.
- **Host:** Apple M5 Max (Mac17,7), 128 GiB, macOS 26.7.2 (25G332), AC power; the adapter negotiated 94 W of a rated 140 W; battery 99 percent at start and 94 percent (charging) at end; process thermal state 1 at the start and before every session; no competing processes at start or end; no sleep events; Low Power Mode off. Host notes recorded by the tool: the adapter wattage and the thermal state.

## Results

Median tok/s over the run with range and count; paired ratio = median of per-block ratios with its range over the three blocks; a standing is ahead or behind only when all three blocks agree.

| Scenario (standard rows) | Stark 8-bit | MLX 8-bit | Stark / MLX, 8-bit pair | Standing | Stark original | MLX original | Stark / MLX, original pair | Standing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Initial coding | 85.1 (81.7–91.1, n 9) | 84.5 (69.6–88.5, n 9) | 1.02 (1.00–1.10) | unresolved | 81.2 (77.6–84.5, n 9) | 76.1 (70.8–89.9, n 9) | 1.07 (1.00–1.07) | ahead |
| Sustained coding | 98.6 (78.1–102.4, n 9) | 93.4 (85.5–95.9, n 9) | 1.06 (0.94–1.08) | unresolved | 88.7 (79.6–91.4, n 9) | 81.6 (73.0–82.2, n 9) | 1.11 (1.08–1.12) | ahead |
| Novel prose | 74.0 (69.1–80.8, n 9) | 84.4 (78.5–88.1, n 9) | 0.86 (0.84–0.93) | behind | 68.1 (64.4–76.5, n 9) | 69.4 (66.5–74.4, n 9) | 0.96 (0.96–0.99) | behind |
| About 512 tokens | 85.6 (78.6–95.2, n 3) | 85.6 (85.0–86.4, n 3) | 1.00 (0.91–1.12) | unresolved | 82.5 (71.5–84.6, n 3) | 79.1 (76.4–79.6, n 3) | 1.06 (0.90–1.08) | unresolved |
| About 4K | 106.5 (94.9–117.0, n 3) | 92.2 (86.2–103.0, n 3) | 1.03 (1.03–1.36) | ahead | 99.9 (92.0–105.2, n 3) | 84.1 (82.4–84.7, n 3) | 1.18 (1.09–1.28) | ahead |
| About 8K | 99.0 (94.1–104.4, n 3) | 97.1 (84.9–100.5, n 3) | 1.04 (1.02–1.11) | ahead | 97.5 (89.6–98.1, n 3) | 89.5 (84.6–101.0, n 3) | 1.06 (0.97–1.09) | unresolved |
| About 16K | 89.7 (85.0–91.0, n 3) | 85.8 (84.6–87.1, n 3) | 1.03 (0.99–1.08) | unresolved | 91.5 (87.5–94.7, n 3) | 88.2 (83.2–93.6, n 3) | 1.07 (0.93–1.10) | unresolved |
| Repeated-sentence prefill, 8,768 tokens | 1,897.8 (1,688.1–1,986.4, n 9) | 1,664.7 (1,549.7–1,854.9, n 9) | 1.13 (1.07–1.14) | ahead | 2,053.4 (1,694.3–2,140.1, n 9) | 1,925.3 (1,771.0–2,097.1, n 9) | 1.02 (0.92–1.07) | unresolved |
| Exact 8,800-ID prefill, client rate | 2,013.9 (1,990.2–2,131.5, n 3) | 2,092.3 (1,958.7–2,227.9, n 3) | 0.96 (0.96–1.02) | unresolved | 1,984.6 (1,663.9–2,104.1, n 3) | 2,030.4 (1,779.8–2,127.1, n 3) | 0.98 (0.93–0.99) | behind |

The exact 8,800-ID row is also reported at engine phase time with the same ratios and standings (8-bit 2,014.9 against 2,093.2; original 1,985.6 against 2,031.2). No samples were excluded.

Tokens per step (Stark: tokens per drafted round; MLX-Serve: as it reports), 8-bit pair: initial coding 2.21 / 1.44 (Stark acceptance 0.55), sustained 2.34 / 1.43 (0.47), novel prose 1.85 / 1.22 (0.39), 512 2.29 / 1.39 (0.53), 4K 3.15 / 2.70 (0.74), 8K 2.95 / 2.95 (0.73), 16K 2.91 / 2.59 (0.69).

**Supplementary rows** (reported only, no standing), Stark / MLX-Serve paired ratio on the 8-bit pair: unseen-text prefill 1.03 (1.03–1.13; 1,963.8 against 1,856.0 tok/s); counting-ceiling rungs 1.01 at 512 (0.97–1.31), 1.09 at 4K (0.93–1.37; Stark reused 2,048 cached tokens there by the scenario's design), 1.06 at 8K (1.03–1.10), 1.01 at 16K (0.83–1.01; Stark reused 14,336). Original pair: unseen-text prefill 1.03, counting ceilings 1.18, 1.06, 1.07 and 1.05. The burst was not run.

**Best single-user decode** (any scenario): Stark 8-bit 106.5 tok/s at about 4K; MLX 8-bit 97.1 at about 8K; Stark original 99.9 at about 4K; MLX original 89.8 at the 8K counting ceiling (89.5 at about 8K cold).

## Quality

Attached after the run from the coordinator's final-build reports (private `runtime/final-check/qc-orig-*.json` and `qc-a8-*.json`), bound to the worker binary above; probe `inkling_slice` `8881eeec47596187e6c20eb1e6e9bc1b85ac0672d7a38f51d8a4299e6e96ba7f` built from the same tree. Engine diagnostic on natural text: 10 inputs (repository prose and source, 300 to 8,000 tokens), 258 teacher-forced rows each, pooled over 2,580 rows, window 256. Reference: the exact quantized prefill path of the original artifact. The corpus has no recorded SHA-256 (the quality result's `corpus.sha256` field is empty); the ten source-report hashes are bound in the run directory.

| Arm | Verdict | Next-token accuracy (reference / arm) | Top-1 agreement: all; reference p ≥ 0.5; p ≥ 0.9 | KL reference to arm, nats (mean / max) |
| --- | --- | ---: | --- | ---: |
| `stark-affine8` | pass | 73.64 / 73.84 percent | 95.39; 99.28 of 1,934 rows; 99.93 of 1,396 | 0.01123 / 1.90432 |
| `stark-original` | pass | 73.64 / 73.57 percent | 95.58; 99.22 of 1,934; 99.93 of 1,396 | 0.01082 / 2.08280 |

These final-build figures differ slightly on every input from the earlier quantization-flip reports that [INKROUND-09](../model-experiments/2026-11-23-inkling-mini-expert-inputs-default.md) quotes; see its follow-up. The final-build figures belong to the committed code.

## Validity, failures and what was not run

- MLX-Serve reused 9 cached input tokens on the novel-prose cell, which is defined as cold, on both MLX-Serve arms. The finding stands; the effect on the 0.86 ratio is not quantified.
- MLX-Serve forces 192 decode tokens; Stark stops at end of sequence.
- The adapter negotiated 94 W of 140 W and the thermal state was 1 throughout. Both are host notes, not refusals.
- No arm failed. The 16-user burst was not run (out of scope). An earlier standard run started at 15:58 (`20261123-155828-inkling-2-mini`) was stopped for the owner's pause after two of twelve arm sessions and is unusable; the Mac then rebooted ([wind-down](../model-experiments/2026-11-23-inkling-mini-wind-down.md)).

## Reading and supersession

Five of the nine standard rows on the default pair are unresolved: one or more blocks disagreed. The one standing gap is novel prose, about 14 percent behind, which the owner set aside when optimization stopped; on that row Stark drafts 1.85 tokens per round at about 0.39 acceptance against MLX-Serve's mostly plain steps. MLX-Serve 8-bit's repeated-sentence prefill median (1,664.7 tok/s) was lower here than in the earlier single-repeat runs (1,909 to 2,220), so the 1.13 standing on that row depends on this run's MLX-Serve rate; the exact 8,800-ID prefill row, unresolved at 0.96, is the matched prefill cell. This record is the current Inkling 2 Mini comparison and supersedes the single-repeat engine-matrix runs 1 to 5 ([engine matrix](2026-11-22-inkling-mini-engine-matrix.md)) as the standing; their measurements are unchanged.
