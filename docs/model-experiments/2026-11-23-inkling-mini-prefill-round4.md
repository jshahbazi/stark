# 20261122-INKFASTPRE-04 · Inkling 2 Mini prefill round 4: routing, drafting overhead, worker sequence, command width, expert ceiling and rejected kernels

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date | `20261122-INKFASTPRE-04` (series registered 2026-11-22) / 2026-11-23 / 2026-11-23 |
| Kind | Kernel and scheduling changes, measurements and rejected kernels in prefill |
| Attempt state | Round 4 closed and verified; later items in this record are measured or rejected; several follow-ups were still building when last reported and are marked not final |
| Result | Routing changes landed with identical hashes and no measurable speed change. Drafting overhead measured at 10 to 70 ms. Width API verified. Expert ceiling measured. Weight-stationary and staged dense kernels and stationary NVFP4 at 2,048 rows rejected. Global causal skip hash-identical with no gain. |
| Evidence scope | Engine diagnostic: hashes, self-consistency, instrumented stage profiles and a few uninstrumented walls. No new paired ABBA ratio for the landed routing change at 4K. No worker timing. |
| Promotion decision | Routing tiles, fused selection and the causal skip promoted (free, hash-identical); width API adopted by the worker under [INKWIDTH-01](2026-11-23-inkling-mini-wide-prefill-worker.md) |
| Disposition | Adopted on branch, uncommitted on top of `f1666090`; not installed |
| Predecessor / related | [20261122-INKFASTPRE-03](2026-11-23-inkling-mini-prefill-round3.md), [prefill attribution](../benchmarks/2026-11-22-inkling-mini-prefill-attribution.md), [MLX-Serve source comparison](../benchmarks/2026-11-23-inkling-mini-mlx-serve-source.md), [differences page](inkling-mlx-serve-differences.md) |

Original checkpoint (`6e7c8248…`), l8800 oracle tokens at context 9,216 unless stated; host as in the other 2026-11-23 records (94 W adapter, shared lock). Baseline binary `final4` (INKFASTPRE-03 final, SHA-256 prefix `6ab9a824f4e7`); round-4 final `r4f`; routing-only `r4b`; close-out on the coordinator's `morning1-build`. Private outputs under `runtime/prefill-fast/` (`r4`, `r4-route`, `r4-final`, `r4-close`, `ceiling`, `dense-stat`, `nv-stat`, `dense8k`).

## Routing (adopted, no measurable gain)

- Default prefill selection uses the existing fused kernel `ink_singleton_route_weights_parallel`: one dispatch replaces select, LogAddExp, negate, softmax and scale.
- The router projection runs on `ink_prefill_router_tiles`: the same 16 × 32 × 16 F32 tensor steps, K order and split as `ink_prefill_router_source`, with each weight fragment shared by four 16-row tiles.
- Prefill state and all four seed and teacher logit rows hash identically to `final4` at 4K and 8,800; the router probe found 0 changed selections in 1,768,704 rows.
- Timing: 8,800 tokens 1.006× (0.960 to 1.053), 5/8 wins. The 4K pairs ran but the batch stopped before its summary printed (one arm 2,221 against 1,933 ms with a 3,226 ms outlier), so there is no clean 4K ratio. Fused selection alone (`r4b`): 1.001× at 4K, 1.005× at 8,800. Routing time is not in the selection chain.

## Drafting overhead in prefill (measured, no change)

In one loaded session at 8,800 tokens, drafting on against off: 5,320 against 5,252 ms median wall, GPU 5,241 against 5,229 ms; command wall minus GPU about 9 ms in both arms and no time between commands. The close-out profile on `morning1-build`: GPU 4,942 ms drafting on against 4,865 off; an 84 ms gap seen once was a first-run outlier; all per-stage deltas under 40 ms. Engine-side drafting overhead is therefore 10 to 70 ms after the packed draft commit of INKFASTPRE-03. The coordinator records the earlier "about 0.5 s" figure as a comparison across power states. How much of INKFASTPRE-03's 5,722 → 4,679 ms packed-commit gain was real is not settled; see conflict C1 on the [differences page](inkling-mlx-serve-differences.md#conflicts-in-the-sources).

## Worker call sequence (measured)

New diagnostic `--profile-worker-sequence` replays the worker's engine call sequence for one request and times each call. Per request outside the bulk commands: `set_context` 32 to 37 ms (first request only), snapshot 9 to 13 ms, 3-row tail 56 to 66 ms, seed step 24 to 28 ms. Plain, resized-from-4,352, draft-loaded and full worker setups all ran 4.4 to 5.8 s with identical seed hashes. The first request after load is consistently faster than back-to-back requests, which points to the power-limited GPU rather than an idle ramp, so a warm-up at admission would not recover it. The decode workstream's reading: most of the gap between the worker's 5.16 s and the diagnostic's about 4.2 s is different power-state windows plus about 0.1 s of seed, tail and snapshot. Elementwise passes: the profile's "other" bucket (433 ms) already includes attention inputs (257) and expert inputs (143), so the real elementwise total is about 430 ms, not 822.

## Command width API (adopted)

`Session::set_batched_prefill_width(enabled, source_allowed, rows)` with rows in `Session::SOURCE_COMMAND_WIDTHS = [2048, 4096, 8192]` reallocates the packed workspace before ingest; `Session::batched_prefill_memory_bytes_at(source_allowed, rows)` returns 1,344, 2,317 and 4,263 MiB (measured 1,133, 2,106 and 4,052 MiB plus the original 211 MiB margin). Snapshot alignment stays at 2,048. Verified on `morning1-build` with `--profile-worker-sequence`, setups plain, wide8192 and wide4096 alternated twice at 8,800 tokens: commands `[2048 × 4, 604]`, `[4096, 4096, 604]` and `[8192, 604]`; the 8,192-boundary snapshot took 10 to 12 ms; seed-logit hashes identical across widths and requests. Timing in that run was dominated by drift (about +10 percent from the first to the second request per process), so no ranking can be read from it. Earlier paired screens: 8,192 against 2,048 rows 1.033× (8/8); 4,096 against 2,048 1.021× (0.886 to 1.164, 5/8). The worker's use of the API is [INKWIDTH-01](2026-11-23-inkling-mini-wide-prefill-worker.md).

## Close-out checks on `morning1-build`

Drafting on equals off at 73, 4,150, 8,800 and 16,000; snapshot and boundary replays and run-to-run identity at 73, 4,150 and 16,000; split consistency within each width class on l8800, b12000 and l16000 (exact control differs, as expected). The routing-only build passed the full set including split consistency.

## Expert-product ceiling (measured)

`--expert-ceiling <target> <dir> <prompt> <context> <visit> <reps>` captures one real routed NVFP4 visit from a 2,048-row l8800 command and times kernel variants on identical inputs, 10 repetitions each; visits 5, 20 and 40 agree within 3 percent. Visit 20, ms per visit:

| Variant | Gate/up | Down | Sum |
| --- | ---: | ---: | ---: |
| Production `grouped_fast` | 6.25 (27.8 TFLOPS-equivalent) | 2.93 (29.7) | 9.2 |
| Weights pre-decoded to BF16 in device memory | 6.67 | 3.17 | 9.8 |
| Decode only on the first K step, activations from device | 4.33 | 1.86 | 6.2 |
| Tensor work only, activations from threadgroup memory | 3.13 (55) | 1.57 (55) | 4.7 |
| Decode only, no tensor product | 3.16 | 1.62 | 4.8 |

Production is about tensor work plus decode back to back (4.7 + 4.8 = 9.5 against 9.2): the double-buffered decode does not overlap the tensor work. Reading pre-decoded weights from device is no faster than decoding them, so the lever is overlap and fewer decodes, not a cheaper decode. Activations from device cost about 1.2 ms of the 6.2 ms against threadgroup activations. FLOP-based floors at the measured 55 TFLOPS (estimates, not measured variants): NVFP4 products 2,085 ms measured against about 1,050; dense 687 against about 175; local attention 252 against about 80; global attention 290 against about 70 (the global floor assumes BF16, but the kernel is F32).

## Rejected kernels

| Kernel | Measurement | Hashes | Reason |
| --- | --- | --- | --- |
| Weight-stationary dense (`ink_dense_stationary`): each 64 × 64 weight block staged once per 256 rows, activations from device | 670 against 713 ms (incumbent first) | Identical | Activations re-read once per 64-column tile (48 times for the q projection) dominate; per-SIMD-group activation loads uncoalesced |
| Weight-stationary NVFP4 (`grouped_stationary.metal`): each 64-column × 32-K weight block decoded once per up to 256 rows of an expert, register epilogue, at 2,048-row commands | Gate/up 1,368 → 1,683 ms, down 666 → 831, wall 4.34 → 4.99 s | Identical (state `70acfe9f383d`) | Occupancy: about 128 rows per expert leave half of each 256-row workgroup idle. Left in the tree off (`FAST_STATIONARY = false`). Test at 8,192-row commands (about 512 rows per expert) queued. |
| Three dense kernels with both operands staged in threadgroup memory, at 8,192-row commands (8,192 + 604 + 3 rows) | Dense stage: incumbent `ink_dense_fast` 967 ms; `ink_dense_staged` (128 × 64, BK 64) 1,123; `_256x64` (BK 32) 1,220; `_256x128` (BK 32, 1,024 threads) 1,115. Uninstrumented walls 4,725 to 5,116 ms with large drift | All `70acfe9f383d` | Staging costs more than it saves; larger staged tiles do not help, so the dense stage is not limited by operand reuse. The coordinator's tile-traffic model (work per staged element 42.7, 51.2, 85.3) did not predict speed. |

The incumbent NVFP4 dot in that 8,192-row profile was 1,804 ms gate/up plus 1,333 ms down. The default dense constant is `("ink_dense_fast", 128, 64)`.

## Global causal skip (adopted, no gain)

The mask-free global attention pipeline is specialized with `do_causal` true, skipping key blocks past the causal diagonal. Those blocks already had a −∞ bias, so each contributed exactly zero with a rescale factor of exactly 1; replay confirmed the state hash `70acfe9f383d`. Global attention measured 311 against 301 ms, within noise: global keys are the whole prefix plus the query tile, so little lies past the diagonal. Kept as a free change. The exact reference keeps the full traversal with the mask.

## Unquantized expert inputs in prefill (in the tree, default unchanged)

Packed prefill calls the shared `ink_nvfp4_activation` and `ink_verify_half_activation`, so `INK_EXPERT_INPUTS_QUANTIZED` covers prefill too; ingress converts the binary16 values to BF16 exactly. When the define is 0, `register_grouped` also compiles `ink_nvfp4_activation_quantized` and `ink_verify_half_activation_quantized` for the exact reference, selected by `nvfp4_activation(runtime)` under `exact_prefill`; the serial single-row path in `mod.rs` does the same for prefill rows below the packed minimum. With the define at 1 nothing new is compiled. Paired prefill timing with the define at 0 was still owed when last reported; see [INKROUND-07](2026-11-23-inkling-mini-decode-round.md#20261123-inkround-07--unquantized-nvfp4-expert-inputs).

## Not final or not started

- In progress: BF16-query local attention (`ink_prefill_relative_fast`; the exact reference keeps F32 queries) and a half-precision global attention instantiation of the same MLX NAX kernel (half Q, K, V and O, F32 accumulation, bias as now). The noise-floor gate and the natural-text bar decide.
- Queued: joined dense projections; elementwise fusion; register-fragment activation loads in `--expert-ceiling`; stationary NVFP4 at 8,192 rows; the MLX-Serve kernel timing harness on the ceiling capture (not run).
- Parked: the tail fold (`PackedPrefill::SOURCE_TAIL_ROWS = MIN_ROWS`, so it does nothing).
- Dropped: a `relaxed_precision` test, because Stark's kernels already set it, as MLX-Serve's do.
- Incidents: a compile error from a private `register` in `grouped_pilot.rs` (fixed with `register_source`); the data volume filling at about 08:25 (a build failed; no source truncated).
