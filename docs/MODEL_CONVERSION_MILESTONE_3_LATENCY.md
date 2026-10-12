# Milestone 3 MLX conversion latency

Date: 2026-10-12. Baseline: `f6f1558d2aa05e65f580c71fa0ce71304ba4f0d7` on `codex/conversion-milestone-3`. This packet optimizes the existing MLX coefficient writer. The [dense continuation](MODEL_CONVERSION_MILESTONE_3_DENSE.md) retains its original implementation, benchmark and qualification bindings.

## Scope and acceptance

The six selected group-64 BF16 tuples are Llama3.2-1B Instruct, SmolLM3-3B and Gemma3-1B IT at 4 and 8 bits. The catalog and nineteen exact named-build native qualifications remain unchanged. This packet adds no model, architecture, quantization scheme or inference qualification.

Hard acceptance requires byte-identical weights, exact tensor/spec/config/tokenizer parity, unchanged selected source content and tooling, exact existing qualification matches, transformation workspace at most 8 MiB, and sampled candidate process-tree RSS below 1 GiB. Performance hypotheses require a candidate complete-command median no greater than the frozen baseline, a candidate median no greater than pinned Lalamo, and candidate median RSS at most half Lalamo for each measured tuple. Hypotheses are recorded separately from correctness acceptance.

## Implementation

The previous writer reads each coefficient tile row separately and seeks through the output to transpose it. Its first seek invalidates incremental hashing, requiring a complete output reread. The new writer emits bounded bands in group/output order and concatenates the complete fused row sequence before padding. It reads complete source rows contiguously when workspace permits; narrow bands reserve a separate selected-column copy. Smaller budgets retain bounded strided reads. The output header, tensor order, packed codes, casts and padding bytes remain identical.

Workspace accounting includes the live output band, input tile or read buffer, cast output and reserved native scratch/NaN mask. Small budgets use a bounded sequential row path. Full initial and final source hashing, file signatures, tied-head validation, atomic publication and receipt qualification remain enabled. Package version is 0.5.1; dependency versions are unchanged. The Rust/Metal engine and native build inputs are unchanged.

## Measurement protocol

Four fresh-process diagnostic profiles of the frozen baseline isolated the coefficient transform and output-hash reread. Inputs were warmed first; these single-run stage timings exclude interpreter startup and are not speedup measurements.

| Slower baseline tuple | Coefficient transforms | Output SHA-256 reread | Reread bytes |
| --- | ---: | ---: | ---: |
| Llama3.2-1B 8-bit | 0.458 s | 0.476 s | 1,313,294,064 |
| SmolLM3-3B 4-bit | 1.315 s | 0.625 s | 1,730,352,904 |
| SmolLM3-3B 8-bit | 1.245 s | 1.175 s | 3,267,827,584 |
| Gemma3-1B 8-bit | 0.531 s | 0.380 s | 1,062,855,208 |

The final profiles bind the baseline source/lock and helper hashes. Earlier incomplete profiling attempts remain private and excluded. Eliminating the reread moves full output hashing into sequential writes; it does not remove SHA-256 work. Both complete source-verification passes remain measurable costs. A separate read-only investigation found that locked CPython already uses native OpenSSL SHA-256; this packet introduces no alternate hash provider.

The new private coordinator compares the exact archived baseline, the final candidate and clean external Lalamo at `831accb8252a6b6606c378048caeb44178f1a050`. It preregisters five repetitions per backend and tuple. Runs are serial fresh direct Python processes with warm selected inputs and rotating/reversed backend order. The measured interval covers the complete command, including startup, metadata, writing, hashing, verification, final checks and receipt publication. Preparation/downloads, warming and parity comparison are excluded.

Every output is compared outside the timed interval with the frozen independent reference artifacts. Baseline and candidate commands require the existing exact native qualification; matching bytes preserve its named-build load-only scope. No native scoring or generation is performed. RSS is sampled every 10 ms; it can miss brief peaks and double-count shared pages. The 8 MiB transformation limit is separate from process RSS. Lalamo performs weaker primary-source verification and receipt work, so the comparison describes those different complete command contracts.

Operational manifests, models, tokenizer/config files, profiling helpers, benchmark receipts and outputs remain ignored and private. Public evidence consists of aggregate results, public identities/revisions and hashes. Historical results are preserved separately when the converter or measurement tooling changes.

## Validation and results

**Accept the scoped sequential-writer change, exact byte preservation, bounded workspace and observed improvement against the frozen implementation.** The [independent review](MODEL_CONVERSION_MILESTONE_3_LATENCY_REVIEW.md) records its final assessment. All 90 complete commands passed exact independent artifact parity; all 60 baseline/candidate commands also matched the previous complete artifact hashes and stored exact native qualification. Every candidate retained at most 8 MiB transformation workspace and sampled tree RSS below 1 GiB. Source, converter inputs, locks, measurement tooling and clean pinned reference remained unchanged.

The six complete-command medians improved **15.6–38.0% against the frozen baseline**. Five raw medians are below Lalamo. SmolLM3 8-bit remains **20.1% slower** than Lalamo, despite its 27.6% improvement against the old writer. Gemma 8-bit's passing median advantage is 2.9%; its observed ranges overlap, as do Llama 8-bit's. These are descriptive outcomes from five observations per backend and tuple, with no significance claim.

| Group-64 BF16 tuple | Frozen baseline median | Candidate median | Lalamo median | Lower time vs baseline | Lalamo/candidate ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama3.2-1B 4-bit | 2.027 s | 1.256 s | 2.397 s | 38.0% | 1.91× |
| Llama3.2-1B 8-bit | 2.697 s | 2.202 s | 2.604 s | 18.4% | 1.18× |
| SmolLM3-3B 4-bit | 4.302 s | 2.779 s | 3.377 s | 35.4% | 1.22× |
| SmolLM3-3B 8-bit | 7.019 s | 5.083 s | 4.233 s | 27.6% | 0.83×; failed |
| Gemma3-1B 4-bit | 2.111 s | 1.551 s | 2.580 s | 26.5% | 1.66× |
| Gemma3-1B 8-bit | 2.919 s | 2.464 s | 2.537 s | 15.6% | 1.03× |

All six observed baseline/candidate timing ranges are separated. Llama 4-bit, SmolLM3 4-bit and Gemma 4-bit also have separated candidate/Lalamo ranges. SmolLM3 8-bit's reference range includes a slow first observation; it remains in the five-run result and the failed hypothesis remains visible.

| Tuple | Frozen baseline range | Candidate range | Lalamo range |
| --- | ---: | ---: | ---: |
| Llama3.2-1B 4-bit | 1.741–2.114 s | 1.229–1.285 s | 2.182–2.761 s |
| Llama3.2-1B 8-bit | 2.564–2.790 s | 2.016–2.332 s | 2.277–2.712 s |
| SmolLM3-3B 4-bit | 4.260–4.621 s | 2.732–2.978 s | 3.352–3.737 s |
| SmolLM3-3B 8-bit | 6.636–7.638 s | 4.858–5.477 s | 3.751–6.698 s |
| Gemma3-1B 4-bit | 2.010–2.184 s | 1.534–1.668 s | 2.395–3.835 s |
| Gemma3-1B 8-bit | 2.909–3.337 s | 2.405–2.859 s | 2.429–3.203 s |

Candidate median sampled tree RSS was **90.7–96.6% lower than Lalamo**; all six median-memory hypotheses pass. The optimization retains approximately the previous converter's RSS.

| Tuple | Frozen baseline RSS | Candidate RSS | Lalamo RSS | Lower candidate RSS |
| --- | ---: | ---: | ---: | ---: |
| Llama3.2-1B 4-bit | 275.8 MiB | 275.6 MiB | 4,720.6 MiB | 94.2% |
| Llama3.2-1B 8-bit | 168.2 MiB | 168.7 MiB | 4,965.1 MiB | 96.6% |
| SmolLM3-3B 4-bit | 273.4 MiB | 275.2 MiB | 5,966.4 MiB | 95.4% |
| SmolLM3-3B 8-bit | 275.3 MiB | 275.0 MiB | 7,247.7 MiB | 96.2% |
| Gemma3-1B 4-bit | 472.4 MiB | 472.0 MiB | 5,101.3 MiB | 90.7% |
| Gemma3-1B 8-bit | 468.2 MiB | 468.2 MiB | 5,106.9 MiB | 90.8% |

Host: Apple silicon arm64, 18 logical CPUs, 128 GiB RAM, macOS 26.7.1. Python is 3.13.13; locked dependencies include NumPy 2.5.3, ml-dtypes 0.6.0, tokenizers 0.23.3, Hugging Face Hub 2.2.0 and psutil 7.2.2. This fresh comparison uses the archived old code in the same batch as the new code; earlier three-repeat medians retain their original host/tooling records and are not substituted for this baseline.

The complete locked converter suite passed **1,655 tests with 39 skipped** using nineteen supplied private metadata references. The independent reviewer passed **811 focused tests**, including frozen exceptional coefficient casts, all accepted MLX group sizes and float pairs, shard fusion, uneven global padding, truncation, source signatures, 16-byte workspace and inline output hashing. Thirty focused tests were added. Ruff/format, lock consistency, repository policy, seventeen CI-policy tests and diff checks passed.

An isolated coefficient-writer experiment used five alternating measurements after one warmup per backend. Realistic Llama, SmolLM3 and Gemma3 MLP planes measured 10.83×, 8.24× and 12.04× writer/flush/output-hash ratios against the archived implementation; a larger banded stress case measured 2.07×. All outputs matched exactly and workspace stayed at most 8 MiB. This experiment excludes complete source verification, startup and fsync, so its ratios do not describe full model conversion.

A fresh candidate SmolLM3 8-bit diagnostic profile, run after the complete benchmark, measured 0.146 s in coefficient transforms and **zero fallback output-hash rereads**. Initial resolution/source validation and final complete verification took 1.152 s and 1.142 s; copy operations including inline SHA-256 took 2.007 s. Its weights hash matches the existing qualified artifact. The remaining costs are largely source verification and copying/hashing. This single-run profile excludes startup and is separate from the definitive five-repeat complete-command result.

## Source and evidence bindings

| Artifact | SHA-256 |
| --- | --- |
| Archived baseline converter sources | `779ec68a7a1c55edff0438ebe60afdca7194e3e6046679491cac06bcc6656aa1` |
| Archived baseline converter lock | `06d5e2a227ce9480adaa69501a6d7552c68c95e2d37dc0bd830d650ac98fae55` |
| Final candidate converter sources | `1e00fd5ee30af56dc8b9df7e0334130124b88ddad0cd8caccfb4624cf339836e` |
| Final candidate converter lock | `dde07746fc72585fd6d42168037aa7e72f4d7434100ab0daeabd010465c5ab63` |
| Final coefficient writer | `054e7d31a1d8d238c4398b80cf70c6ead219b1337b048484d0b9fe93b38433e6` |
| Thirty focused latency tests | `184837c26f995e652f249bd36e1f602808d35859177159707f2ee19eb447c9b5` |
| Preregistered benchmark objectives | `068fcf6710c9070cc1d429c54575ae248da34bd438ffae2149bd1094d78224f6` |
| Frozen benchmark inputs | `d0af4955e1c0870760ebcf729eb33c70095609e3fc164490bfbad9dbfbf9998f` |
| Benchmark coordinator | `fd5d280463cda19235841f045942c28b08bf4005844892b13bb780154a9b5538` |
| Final ninety-run benchmark | `78c1226d3dcc5e714e2c2905594ec5ecfeecdf0643965d2c7b1024b219d59b53` |
| Exact external reference runner | `e785c2272366f523f79237be847549f76ea07bdc44b20cf69b7d76b6e38ad010` |
| Final baseline profiling aggregate | `4e5872ca9a0fa6fd4a21caab52fec740f0ac2f861e0bf762c3dd613b567678ee` |
| Final profiling helper | `1a52722d37f9bc443e9c4eef279ed280b4de8fd7b47c34f29a0e96e5358488e9` |
| Post-benchmark candidate diagnostic profile | `15715be680c33506c32255b10c23d7bc37125aba9eb6b55fddbffc84ff4d74b6` |
| Candidate diagnostic helper | `82482e05190acc80f75cec650aa21084053ed2c09b88dd76c74295abf8237c24` |
| Isolated coefficient experiment | `cfc6a9df95a63513c158260b0b4fe9eaeda6a2d48f96989ad2d3048cbe4796a7` |
| Isolated experiment coordinator | `33ad966c694e2a66fcd9bf277bfdc641802e37654d4f68e300fceafd3ddd743f` |

The source identities, immutable checkpoint revisions, manifest content hashes and exact artifact hashes match the six tuples in the [preserved dense qualification packet](MODEL_CONVERSION_MILESTONE_3_DENSE.md). Llama metadata retains its recorded canonical/transport provenance. The catalog hash is `d855e4c02ada48cca0062e7c273d3d7372b67b4ba70379ef3cf275acd9b78a3a`; the unchanged nineteen-record native ledger hash is `26074c55ae93a5ef0a7aba15a676668ba4f0d4fa76466e23c2076b3515b8beac`.

## Remaining work

The remaining SmolLM3 8-bit raw latency objective is open. Further optimization must retain complete source verification, identical artifacts and truthful workspace accounting. Broader sources, cold-cache/download timing, other hosts, hard process-memory enforcement and native execution/quality remain separate acceptance work. The 54 implemented built-ins plus four extensions, 56 unimplemented built-ins and 39 implemented identities without exact native qualification are unchanged. Full Phase 4/G4 and complete Lalamo replacement remain open; the runtime has no Lalamo fallback.
