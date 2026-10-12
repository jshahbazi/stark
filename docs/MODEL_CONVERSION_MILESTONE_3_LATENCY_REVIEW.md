# Independent milestone 3 latency technical review

Review completed: 2026-10-12. Baseline: `f6f1558d2aa05e65f580c71fa0ce71304ba4f0d7` on `codex/conversion-milestone-3`. The reviewer authored no production converter, profiling helper or benchmark coordinator. This review covers the [MLX coefficient-writer latency packet](MODEL_CONVERSION_MILESTONE_3_LATENCY.md). Earlier dense conversion and native qualification evidence retains its original source bindings.

## Disposition

**Accept the scoped sequential writer, exact artifact preservation and measured improvement over the frozen Stark baseline.** No unresolved blocking finding remains. All six selected complete-command medians improve by 15.61–38.04%, all ninety tensor/spec/config/tokenizer comparisons pass, and all thirty final-candidate receipts match the existing exact named-build native qualifications.

Five of six medians satisfy the raw latency hypothesis against external Lalamo. SmolLM3-3B 8-bit remains slower by median. Llama3.2-1B 8-bit and Gemma3-1B 8-bit have overlapping observed timing ranges with Lalamo; Gemma's 1.03× median ratio is marginal. These descriptive results grant no general latency guarantee, complete Phase 4/G4 acceptance, inference qualification or complete Lalamo replacement.

## Reviewed bindings

| Artifact | SHA-256 |
| --- | --- |
| Archived baseline converter sources | `779ec68a7a1c55edff0438ebe60afdca7194e3e6046679491cac06bcc6656aa1` |
| Archived baseline converter lock | `06d5e2a227ce9480adaa69501a6d7552c68c95e2d37dc0bd830d650ac98fae55` |
| Final candidate converter sources | `1e00fd5ee30af56dc8b9df7e0334130124b88ddad0cd8caccfb4624cf339836e` |
| Final candidate converter lock | `dde07746fc72585fd6d42168037aa7e72f4d7434100ab0daeabd010465c5ab63` |
| Final coefficient writer | `054e7d31a1d8d238c4398b80cf70c6ead219b1337b048484d0b9fe93b38433e6` |
| Thirty new focused tests | `184837c26f995e652f249bd36e1f602808d35859177159707f2ee19eb447c9b5` |
| Ninety-run complete-command benchmark | `78c1226d3dcc5e714e2c2905594ec5ecfeecdf0643965d2c7b1024b219d59b53` |
| Frozen benchmark inputs | `d0af4955e1c0870760ebcf729eb33c70095609e3fc164490bfbad9dbfbf9998f` |
| Preregistered benchmark objectives | `068fcf6710c9070cc1d429c54575ae248da34bd438ffae2149bd1094d78224f6` |
| Benchmark coordinator | `fd5d280463cda19235841f045942c28b08bf4005844892b13bb780154a9b5538` |
| External reference runner | `e785c2272366f523f79237be847549f76ea07bdc44b20cf69b7d76b6e38ad010` |
| Final baseline diagnostic profiles | `4e5872ca9a0fa6fd4a21caab52fec740f0ac2f861e0bf762c3dd613b567678ee` |
| Final candidate SmolLM3 8-bit diagnostic | `15715be680c33506c32255b10c23d7bc37125aba9eb6b55fddbffc84ff4d74b6` |
| Candidate diagnostic helper | `82482e05190acc80f75cec650aa21084053ed2c09b88dd76c74295abf8237c24` |
| Isolated coefficient experiment | `cfc6a9df95a63513c158260b0b4fe9eaeda6a2d48f96989ad2d3048cbe4796a7` |
| Isolated experiment coordinator | `33ad966c694e2a66fcd9bf277bfdc641802e37654d4f68e300fceafd3ddd743f` |
| Independent aggregate review checks | `19a6122d2ca95b816fa287418ef7436d4412def99749b3148f06480640214c0d` |
| Unchanged catalog | `d855e4c02ada48cca0062e7c273d3d7372b67b4ba70379ef3cf275acd9b78a3a` |
| Unchanged nineteen-record native ledger | `26074c55ae93a5ef0a7aba15a676668ba4f0d4fa76466e23c2076b3515b8beac` |

The reviewer recomputed these final source, lock, writer, test, benchmark, freeze, coordinator, objectives, reference-runner, catalog and ledger bindings. Baseline and candidate tooling match the freeze and the benchmark's post-run check. Selected private manifest file hashes match the frozen inputs. Catalog and native-ledger bytes match the baseline commit. Parsing both lock files confirmed that only the editable package version changes from 0.5.0 to 0.5.1; every dependency record is identical.

The independent aggregate check recomputed all medians, ranges, timing ratios, RSS reductions and hypothesis flags from the ninety observations. It verified all ninety parity records, and all sixty baseline/candidate receipts against complete previous Stark artifact hashes and the exact qualification matcher. This includes thirty final-candidate conversions. The external comparison requires exact tensor payloads, shapes, metadata/specs, configuration and tokenizer parity; Stark additionally preserves its previous whole-file artifact bytes.

## Technical assessment

**Ordering and numerical behavior.** The writer traverses output groups in ascending order and fills each group's complete fused row sequence before adding global padding. The reviewed uneven fusion fixture totals thirteen rows and pads once to sixteen across tiny, partial-band and full-band workspaces. Source shards, per-source offsets and source ordering retain their prior interpretation. Header serialization, tensor ordering, packed weight codes and coefficient shapes remain unchanged. Casting still uses the frozen portable BF16/F16/F32 policy, including exceptional NaN payload/sign handling.

**Workspace.** A retained output band is reserved before selecting input rows. Full-width reads retain a bounded contiguous input buffer. Narrow contiguous reads additionally reserve the selected-column copy; strided reads reserve an input tile and one read row. Both branches reserve the cast result, native scratch and the F16 NaN mask. The small-workspace path emits one group in bounded row tiles and pads after all fused sources. The reviewed formulas cover the simultaneous live buffers, and the reported transformation peak remains within the requested budget. This is a transformation-workspace bound; it does not cap Python, tokenizers, allocator or file-I/O process memory.

**Hashing and failure handling.** No destination seek occurs in the coefficient writer, so it preserves the shared writer's incremental complete-file SHA-256. Full output hashing remains present. Other operations that need random writes retain the shared fallback behavior. Truncated reads fail; opening each source still checks its file signature, and the shared writer rechecks all original tensors after writing. Both complete source-content verification passes, tied-head validation, exact qualification matching, staging cleanup and atomic publication remain enabled.

The attributed upstream conventions and MIT license remain intact. The converter gains no Lalamo dependency or fallback. No Rust/Metal engine, native build input, catalog, loader qualification or model distribution changes.

## Validation and performance

The reviewer independently ran **811 integrated focused tests** covering the new sequential writer, shared MLX operators, frozen external numerical exports, dense and Gemma MLX paths, and exceptional casts. They passed in 5.26 seconds. Meaningful new checks cover sequential writes with an existing output prefix, retained whole-file inline hashing, contiguous reads, uneven fusion/global padding, all nine floating-point cast pairs, truncation and source mutation. Existing frozen tests also exercise sixteen-byte workspace and all four accepted group sizes.

The coordinator reports **1,655 passed and 39 skipped** with nineteen supplied private metadata references. Thirty focused tests were added. Ruff/format, lock consistency, repository policy, seventeen CI-policy tests and diff checks passed. No engine build or native inference was required for this writer-only packet.

The benchmark preregisters five repetitions per backend and tuple, comparing the immutable prior converter, the frozen candidate and clean external Lalamo at `831accb8252a6b6606c378048caeb44178f1a050`. Runs are serial fresh direct Python processes with warmed selected inputs and rotating/reversed order. Timed intervals include startup, metadata, writing, output hashing, complete candidate source verification, final checks and receipt publication. Setup, downloads, warming and parity comparisons are excluded. The external reference performs weaker primary-source verification and receipt work, which remains explicit in the comparison.

| Group-64 BF16 tuple | Prior Stark median | New Stark median | Lalamo median | Raw latency hypothesis |
| --- | ---: | ---: | ---: | --- |
| Llama3.2-1B 4-bit | 2.027 s | 1.256 s | 2.397 s | Passed |
| Llama3.2-1B 8-bit | 2.697 s | 2.202 s | 2.604 s | Passed; ranges overlap |
| SmolLM3-3B 4-bit | 4.302 s | 2.779 s | 3.377 s | Passed |
| SmolLM3-3B 8-bit | 7.019 s | 5.083 s | 4.233 s | Failed |
| Gemma3-1B 4-bit | 2.111 s | 1.551 s | 2.580 s | Passed |
| Gemma3-1B 8-bit | 2.919 s | 2.464 s | 2.537 s | Passed; marginal, ranges overlap |

All six tuples pass workspace, sampled process-tree RSS, source/tooling and artifact hard checks. Candidate median sampled RSS is 90.75–96.60% below Lalamo, satisfying the memory hypothesis for each tuple. RSS sampling every ten milliseconds can miss brief peaks and count shared pages more than once. Five observations per backend on one warm-cache host support the reported medians and ranges; no statistical significance, cold-cache or download-speed result is claimed.

The baseline profiles separately establish one full-output hash reread in each previously slower tuple and identify coefficient transformation time. The isolated synthetic experiment reports writer/flush/hash timings with exact expected output checks, bounded workspace, one warmup and five alternating observations. Its large coefficient ratios exclude interpreter startup, full source verification and fsync, so they are not complete conversion gains.

The post-benchmark SmolLM3 8-bit diagnostic binds the final source/lock and its helper, retains the exact qualified output hash, and reports zero output-hash fallback rereads. The reviewer independently checked those bindings and aggregate flags. In this single warm run, coefficient transforms took 0.146 s, initial and final source verification took 1.152 s and 1.142 s, and copy operations including inline hashing took 2.007 s. This identifies remaining costs after coefficient export improves. Instrumented stage timings exclude interpreter startup and are separate from the five-repeat complete-command result; they do not establish an additional speedup or justify removing source verification.

## Qualification and remaining scope

Byte-identical output reuses the six existing exact BF16 source/artifact qualifications for their named optimized Rust 1.98.1 worker. The benchmark invokes no native worker, scoring or generation, and adds no qualification record. These records prove prior load readiness for the named build only. They grant no new inference quality, throughput, other checkpoint/group/dtype, future binary or family-wide execution claim.

SmolLM3-3B 8-bit complete-command latency remains open work. Broader model/format coverage, unqualified tuples, strict process-memory enforcement and complete Phase 4/G4 replacement gates remain open. Operational artifacts and diagnostics remain private and ignored; the published evidence contains aggregate results, public identities and hashes.
