# Inkling 2 Mini: the Stark 8-bit artifact becomes the default

This record covers the catalog and preparation change that makes Stark's 8-bit dense artifact the Inkling 2 Mini default. It is a different artifact from the original checkpoint. Its evidence is listed per gate below; nothing here transfers qualification from the original checkpoint, and nothing in it was installed.

## 20261123-INKAFFINE8-04 · Default preparation and catalog route for the 8-bit artifact

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date | `20261123-INKAFFINE8-04` / 2026-11-23 / 2026-11-23 |
| Kind | Compatibility (catalog, preparation, upgrade and memory sizing); artifact identity |
| Attempt state | Completed for the converter, catalog and benchmark profile; qualification items listed as not done remain open |
| Result and acceptance criterion | Passed within scope: the default choice derives, verifies and publishes the 8-bit target whose SHA-256 equals the pinned `c9e14848…acddd`, and the published folder passed the worker smoke. Model-free tests pass. |
| Evidence scope | Model-free tests; one offline end-to-end preparation on the real pinned original; one worker smoke on the published folder. Earlier artifact evidence is cited, not regenerated. |
| Promotion decision | Owner decision, 2026-11-23: "Make the 8bit the default." Promoted to the default catalog and preparation choice. Qualification gates that are open stay open. |
| Disposition | Adopted on branch `claude/stark-model-optimization-7b576d`, uncommitted at registration. Not released, not installed. |
| Predecessor / related attempts | [INKAFFINE8-01 to -03](2026-11-22-inkling-mini-affine8.md), [INKROUND-07 and -08](2026-11-23-inkling-mini-decode-round.md), [MLX-Serve differences](inkling-mlx-serve-differences.md), [engine matrix run 4](../benchmarks/2026-11-22-inkling-mini-engine-matrix.md#20261123-inkengine-04--fourth-run-with-the-stark-8-bit-artifact) |

### Decision and what the default now is

The owner made the Stark 8-bit artifact the default on 2026-11-23. After this change:

| Choice | What it prepares | Published layout | Weight bytes | Quantization label |
| --- | --- | --- | --- | --- |
| `inkling2-mini` (default, the only Inkling item in Model Catalog) | Downloads and verifies the pinned original, derives the 8-bit artifact beside it, verifies the derived target against the pinned hash and size, publishes the derived folder, then removes the download | `inkling2-mini-affine8-dense` | 18,687,815,064 (target 18,052,013,648 plus draft head 635,801,416) | Original NVFP4 / MXFP8 experts; MLX affine 8-bit (group 64) dense projections, embedding and head |
| `inkling2-mini-original` (preparation command only) | The original checkpoint, prepared exactly as `inkling2-mini` was before this change | `inkling2-mini-original` | 20,161,820,584 | Original NVFP4 / MXFP8 |

The original is a separate artifact for reference and qualification work, not a switch on the default. It is not offered in the app's Model Catalog: both choices share one repository and revision, and the app matches installed models to catalog items by repository, revision and artifact mode, so a second item would show the other as installed. The benchmark profile declares the default too: `default_artifact = "affine8"` in `scripts/benchmark_profiles/inkling-2-mini.toml`; `original` remains a declared artifact and arm, and the matching classes are unchanged (`affine8` with the author 8-bit pack, `original` with the original checkpoint).

### Implementation

- `tools/converter/stark_convert/inkling_prepare.py`: `CHOICE = "inkling2-mini"` (default), `ORIGINAL_CHOICE = "inkling2-mini-original"`, `AFFINE8_SOURCE` (the original's download facts with the derived folder's weight bytes and quantization label and `derived_from`), `AFFINE8_QUANTIZATION`, `AFFINE8_EXTRA_RESIDENT_BYTES` (1,670,000,000, mirroring the worker) and `resident_weight_bytes(config)`.
- `tools/converter/stark_convert/hybrid_prepare.py`: `SOURCES` maps the default to `AFFINE8_SOURCE` and the original choice to the original source. `prepare` downloads the base exactly as before (same patterns, revision and resumable staging marker, so a download started before this change resumes into the default) and requires free space for the download plus the derived folder plus 4 GiB up front. `_derive_inkling_affine8` checks free space again (derived bytes plus 4 GiB, failing with `ENOSPC`, which the app reports as a full disk), writes into an owned `.native-derive` folder, verifies the target hash and size and the prepared identity, and only then is the folder published with the existing no-replace rename; the download is removed after publication. `upgrade_inkling_affine8` and the command-line flag `--upgrade-8bit` replace a prepared original folder in place: derive beside it into an owned `.affine8-upgrade` folder, verify, exchange the two atomically (`renamex_np` with `RENAME_SWAP` on macOS, `renameat2` with `RENAME_EXCHANGE` on Linux), then remove the original. Running it on a folder that already holds the 8-bit artifact changes nothing.
- `tools/converter/stark_convert/inkling_affine8_artifact.py`: the unused `CHOICE = "inkling2-mini-affine8"` is removed; `LAYOUT` and `QUANTIZATION` come from `inkling_prepare`. `build` reads `source-config.json` when it exists, so a prepared original folder can be its input.
- `tools/converter/stark_convert/image_checks.py`: the image-check memory preflight uses the loaded layout's resident bytes: 21.4 GB for the original and 23.0 GB for the 8-bit artifact (weights plus 1,152 MiB).
- `tools/converter/stark_convert/native_catalog.py`: the Inkling family message no longer says "original"; the catalog item keeps the identifier existing installations were matched by.
- `scripts/benchmark_standard.py`: an optional `[stark] default_artifact` profile field (the first artifact when absent), recorded in the run manifest's profile facts and shown by `profiles`.

Existing installations: a folder prepared from the original before this change is still recognized by `prepared_identity` with its original weight bytes and label, still loads (the worker selects the layout from the config binding), and still satisfies the app's "already downloaded" check for the catalog item. Nothing migrates it; the owner or user runs the upgrade explicitly:

```sh
uv run --project tools/converter --locked --python 3.13.13 python -m stark_convert.hybrid_prepare \
  inkling2-mini --upgrade-8bit --output <prepared-original-folder>
```

### Artifact identity and provenance

| Field | Value |
| --- | --- |
| Source checkpoint | `thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, Apache-2.0; target SHA-256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1` (19,526,019,168 bytes), draft head `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2` (635,801,416 bytes) |
| Pinned metadata | config `8216dd43…`, template `53a9035f…`, processor `d7350adc…`, tokenizer `7f332727…`, quantization config `7c656207…`, index `c1631065…` (full values in `inkling_prepare.py`) |
| Derived target | SHA-256 `c9e14848ad017467dc0deb91c6aa67022e181dbcdfe02a1b04024434bf9acddd`, 18,052,013,648 bytes; the draft head is copied byte for byte |
| Quantizer | `stark_convert.mlx_affine`, an independent transcription of MLX `affine_quantize` (ml-explore/mlx at `58d91ee81a86dd9710123ec42e7ff91f6d1d21b1`, MIT, Copyright (c) 2023 Apple Inc.); tensor selection from MLX-Serve `tests/convert_inkling2.py` at `2fe9be834a601c70ef1a597563026340f0e722fe` (Apache-2.0). All 178 quantized tensors are byte-identical to the author pack `ddalcu/Inkling-2-Mini-MLX-8bit` at `cb7fa052a9bbafbbeb91750edc67046e7bacd4f1` (Apache-2.0; [INKAFFINE8-02](2026-11-22-inkling-mini-affine8.md#20261122-inkaffine8-02--reproduce-the-published-pack-tensors-byte-for-byte)) |
| Receipts in the folder | `affine8-provenance.json` (source, algorithm, licences, quantized-name hash), `native-source.json` (layout `inkling2-mini-affine8-dense`) |
| Precision policy | Unchanged from INKAFFINE8-03: dense projections, embedding and head stored as unsigned 8-bit codes with BF16 scale and bias per row and group of 64; experts, router, gates, norms, convolutions, vision tower and draft head as published |

### End-to-end check (this attempt)

Run without network on 2026-11-23 on the M5 Max (macOS 26.7.2), from the uncommitted tree on top of `5a37dd37`. A staging folder was populated from the local pinned original with copy-on-write clones (`cp -c`) of the two weight files and the nine metadata files the preparation downloads, plus the staging marker a resumed download carries. Then `python -m stark_convert.hybrid_prepare inkling2-mini --output <dir> --offline --artifact-mode vision-language` ran the same `prepare` the catalog runs: verify, derive, verify, publish.

| Check | Result |
| --- | --- |
| Exit status, wall time | 0; 30.6 s for verification, derive and publication (no download) |
| Published target SHA-256 (computed independently with `shasum`) | `c9e14848ad017467dc0deb91c6aa67022e181dbcdfe02a1b04024434bf9acddd`, 18,052,013,648 bytes: equal to the pinned value |
| Draft head | `7c3b6513…` (unchanged) |
| `config.json` and `native-source.json` | Byte-identical to the artifact built for INKAFFINE8-03 (`a63529b3…`, `37e9c9c0…`) |
| Prepared identity | Equals the catalog source facts the desktop checks after an install |
| Leftovers | Download and derive folders removed; only the published folder remained |
| Worker smoke (`a8_worker_smoke.py`, behind the shared GPU lock; worker `stark-generate` SHA-256 `a7e1fee1c05f902b42f3a356d3beb8b423e162d9ec4002739dde1c719f0528df`, the commit-check build of `2a1d72a2`, quantized expert inputs) | Passed: started in 11.74 s with MTP supported (default 3 drafts); greedy 96-token output with MTP on equals MTP off (85 proposed, 53 accepted); a follow-up reused 2,048 tokens and equals cold; a synthetic image request completed with drafting off; the following text reused 0 tokens, equals cold and drafted (36 proposed, 13 accepted); no errors. The output hash equals the commit-check smoke on the separately built artifact. Result SHA-256 `7b530e0d9f839836c45d41104f87cc19783296c8a4b095c228363b43528e0e2d` (private, deleted with the folder). A smoke, not qualification |

The published folder was deleted afterwards.

### Qualification status per gate

Each item names its receipt or says it is not done. Evidence was gathered on the identical target bytes (pinned SHA-256 above) built by `inkling_affine8_artifact.build`; this attempt changed how that artifact is prepared and selected, not its bytes.

| Gate ([onboarding](../MODEL_ONBOARDING.md), [image parity plan](../IMAGE_DEFAULT_PARITY_PLAN.md)) | Status | Receipt |
| --- | --- | --- |
| Default route keeps images | Done for the converter: the default prepares a vision-language folder, recognized as the Inkling image family | This attempt's tests and end-to-end check; [INKAFFINE8-03 follow-up](2026-11-22-inkling-mini-affine8.md#follow-up-history) (client image check passes). Not exercised through the installed app |
| Natural-text quality against the original | Done, passes the benchmark bar | 2,580 rows, next-token accuracy 73.41 percent for both, 98.92 percent agreement where the reference is at least 50 percent confident; `runtime/inkling-engine-matrix/20261123-060520-both/stark-affine8/quality.json` SHA-256 `10c2293a8f56fc15e23299e01d47f47012db77c06e5b5165c3fb546f63924155`; [INKROUND-08](2026-11-23-inkling-mini-decode-round.md#20261123-inkround-08--natural-text-quality-probe-and-the-synthetic-capture-finding). Reference is the original's default prefill path, not exact prefill |
| Drafting on equals drafting off | Done at 73, 4,150, 8,800 and 16,000 tokens on the committed tree | `runtime/commit-check/` summary SHA-256 `8cb9cd12821508c4d284f0873d2ef85d08db89a62524db720f9487819eeab8cd` |
| MTP head retained and drafting with images loaded | Done as a smoke: head loaded, MTP on equals off, drafts proposed and accepted; image requests had drafting off | Commit-check smoke `smoke-affine8.result.json` SHA-256 `498b7ee56fca7defa66abdce23b264da64ae9e269282ea27536d1d9d8bc16920`, and this attempt's smoke |
| Prefix reuse exactness on the image-loaded worker | Met later: rule conformance 45 of 45 after `7e73e180` (follow-up history). At the time of this attempt: exactness shown; validator rule conformance not yet shown. `validate_generation_prefix_cache.py` in exactness mode, `--reuse boundary --vision`, context 16,384, first turn about 9,100 tokens, user-assistant session, 37 cases: every case successful, output with the prefix cache on identical to the same build with it off in 37 of 37, all 6 invalidation paths as documented (including a completed image request), follow-up turns reuse. The original artifact on the same worker gives the same 37 of 37. The validator's overall verdict is "not passed" on both, not for an output difference: the reuse count matches its documented rule in 15 of 37 cases. A control with the morning 2,048-row worker as its own candidate shows the identical 22 mismatching cases, so they predate the wide commands and the 8-bit artifact. Causes: the worker advertises 1,024-token reuse chunks while source prefill snapshots every 2,048 rows with a 4-token suffix, and the validator models a request with no drafting setting as not drafting while Inkling now drafts by default. Open item, not fixed: the worker's advertised chunk and the validator's expectations. "Identical to the baseline worker" is 0 of 37 in both candidate runs because the baseline was the morning build and default prefill arithmetic (half-precision attention) changed since; it is not evidence about the artifact (an earlier run before that change showed the wide-command worker identical to the 2,048-row worker in 37 of 37). Worker: candidate SHA-256 prefix `79653e4450cd`, 8,192-row prefill commands, quantized expert inputs. Since then the default changed to unquantized expert inputs (`8b550c9d`), verified by drafting on equals off at four lengths and the worker smoke on both artifacts, not by a fresh validator run | Receipts in `runtime/wide3/`: `prefix-wide-affine8.json` SHA-256 `c337b9d9aee6f753ee2c20a127d27451c4ab8a7127eb6f405258068188615779`, `prefix-wide-original.json` `0351853408d6afd662c29d75e9c63b3cfdcdebea160a59bd1efc4fbce5914761`, control `prefix-base-original.json` `cdc5796b6e6a2c7d7bbec01aaea2b25605125ebc5ede0eff0249d419210cfa8c` (coordinator's runs); commit-check and this attempt's smokes |
| Image request between text requests | Smoke only: image request with drafting off, following text reused 0 tokens, equals cold and drafted again. Failure and cancellation recovery not exercised on this artifact | Same |
| Matched text speed with images loaded against images not loaded | Met later (follow-up history). At the time of this attempt: not done. Engine-matrix run 4 timed the 8-bit arm with images loaded, one repeat, legacy layout; it is not a standard result and has no image-off arm | [Differences page scoreboard](inkling-mlx-serve-differences.md#latest-scoreboard-server-level); manifest SHA-256 `72a056bf5f4e91b6a512a2574cb129ce8457e53ca7a534c39e145ab692ca4804` |
| Standard benchmark result | Done later: [20261123-INKENGINE-06](../benchmarks/2026-11-23-inkling-mini-standard-benchmark.md). At the time of this attempt: no standard run on the committed tree | |
| Memory and load | Recorded in INKAFFINE8-03; admission and image preflight now charge the 8-bit layout | See below |
| Tool calling | Met later: all four checks pass (follow-up history). At the time of this attempt: not run on this artifact (`validate_tool_calling.py`); the template is the pinned original's | |
| Installed app end to end (catalog install, upgrade from an existing install, image check, chat) | Not done | |

### Memory, disk and load

| Figure | Original | 8-bit default | Source |
| --- | --- | --- | --- |
| Installed weight bytes | 20,161,820,584 | 18,687,815,064 | Pinned identities |
| Free disk required at preparation start | download + 4 GiB | download + derived + 4 GiB = 43,144,602,944 bytes; the download and the derived folder coexist until publication | `hybrid_prepare.prepare` |
| Worker load admission | target + draft + 1 GiB | the same plus 1,670,000,000 | `crates/stark-generate/src/inkling.rs` |
| Image-check memory preflight | 21.4 GB | 23.0 GB | `image_checks.preflight_family_memory` |
| Metal allocation after load | 21.09 GB | 22.76 GB (affine tensors add 1,668,243,456 bytes beside the BF16 copies) | INKAFFINE8-03, single load each |
| Worker start | 10.8 s | 8.6 s; 8.7 s in the commit-check smoke; 11.74 s in this attempt's smoke on a busy host | Single samples |

The 8-bit artifact is smaller on disk and larger in memory, because the BF16 copies stay resident.

### Tests

- `tools/converter/tests/test_inkling_default_prepare.py` (new, 14 tests, model-free, real tensor set): default preparation publishes only the verified derived folder and removes the download; the original choice is unchanged; a pre-change download resumes into the default; a derived hash or size mismatch publishes nothing, keeps the download and is retried cleanly; an interrupted derive leaves no published folder; a derive folder not owned by the operation is refused; disk space covers both folders up front and the derive refuses with `ENOSPC`; the upgrade replaces a prepared original, is idempotent, and leaves the original untouched on a hash mismatch or full disk; the upgrade refuses other folders and choices; catalog facts; image preflight memory per layout.
- `tools/converter/tests/test_guided_hybrid_models.py`: the original choice is listed as unprofiled.
- `scripts/test_benchmark_standard.py`: the Inkling default artifact is `affine8` and its default arm is `stark-affine8`; matched pairs unchanged; the field falls back to the first artifact; an unknown default is refused.
- Results: converter suite 8,024 passed, 96 skipped (8,010 before); benchmark tests 127 passed (124 before); `ruff check` and `ruff format --check` clean on the touched converter files.

### Not done and what remains before an installed build

- Prefix-reuse validator rule conformance: the advertised reuse chunk and the validator's drafting expectation must be fixed and the validator rerun on the current default (unquantized expert inputs, `8b550c9d`). Exactness is shown on the earlier worker.
- A matched image-off against image-on text speed comparison and a standard benchmark run on the committed tree.
- Tool-calling validation on this artifact.
- The app end to end: catalog install of the default, the upgrade of an existing original install, the local image check and chat. The app offers no upgrade button; an existing original install stays on the original until the upgrade command is run.
- Failure and cancellation recovery around image requests on this artifact.
- The worker's comment on `AFFINE8_LAYOUT` in `crates/stark-generate/src/inkling_artifact.rs` still says the artifact is "never the default"; it was not edited here.
- Release and installation.

Keep prompts, completions, token vectors and probabilities out of this record; it contains aggregates and hashes only.

## Follow-up history

- 2026-11-23, commit: this change is commit `222a9436`, which also removed the worker comment on `AFFINE8_LAYOUT` in `crates/stark-generate/src/inkling_artifact.rs` that said the artifact is "never the default"; the item about that comment under Not done is closed. The disposition line above ("uncommitted at registration") describes the state when the record was written. Not released or installed.
- 2026-11-23, wording: the prefix-reuse paragraph attributes the 0-of-37 baseline difference to "half-precision attention"; the default prefill change in question is BF16 attention with F32 scores and accumulation ([INKFASTPRE-05](2026-11-23-inkling-mini-prefill-round4.md#20261122-inkfastpre-05--bf16-prefill-attention-and-a-1024-row-global-query-tile)).
- 2026-11-23, later default changes on the same artifact bytes: unquantized NVFP4 expert inputs (`8b550c9d`, [INKROUND-09](2026-11-23-inkling-mini-expert-inputs-default.md)), on which the 8-bit artifact scored 73.99 percent next-token accuracy against the exact original reference's 73.64 with 99.43 percent confident agreement on 2,580 natural rows; planner tuning (`1206760e`) and the planner hold rule (`4314ac51`, [planner record](2026-11-23-inkling-mini-planner-tuning.md)); the gathered NVFP4 prefill kernel for commands of 768 rows or more (`4314ac51`, [INKFASTPRE-07](2026-11-23-inkling-mini-gathered-nvfp4.md)), hash-identical on this artifact. The tree at `4314ac51` passed drafting on equals off at four lengths and the worker smoke on both artifacts (`runtime/commit-check-3/`). None of these was covered by a fresh prefix-reuse validator run; the open items above are unchanged.
- 2026-11-23, matched image-on against image-off text speed, interim (coordinator; private outputs in `runtime/image-parity/`): one set of four worker loads in the order off, on, on, off, six inputs with two requests each, default drafting, worker `stark-generate` beginning `a966b11554ae` (the `4314ac51` build). Outputs were identical with images on and off on every input and run. The first pair is discarded: the teardown of a killed 23 GB engine process overlapped it and held worker decode at 14 to 17 tok/s for three requests ([wind-down](2026-11-23-inkling-mini-wind-down.md)). In the second pair image-on over image-off read 0.93 to 0.97 on decode and 0.92 to 0.98 on prefill, one pair with about 4 percent run-to-run spread, so the gate is not settled. A clean re-run is owed.
- 2026-11-23, final gates (coordinator, on the final build `stark-generate` `f86d94b877b3…`, whose gate hashes equal the `4314ac51` build's):
  - **Matched image-on against image-off text speed: met.** Two off, on, on, off sets inside the same lock hold as the standard run, three inputs (about 50, 39 and 4,000 tokens), two requests each, default drafting. Outputs identical across every load. Image-on over image-off: decode median 1.002 (0.968 to 1.019) and 0.995 (0.967 to 1.010); prefill 1.018 (0.997 to 1.049) and 1.000 (0.992 to 1.036). Receipts `runtime/final-timed/image-b-pairs.json` and `image-c-pairs.json`, SHA-256 prefixes `67fcf08aa14b6260` and `c743ce565142daca`. The interim 0.93 to 0.97 reading above came from sets contaminated by a process teardown and is superseded.
  - **Prefix-reuse rule conformance: met** after `7e73e180`, which makes the worker advertise reuse chunks of 2,048 tokens with a 4-token tail (`tail_tokens` 1 for every other backend) and makes the validator read `tail_tokens` and `mtp.default_drafts`, with added tail-edge and explicit `mtp_drafts` 0 cases. The validator passes on both artifacts, 45 of 45 cases each: reuse matches the advertised rule, output with reuse equals fresh state and equals the baseline worker built from `670818b5` (SHA-256 prefix `33ed4dc91d2e`), all six invalidation paths as documented; exactness mode, boundary reuse, vision loaded, context 16,384, user-assistant session with turns of 9,096 to 11,367 tokens; 439 and 448 seconds. Receipts `runtime/final-checks/prefix-affine8.json` and `prefix-original.json`, prefixes `77046502f9c62241` and `950890c6d53aee4e`. The 22 mismatches of the earlier runs are explained by the two modelling gaps that commit closes.
  - **Tool calling: met.** `scripts/validate_tool_calling.py` passed all four checks (single call and arguments, tool-result use, autonomous tool-result use, ordinary chat), context 8,192, greedy, the validator's own drafting setting of 0, parser `inkling2-json-v1`, worker `f86d94b877b3…`; receipt `runtime/final-checks/tools-affine8.json`, prefix `59678a3afee6db47`.
  - **Standard benchmark: done.** [20261123-INKENGINE-06](../benchmarks/2026-11-23-inkling-mini-standard-benchmark.md) times this artifact as the default arm with its quality result attached (pass).
  - **Still open:** the app end to end (catalog install, upgrade of an existing original install, image check, chat); failure and cancellation recovery around image requests on this artifact; release and installation. Nothing is installed.
