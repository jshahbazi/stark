# Stark standard benchmark

One protocol and one tool that every model Stark supports is measured with, so that a result for one model, build or artifact can be read next to any other. The protocol is this page. The tool is [`scripts/benchmark_standard.py`](../scripts/benchmark_standard.py), and everything model-specific lives in a profile under [`scripts/benchmark_profiles/`](../scripts/benchmark_profiles/).

This page defines the benchmark; it holds no results and qualifies nothing. A dated record of a run goes under [`docs/benchmarks/`](benchmarks/README.md) and into the [experiment ledger](model-experiments/README.md), bound to the builds and artifacts named inside it. The rules for optimization work (same-engine A/B gates, correctness gates) stay in the [performance playbook](PERFORMANCE_PLAYBOOK.md).

## Purpose and scope

The standard answers one question: how fast is this model, as Stark ships it, against the engine people would otherwise run it on, on matched work. It measures text generation at the server level, through the real `stark-generate` worker, with the same client that the reference engine publishes its numbers with.

In scope: single-stream decode and prefill speed, speed across context lengths, a concurrent burst, and a quality result, measured on natural text, for the artifact and execution paths that were timed. Every speed result is paired with one.

Every new family, checkpoint, quantization or format adds or extends a profile and gets its own standard run and quality result. A figure taken any other way states its level and is not a standard result.

Out of scope: scoring and decision workers (`stark-score`, `stark-decide`), image requests, conversion time, memory, startup and idle wake. Those have their own records. Kernel-level and stage-level profiling is diagnostic evidence and is covered by the playbook.

The protocol first ran on Inkling 2 Mini on 2026-11-22 ([engine matrix](benchmarks/2026-11-22-inkling-mini-engine-matrix.md), [prefill attribution](benchmarks/2026-11-22-inkling-mini-prefill-attribution.md)). Each rule under [Validity rules](#validity-rules) is there because a comparison went wrong without it.

## Levels

Every figure states the level it was measured at. Figures from different levels differ by tens of percent on the same build and are never placed in one row.

| Level | What is timed | Where it appears |
| --- | --- | --- |
| Server, client-timed | A request over HTTP through the real worker or server, on the client's clock | Standard results |
| Server, engine phase time | The engine's own prefill or decode time for a server request, read through the server | Standard results, marked in the Level column |
| Worker | The raw worker protocol on standard input and output, without the gateway or the client | Optimization harnesses; diagnostic |
| Engine diagnostic | One process, no server: a profile, a stage table, a prefill replay, the quality probe | Diagnostic; the quality slot |

Only the two server levels are standard results. On 2026-11-22 the Inkling worker's 8,800-token prefill was about 19 percent slower than the same tree's engine diagnostic, about half from draft work inside the worker's prefill and about half from GPU clock state ([prefill attribution](benchmarks/2026-11-22-inkling-mini-prefill-attribution.md)). Neither number is wrong; they are different measurements.

## Scenarios

The client is llmprobe 0.6.18, pinned by file hash, with a patch that emits aggregate receipts. Every request is greedy. Decode scenarios cap output at 192 tokens.

| Scenario (key) | Request | Why it is measured |
| --- | --- | --- |
| Initial coding (`decode`) | One short user turn asking for a TypeScript function, cold prefix; 1 warmup and 3 measured per session | The everyday short request on a cold prefix |
| Sustained coding (`sustained_decode`) | The initial-coding request again, 3 measured, after every other client scenario and a 5 s idle | The same work on a warm engine and a loaded machine. MLX-Serve's best published Inkling 2 Mini rate came from this cell |
| Novel prose (`spec_novel`) | One short turn asking for original prose; 1 warmup and 3 measured | Low draft acceptance by design, so it shows the rate when drafting helps least |
| Context ladder (`context_512`, `context_4k`, `context_8k`, `context_16k`) | A synthetic code archive fitted to about 512, 4K, 8K and 16K input tokens, then a coding instruction; 1 cold run per rung per session | Decode rate as the prompt grows: attention cost, long-position paths, context planning |
| Long-prompt prefill through the client (`prefill_repeated`) | About 40 KB of one repeated sentence and a one-word reply instruction, 8-token cap; 1 warmup and 3 measured | Prefill as the client sees it, on the client's own long prompt |
| Exact prefill control (`prefill_8800_client`, `prefill_8800_native`) | Seeded unseen text cut to exactly 8,800 token IDs, sent as raw IDs, 8-token cap; 1 warmup and 1 measured per session | The matched prefill cell: every arm receives identical token IDs, checked by hash |
| Burst (`burst_16x300`) | 16 distinct prompts released together, 300 tokens each, drafting off; 1 warmup and 1 measured per session | Throughput under concurrent users. Reported only: see below |

Every row above except the burst is in the standard set: the cells a standing is read from.

The 16-user burst is run and reported in every standard run, and it does not count toward a standing. Stark serializes one worker by design, so a batching reference engine is expected to lead this row; the row shows the cost of that design choice and decides nothing about engine speed.

The tool also records supplementary cells, which like the burst are reported and never used for a standing: unseen-text prefill through the client (its token count depends on the tokenizer), the echo ceiling (`spec_predictable`), the counting ceiling of each ladder rung (it shares the rung's prefix on purpose, so it is not a cold number) and the burst's wall time. The summary marks every row as `standard` or `supplementary`, and a supplementary row's paired ratio is labelled reported only.

"Sustained" means the initial-coding request repeated at the end of the client's suite. It is not multi-turn and not long-context, and its value depends on everything that ran before it.

A profile whose context limit cannot hold a scenario does not run it, and the plan says so. The record lists it as not applicable with the reason.

## Metrics

A token is a token the engine reports in its usage counts: input tokens after chat rendering, and every generated token, reasoning included. The tool never counts tokens itself.

| Metric | Definition |
| --- | --- |
| Decode tok/s (client scenarios) | Output tokens delivered after the first text frame, divided by the time from the first text frame to the last. The first frame's share of the output (by characters) is left out of the numerator, so a server that sends a large first frame is not credited for it. On a one-token-per-frame stream this is (N - 1) / span. Reasoning frames are text frames |
| Prefill tok/s (client scenarios, and `prefill_8800_client`) | Input tokens divided by the time from sending the request to the first text frame. It includes rendering, tokenizing, HTTP and the first decoded token |
| Prefill tok/s, engine phase (`prefill_8800_native`) | Input tokens divided by the engine's own prefill time for that request: the increase of the server's prefill phase counter across the request. For Stark the gateway sums the worker's reported prefill seconds |
| Burst tok/s | All generated tokens of the 16 streams divided by the time from releasing the streams to the last one finishing |
| Tokens per step | The client's estimate of tokens emitted per model step. It describes drafting; it is not a rate, and two engines may not count steps the same way |

Prefill and decode are separated by the first text frame on the client's clock, and by the engine's own phase boundary on the engine's clock. A sample is left out of a cell and counted when the request failed, when it has no rate, or, in a decode scenario, when it produced fewer than 192 tokens: a reference engine that honours the client's forced length runs to 192, while Stark follows the model's own stop, and only equal-length samples did equal work. A prefill or burst trial is invalid when its prompt token count is wrong, when any input was served from cache, or, in the burst, when any stream stopped short of 300 tokens.

## Arms and reference engines

An arm is an engine build serving one model artifact. Arms are never configuration flags.

- A Stark arm is one `stark-generate` binary and one artifact the profile declares. `--stark-binary` is the build under test, `--stark-build NAME=PATH` adds another build, and every build runs every selected artifact.
- A reference arm is one pack served by a reference engine.

**Arm names.** Stark arms are named for the artifact, `stark-<artifact>`: `stark-original`, `stark-affine8`. When one run compares builds, the added build's arms carry its name as a suffix: `stark-original-baseline`. Reference arms are `<prefix>-<artifact>` with the prefix the profile gives the engine: `mlx-original`, `mlx-author-8bit`.

**Reference engines.** Each profile declares its reference engines and their packs. The policy for choosing them:

- The reference engine is the fastest engine that publishes numbers for that model on Apple silicon, run at its shipping defaults.
- The reference arm, which every speed ratio is taken against, is that engine serving the original checkpoint. A profile cannot name one of Stark's own arms as the reference arm.
- For each further Stark artifact there is one matched reference arm, where the reference engine has an artifact of the same quantization. Where it has none, that Stark artifact is compared with the reference arm only, and the summary says the ratio compares artifacts as well as engines.

For Inkling 2 Mini that is MLX-Serve: the original checkpoint as the reference arm, and the author's 8-bit pack matched to Stark's `affine8` artifact. A reference engine is an adapter in the tool (see [Adding a reference engine](#adding-a-reference-engine)).

**Matched artifacts.** Two arms are a matched pair when their artifacts come from the same source checkpoint revision and have the same quantization: the same tensors at the same bit width under the same scheme. The profile states this with a `match` class per artifact, and cites the record that justifies it. Equal bit width alone is not a match. Only a matched pair compares engines. A ratio between arms of different classes compares artifacts as well as engines, and the summary says so. For Inkling 2 Mini the original checkpoint pairs with MLX-Serve's load of the original, and `affine8` pairs with the author's 8-bit pack because it holds the same quantized tensors: all 178 are byte-identical ([20261122-INKAFFINE8-02](model-experiments/2026-11-22-inkling-mini-affine8.md#20261122-inkaffine8-02--reproduce-the-published-pack-tensors-byte-for-byte)).

An artifact the profile does not declare can be run by giving its path. Its arm is diagnostic.

## Validity rules

A comparison is valid only when prompt rendering, reasoning mode, sampling, output length and drafting policy are matched, or the difference is the stated subject of the comparison. The tool enforces what it can and records the rest.

1. **Matched rendering.** Both engines must render the same prompt. The profile fixes the client's reasoning mode; for Inkling 2 Mini it is `off`, which renders effort level 0 on both engines while the model may still think. The check is that every arm reports the same input token counts for the same request; the summary compares them per scenario and marks a mismatch as not comparable. The exact prefill control compares input-ID hashes. An earlier comparison was invalid because the client's default effort was rejected by one engine and accepted by the other.
2. **Engine defaults, not tuned arms.** Each engine runs what it ships. The gateway forwards what the client asked for and nothing else, so the worker's drafting default applies; a reference engine starts without a draft flag. `STARK_*` and `UZU_*` variables inherited from the shell are removed before a worker starts. A diagnostic environment variable, a forced drafting setting, an effort alias, a changed context minimum, a changed client run count or reasoning mode, a busy host that was allowed, or an unpinned client build is recorded and takes the arm or the run off the scoreboard.
3. **State the level.** See [Levels](#levels). The summary's Level column carries it per row.
4. **Read what the other engine loads.** A reference engine may skip a draft head, pick a different kernel path or serve a different artifact than its name suggests. The profile can keep its console lines as fixed events, and a record says what the engine reported.
5. **Equal work.** Samples of unequal output length are excluded and counted. Cached input in a cell defined as cold is reported.
6. **Robust arms.** Before anything starts, the tool checks that binaries exist, that a Stark artifact is the profile's model with the profile's chat template, and that every weight shard an index names is present. A failed arm is recorded with the server's own startup lines and does not stop the others; `--arms` fills it in later.
7. **Privacy.** Receipts hold counts, timings, hashes and fixed codes. Prompts, responses, token vectors, probabilities and credentials are never written. Run directories go under the ignored `runtime/` directory, and the tool refuses an output directory inside the repository that Git does not ignore. A published record carries aggregates and hashes.
8. **Speed without quality is not a result.** See [Quality](#quality).
9. **One GPU job at a time.** A run holds an exclusive lock file for its whole duration. When several agents or people share the machine, every build and GPU job goes behind one shared lock.
10. **Image-capable models are timed with images loaded.** Text speed is measured as the model ships. When the profile says the model is image-capable, a worker that reports images not loaded fails its arm (`stark_images_not_loaded`) instead of being timed without them. This applies per artifact: a new quantization whose image path the worker does not load cannot be timed until it does. There is no exception and no override, diagnostic or otherwise.

## Host requirements

A timing run refuses to start unless the machine is on AC power, Low Power Mode is off and no competing model server or compiler is running. `pmset` reports `powermode 2` for High Power; only `powermode 1` or `lowpowermode 1` is Low Power Mode. The run holds `caffeinate`, checks the power source before every session and stops if it changes.

The manifest records hardware model, chip, memory, macOS version, power source, the adapter's negotiated and rated wattage, battery state, thermal state before every session and sleep events. Three conditions are recorded as notes and do not stop a run: a charging battery, an adapter negotiating less than its rating, and a thermal state above nominal. On 2026-11-22 a 140 W adapter negotiated 94 W, the GPU moved between performance states, identical requests varied by 10 to 15 percent, and the first request after an idle period was consistently faster. That is why results are read from paired blocks and not from single absolute numbers.

## Repeats, layout and comparison

A standard run has at least three repeats and uses the block layout. Each repeat is one block: a fresh server session per arm for the client suite, then one per arm for the exact prefill control, then one per arm for the burst. A prefill or burst session holds one warmup and one measured trial, so every measured trial starts from a fresh server. Arm order is reversed on odd repeats. Stark and reference arms alternate, with matched arms next to each other. So within a block every arm's figure for a scenario is adjacent in time to the others'.

Each cell reports the median, the range and the count of its samples, and how many were excluded. Two comparisons are reported:

- The ratio of medians against the reference arm, for continuity with earlier records.
- The paired ratio, against the reference arm and against the matched arm where there is one: within each block, the arm's median over the other arm's median; then the median, range and count of those per-block ratios. This is the comparison to use.

**The standing rule.** The paired ratio and its range are always reported. A standing is read from them with no numeric threshold on top: an arm is **ahead** only when every block's ratio is above 1 and there are at least three blocks, **behind** only when every block's is below 1 and there are at least three, and **unresolved** when the blocks disagree. Fewer than three blocks resolve nothing. Three further cases hold no standing, and the summary names each in place of one:

- a supplementary row, the burst included, is reported only;
- trials that were not adjacent in time (the legacy layout's prefill and burst sessions) are not time-paired;
- a Stark arm without a passing quality result is speed only (see [Quality](#quality)).

An arm's figures count as a standard result when the run followed the protocol without overrides, the arm has no disqualification, its inputs matched the other arms', and, for a Stark arm, its quality result passes. The summary states this per arm with the reasons.

`--layout legacy` exists only so a new run can be compared with the runs recorded on 2026-11-22, which predate the block layout: all client repeats first, then one prefill session and one burst session per arm with every trial inside, in fixed arm order. It is a diagnostic override and is not for new results. `summarize` reads those run directories and keeps their arms separate from block-layout arms of the same build.

## Quality

Stark ships execution paths that are not bit-exact with their reference, and quantized artifacts. A speed figure for such an arm means nothing until the output is shown to be close enough. Every Stark arm therefore has a quality slot, holding one quality result for the artifact that was timed, measured on natural text. A Stark arm whose result fails, is not assessed or is missing is listed as speed only and cannot hold a standing; its ratios and ranges are still reported.

**The probe.** The probe is pluggable and separate from the tool. It is an engine diagnostic that compares the arm's artifact on its default paths with a reference artifact or path, teacher-forced on natural text, and it runs under the GPU lock like any other job. `quality attach` validates its reports, reduces them to numbers and fixed codes, pools them, judges the result and stores it beside the arm's speed receipts. The result is bound to the artifact that was timed: the artifact directory given at attach time must have the fingerprint the run recorded for the arm, so a result cannot be carried to another artifact, and a stored result bound to a different artifact is not assessed.

**The bar.** One bar for every model, measured against the reference artifact or path:

| Verdict | When |
| --- | --- |
| pass | The arm's next-token accuracy is not more than 2.0 percentage points below the reference's, and the two choose the same top token in at least 98 percent of the rows where the reference's top-1 probability is at least 0.5 |
| fail | Either criterion is not met. The reason names which |
| not assessed | The corpus is not natural text; or the reference predicts fewer than 10 percent of its next tokens; or no row has a reference top-1 probability of at least 0.5; or the result is bound to another artifact. Not assessed is never a pass |

The accuracy criterion is one-sided: an arm that scores higher than the reference does not fail on it.

Both criteria are evaluated on the pooled rows of the whole corpus, not per input. A probe writes one report per input; give `quality attach` every report (repeat `--report`) and it pools them over their rows, so every rate is the rate over all rows of all inputs, and judges that. The bar was set on 2026-11-23 from natural-text measurements in which single inputs of 258 rows differed from the reference by up to 2.3 points, more than the limit, while agreement on confident rows stayed between 98 and 100 percent. A per-input limit would fail on the noise of a few hundred rows.

The 10 percent floor is there because synthetic or padded test prompts can be nearly unpredictable for the model: on one such input the reference's top token matched the actual next token 0 times in 34 ([prefill round 3](model-experiments/2026-11-23-inkling-mini-prefill-round3.md#quality-limit-synthetic-captures)). Agreement on such text says little, so the result is not assessed however good its figures are.

KL divergence and the log-probability change are reported with every result and have no bar.

A report in the slot schema `stark-benchmark-quality-v1` (print it with `quality schema`):

| Field | Meaning |
| --- | --- |
| `corpus.rows`, `corpus.inputs` | How many next-token positions were compared, over how many inputs. Pooling adds them up |
| `corpus.prompt_tokens`, `corpus.context`, `corpus.window` | The geometry, kept when every pooled input has the same |
| `corpus.kind`, `corpus.sha256` | `natural-text` or `synthetic`, and the hash of the private token file. Given at attach time |
| `reference.path`, `candidate.path` | `exact` or `default`: which execution path each side used |
| `metrics.next_token_accuracy.reference`, `.candidate` | Share of positions where each side's top token is the text's actual next token |
| `metrics.top1_agreement.all` | Share of positions where both sides choose the same top token |
| `metrics.top1_agreement.strata[]` | The same, restricted to positions where the reference's top probability is at least a threshold (`reference_top1_probability_min`, `rows`, `agreement`) |
| `metrics.kl_reference_to_candidate.mean`, `.max` | KL divergence from the reference distribution to the candidate's, in nats |
| `metrics.next_token_logprob_abs_change_mean` | Mean absolute change in the log-probability of the actual next token |

The tool adds the verdict, its reason and the figures it was judged on (`verdict`, `verdict_reason`, `criteria`), the probe's name and binary hash, both artifact fingerprints, the hash of every source report, the arm it is bound to and the worker binary hash of that arm. `summarize` judges the stored figures again, so every result in one table is held to the same bar. When an arm has results in several runs, one failing result fails the arm. The probe should be built from the same source as the worker that was timed, and a record states both hashes.

`--format inkling-slice-quality-compare` accepts the printed report of `inkling_slice --quality-compare` as it stands today, one report per input. That probe is still being developed, and the converter will need to follow it.

## Running it

From the repository root. `BENCH` stands for `uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_standard.py`.

| Command | What it does |
| --- | --- |
| `BENCH profiles` | Lists the profiles, their artifacts and the arms each can form |
| `BENCH plan --profile KEY [options]` | Validates the profile and every path, then prints everything a run would start, in order. Starts nothing; needs no GPU |
| `BENCH run --profile KEY [options]` | Runs it. `--dry-run` is the same as `plan` |
| `BENCH summarize RUN [RUN ...]` | Joins run directories into `summary.md` and `summary.json` |
| `BENCH quality attach --run RUN --arm ARM --report FILE [--report FILE ...] --candidate-artifact DIR --corpus-kind natural-text` | Pools and judges the probe's reports and fills the arm's quality slot |

Prerequisites: `stark-generate` and `stark-template` built with Rust 1.98.1 (`cargo +1.98.1 build --release --locked -p stark-generate`, and `-p stark-template`), the artifacts, the reference engine binary and packs, an llmprobe 0.6.18 checkout with `node_modules` and the receipt patch bundled, and Node.js 22.19 or newer.

The standard run for Inkling 2 Mini, with the original checkpoint and the 8-bit artifact on Stark and on MLX-Serve:

```sh
uv run --project tools/converter --locked --python 3.13.13 \
  python scripts/benchmark_standard.py run --profile inkling-2-mini \
  --stark-binary target/release/stark-generate \
  --stark-artifact original=<prepared-original-dir> \
  --stark-artifact affine8=runtime/inkling-affine8/artifact \
  --reference-binary <mlx-serve> \
  --reference-artifact original=<original-checkpoint-dir> \
  --reference-artifact author-8bit=<author-8bit-pack-dir> \
  --llmprobe <llmprobe-0.6.18-checkout> \
  --output runtime/benchmark/inkling-2-mini
```

The arms are `stark-original`, `mlx-original`, `stark-affine8` and `mlx-author-8bit`, matched in pairs, and `mlx-original` is the reference arm. Replace `run` with `plan` to check it first. An artifact named without a path uses the profile's default path. Useful options: `--repeats N`, `--sessions probe,prefill,burst`, `--engines stark`, `--arms NAME` to run one arm into a new run directory, `--stark-build baseline=PATH` to add an earlier build. The options under "diagnostic overrides" in `--help` each take a run off the scoreboard.

Exit status: 0 complete, 2 configuration problems, 3 a failed arm, 1 a refusal or error.

The earlier command line, [`scripts/benchmark_inkling_engine_matrix.py`](../scripts/benchmark_inkling_engine_matrix.py), still works. It is an entry point over the same code with the Inkling 2 Mini profile, the legacy layout and its original arm names (`stark-current`), kept for comparing with the runs recorded on 2026-11-22. New results use the standard tool.

## What a published record contains

A record under `docs/benchmarks/` for a standard run states:

- attempt ID, date, and what the run was for;
- profile key and file hash, the harness file hashes from the manifest, the repository commit and its count of modified paths;
- for each Stark arm: the worker binary SHA-256, the artifact (source repository and revision, quantization, match class) and what the build contains relative to the last record;
- for each reference arm: engine version, binary SHA-256 and the pack's identity;
- client version and whether the build is the pinned one;
- the command, layout, repeats and sessions, and every deviation the manifest lists;
- reasoning mode, the evidence that renders matched (input token counts per scenario), and drafting as requested and as each engine reported it;
- host: hardware, macOS, power source, adapter wattage, thermal states, sleep events, and the host notes;
- the result table: per scenario and arm the level, median, range, count and exclusions, the paired ratio with its range and count against the reference arm and the matched arm, and the standing;
- the burst as its own reported row, not as part of any standing;
- the quality result of every Stark arm: the verdict, the pooled rows and inputs, the corpus hash, next-token accuracy of both sides, top-1 agreement overall and on confident rows, KL divergence, and the probe and worker binary hashes; or that the arm is speed only and why;
- the validity findings, what was not run and why, and failed arms;
- where the private receipts are kept and the SHA-256 of each run manifest;
- links to the experiment ledger entries, and which earlier rows the record supersedes.

A record never contains prompts, responses, token vectors or probabilities.

## Model profiles

A profile is one TOML file, `scripts/benchmark_profiles/<key>.toml`, read with the standard library. It holds facts about one model, each with a comment naming where the fact was verified. Protocol constants are not in profiles. A key the format does not define is an error.

| Field | Meaning |
| --- | --- |
| `schema`, `key`, `title`, `status` | `stark-benchmark-profile-v1`; the key equals the file name; `status` is `complete` or `draft`, and only a complete profile runs |
| `reference_arm` | The arm that speed ratios are taken against: a reference engine's arm, by policy its load of the original checkpoint. One of Stark's own arms is refused |
| `[source]` `repo`, `revision` | The source checkpoint |
| `[client]` `reasoning`, `cache_tag_base` | The client reasoning mode that renders identically on every engine, and the tag in each cache-busted prompt |
| `[stark]` `served_model_id` | Model id the gateway serves |
| `[stark]` `model_type`, `template_sha256` | What a Stark artifact's `config.json` must report; an artifact that differs is refused |
| `[stark]` `vocab_size` | Used only when an artifact's `config.json` does not carry it |
| `[stark]` `chat_adapter` | How replies are split into content and reasoning: `inkling2-json-v1` or `think-tags-v1`. It is checked against the artifact's template |
| `[stark]` `image_capable` | When true, the worker must report images loaded in every session, so text speed is measured with images resident |
| `[stark]` `drafting` | `engine-default` |
| `[stark.context]` `minimum`, `maximum`, `burst` | Worker context planning: the smallest planned context, the limit, and the per-stream context of the burst |
| `[stark.artifacts.NAME]` | `description`, `quantization`, `match`, and optionally `path`, `repo`, `revision`, `record` |
| `[[reference]]` | `engine`, `label`, `arm_prefix`, `binary`, `version`, launch `flags`, `draft_off_flags`, `draft_force_flags`, optional `[[reference.log_event]]` console lines to keep as fixed events, and `[reference.artifacts.NAME]` |
| `[quality]` | `reference_artifact`, `reference_path`, and the probe's name, report format and command template (printed by `plan`, not executed) |
| `[notes]` | Model-specific readings of standard scenarios, by scenario key |

[`inkling-2-mini.toml`](../scripts/benchmark_profiles/inkling-2-mini.toml) is the first complete profile and the example to copy.

## Adding a model

1. Finish the family's onboarding first ([model onboarding](MODEL_ONBOARDING.md)): the standard measures a working model.
2. Copy the example profile to `scripts/benchmark_profiles/<key>.toml` with `status = "draft"`. Fill each field from a source you can point at, and say where in a comment. Leave out what you cannot verify.
3. Declare every artifact that ships, each with its quantization and a `match` class. Give two artifacts the same class only when a record shows they hold the same tensors at the same widths.
4. Choose the reference engine and its packs by the policy under [Arms and reference engines](#arms-and-reference-engines): the fastest engine that publishes numbers for the model on Apple silicon, the original checkpoint as the reference arm, and one matched pack per further Stark artifact where one exists. Read what the engine actually loads for this model (draft head, kernel path, quantization) and declare the launch flags that make it serve the protocol's endpoints.
5. Run `plan`. Fix every configuration problem. The plan shows the worker and server command lines, the context limits, the session order and which scenarios the context limit cannot hold.
6. Under the GPU lock, run one arm, one repeat, the client session only, on each engine. Compare the input token counts per scenario between the two engines. If they differ, the reasoning mode or template does not render the same prompt; fix the profile before timing anything. Read the client's reasoning note for both engines in each `bench.json`.
7. Set `status = "complete"`, add a test that the profile loads if it needs one beyond the catch-all, and run `scripts/test_benchmark_standard.py`.
8. Run the standard: at least three repeats, all arms, block layout. Run the quality probe on natural text for every Stark arm and attach every per-input report in one `quality attach`. An arm whose result does not pass is speed only.
9. Summarize, write the record, index it in [benchmarks](benchmarks/README.md) and [model research](MODEL_RESEARCH.md), and add the family's entry to its speed history.

What a profile for the other native families still needs, because it could not be verified from the repository alone:

| Family | Known from the repository | Still needed |
| --- | --- | --- |
| Qwen3.8-Flash-Next | Artifacts and revisions (Lily Q4 XL, the MLX-Serve mixed 4/8-bit pack), a 32,768-token context policy, matched-checkpoint MLX-Serve comparisons with the earlier harnesses | MLX-Serve launch flags that serve this protocol; the reasoning mode that renders identically on both engines; a model-bound run of the `think-tags-v1` adapter; the worker's planned context minimum; a natural-text quality probe |
| GLM-5.3-Flash | A greedy MLX-Serve baseline harness and its launch flags for a 512-token context | Which pack is a matched artifact for the native Q2 source; whether the ladder and the 8,800-token control fit the supported context; everything listed for Qwen3.8 |
| Dense Qwen3.5 and other converted families | Stock-engine comparisons against MLX-LM, llama.cpp, vLLM and TensorFold at the worker and server level | A reference-engine adapter for the chosen server; its endpoints for token-ID prompts and phase times; the profile facts above |

### Adding a reference engine

A reference engine is a subclass of `ServerEngine` in the tool with one required method, `command`, which returns the server's full argument list for a session, and an optional `console`, which keeps numeric lines from its console output. Startup polling, the private home directory, the startup tail on failure and shutdown are shared. Register it in `ENGINES` and declare a `[[reference]]` block for it in a profile. The server must provide `/v1/models`, `/tokenize`, the chat completions route the client uses, streamed `/v1/completions` that accepts token-ID prompts and reports usage, and `/metrics` with prefill and decode phase-time counters. The test `test_a_new_reference_engine_is_an_adapter_and_a_profile_block` does exactly this with a fixture server.

## How the older benchmark scripts relate

The older scripts stay where they are. Most are optimization harnesses at the worker level, which the standard does not replace; a few are engine comparisons that the standard supersedes for new work.

| Script | Level and subject | Relation to the standard |
| --- | --- | --- |
| [`benchmark_inkling_engine_matrix.py`](../scripts/benchmark_inkling_engine_matrix.py) | Server; Inkling 2 Mini against MLX-Serve | The origin of the standard. Now an entry point over the same code with the legacy layout, kept for comparing with the runs recorded on 2026-11-22 |
| `benchmark_stock_engines.py`, `benchmark_clean_qwen.py` | Worker pipes and HTTP servers; one 4-bit Qwen3.5 pack on five engines; end-to-end output rate | Historical engine comparison. Its rotating blocks, host guard and sleep check are in the standard. Its continuous CPU, GPU and paging monitor with per-block exclusion is not: the standard records host state at session boundaries. A new comparison of that cohort should be a profile with a reference adapter |
| `benchmark_stock_serving.py` | Server; one engine at concurrency 1, 4 and 8 | Within-engine scaling. The standard's burst is the cross-engine concurrent row, reported only |
| `benchmark_stock_scoring.py` | Scoring workers | Out of scope: the standard covers generation |
| `benchmark_glm_mlx_serve.py` | Server, engine phase time; MLX-Serve alone on GLM | Becomes the reference arm of a GLM profile; kept until that profile exists |
| `benchmark_hybrid.py`, `benchmark_hybrid_context.py` | Worker; fixed-context prefill and decode of one build | Optimization harness. Use it for same-engine A/B work; cite its figures as worker-level |
| `benchmark_qwen38_mtp.py`, `benchmark_qwen38_pilot.py`, `benchmark_qwen38_dense.py`, `benchmark_qwen38_moe_crossover.py`, `benchmark_lily_release_baseline.py` | Worker; paired same-worker arms for drafting, schedules and kernels | Optimization gates under the playbook's measuring rules. Not scoreboard results |
| `benchmark_lily_fixed_corpus.py` | Worker; another engine's published corpus and request ladder | Reproduces a published protocol's conditions, as the standard's client does for MLX-Serve. Kept until that engine is a reference adapter |
| `benchmark_idle_wake.py`, `benchmark_qwen38_cadence.py` | Worker; idle wake and request cadence | Host-condition studies. They are the reason the standard has a sustained cell and reads paired blocks |
| `prefect_benchmark_run.py` | Inherited upstream dispatch, disabled | Unrelated |

Records made with those scripts keep their original scope and are not regenerated.

## Still open

The standing rule, the quality bar, the burst's status, the reference-engine policy, the block layout, the image rule and the arm names were decided on 2026-11-23 and are stated above. What remains:

- the reference engine and matched packs of every family other than Inkling 2 Mini, which are chosen by the policy above when that family's profile is written;
- whether KL divergence or the log-probability change should have a bar of their own; today they are reported only;
- the quality probe's own interface, which is still changing, so the converter for its output is provisional;
- continuous host monitoring with per-block exclusion, which the older stock-engine harness had and the standard does not.
