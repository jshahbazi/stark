# Drex Max v1 Q4 and JevK5 BF16 on local decision tasks

On the reused public JevBench cohort, Drex Max v1 Q4 answered **212/231
(91.8%)** correctly and JevK5 BF16 answered **202/231 (87.4%)**. Both answered
all 12 synthetic controls correctly. Each model produced exactly the same
probabilities in a second complete pass.

This compares two available local decision stacks on an Apple M5 Max with
128 GiB unified memory. Drex is a 27B model on its upstream MLX runner;
JevK5 is a 4B model on Stark. Parameter count, precision, prompt, attention,
readout and runtime all differ. This evidence supports Drex's quality advantage
on these cases. It does not isolate the benefit of a model architecture or
establish performance on fresh deployment data.

The triggering [Reddit post](https://www.reddit.com/r/machinelearningnews/s/WWp2jSU8dW)
announces **Drex Max v1.1** and says its weights are coming later. The available
Hugging Face release is [Drex Max v1](https://huggingface.co/nace-ai/drex-max-v1),
including the [affine-Q4 artifact](https://huggingface.co/nace-ai/drex-max-v1-affine-q4)
tested here. This experiment does not reproduce the post's v1.1 result or the
Artificial Analysis Decision Index. Publisher benchmark numbers are separate
from these local measurements.

## Decision quality

| Metric, 231 public cases | JevK5 BF16 / Stark | Drex Max v1 Q4 / MLX |
| --- | ---: | ---: |
| Correct argmax | 202 | 212 |
| Accuracy | 87.45% | 91.77% |
| Multiclass Brier sum, lower is better | 0.184880 | 0.120900 |
| ECE, ten bins | 0.029446 | 0.043681 |
| Synthetic controls | 12/12 | 12/12 |

Drex alone answered 15 public cases correctly; JevK5 alone answered five.
The paired accuracy difference is **+4.33 percentage points**, with a
10,000-resample paired percentile interval of **+0.43 to +8.23 points**
(fixed seed 20261108). The exact two-sided McNemar p-value is 0.041389.
The 231 cases are reused public development data. These statistics do not
correct for training overlap, task selection or other comparisons, and do
not establish reliable calibration or replacement thresholds on a new workload.
Repeated passes do not increase the independent case count.

## Observed latency and memory

**The comparative timing result is unqualified.** Both Drex blocks failed the
predeclared background CPU screen and recorded host pageouts/swapouts. Their
measurements are retained below. Both JevK5 blocks passed the sampled screens;
passing those screens does not establish physical GPU exclusivity or fixed
clocks. No quiet-host comparative speed ratio is claimed.

| Run order | Median score | p95 score | Sampled process-tree RSS peak | Timing screen |
| --- | ---: | ---: | ---: | --- |
| JevK5, block 0 | 47.1 ms | 516.5 ms | 8.28 GiB | Passed |
| Drex, block 0 | 308.7 ms | 4124.1 ms | 13.86 GiB | Failed |
| Drex, block 1 | 324.1 ms | 4575.9 ms | 13.86 GiB | Failed |
| JevK5, block 1 | 66.8 ms | 643.1 ms | 8.25 GiB | Passed |

Timings cover all 243 cases, one request at a time. Prepared inputs, native
forward execution, readout, response formatting and private pipe exchange are
included. Tokenization, model load and warmup are excluded. Jev uses a full
vocabulary readout; Drex uses a dedicated 255-way head and retains upstream
response formatting. Input lengths differ with the models' prompts: Jev's
median is 208 tokens (104–4033), Drex's is 168 (89–3965). Neither truncates.

RSS is sampled every 50 ms from process startup and sums the Python harness
and its descendants. Shared pages can be counted more than once. This is
neither exact Metal allocation nor minimum device capacity. The observed
model files occupy 8,413,483,320 bytes for Jev and 14,451,748,024 for Drex,
excluding tokenizers and Drex's 5,222,420-byte readout.

The first Drex block recorded 441 pageouts and 551,136 swapouts; the second
recorded 111 and 77,996. These are whole-host counter deltas, with no attribution
to a process or model. Maximum sampled background process CPU was 258.6% and
226.9%, above the 200% threshold. No background model activity crossed the
10% threshold. Sampling had no failures, and the whole run had no sleep events.
Continuous GPU/thermal/power isolation was not measured.

## Method and provenance

The four-case admission probe passed for both models. The full run scored
243 cases twice per model (972 measured requests) in Jev/Drex/Drex/Jev order.
Each block started a new worker, completed a first load/warmup and three
additional untimed warmups. The request context limit was 8192 tokens.
Output caching was disabled and every request used fresh model state.

Jev's input and candidate token IDs come from the sealed reference. Its
temperature is 1.22. Candidate IDs follow the option order in
`scoring_fixture.decision_options`, which is independently checked against
the sealed typed decisions. The dataset's answer-label array has a different
order for 119 choice cases; its 77 boolean cases use `yes/no` aliases.
The harness maps these aliases to `true/false` and aligns both models by
answer identifier. It never treats an answer-set ordering as an output-vector
ordering. Historical source-bound records remain unchanged.

Drex uses its pinned upstream `DrexMax.prepare` and `DrexMax.decide`, native
affine-Q4 group-64 tensors, BF16 control tensors and a float32 readout.
Its embedded temperature is 1.0087417364. Full-attention layers receive
`mask=None` and attend across the complete request; the DeltaNet layers remain
causal. Ordinary causal text generation does not reproduce this scorer.
Full precision probabilities are captured in memory before upstream rounds
its response fields. Generation prefix reuse is blocked by noncausal attention
in this external scorer: earlier states depend on later request tokens.
No Stark backend or production model support was added by this experiment.

| Identity | Revision or SHA-256 |
| --- | --- |
| Stark engine/build source | `7dc5183499eb9248ab6805474291980dbf0a57db` |
| JevK5 checkpoint | `c4f7fdb3aeab5582336406e78d3bef11bf98833d` |
| Jev converted weights | `ea3fd57de724a9cc34cce999f42e78f30245e11bdf18938377bd367d4bbf0dc3` |
| Drex upstream scorer source | `d2fd26cdf211f87b9a6b1261995528629c9e1401` |
| Drex Q4 snapshot | `23f21ac6e664d3edb32faa7b55b0c8a7fb09134a` |
| Drex Q4 weights | `e9b0439e8c0cd994b54487c114b5d8fe683f10d2d05cb0f9407b2d869cfe51ac` |
| Drex readout | `a89619d4ee1a61e6eec8580ed6792ca609b8972c240730dc3d193dfa00f069dd` |
| Dataset | `acf56888f47486fd8eba45d17a46372d65d911e2f57bc390474e7b41f37aae58` |

Rust 1.98.1 built the scorer with locked dependencies and `optimized`. The
separate uv environment used Python 3.13.13 with pinned MLX 0.32.4, MLX-LM 0.32.2 and
Transformers 5.22.0. Source, worker, config, tokenizer, harness and lockfile
hashes are in the [aggregate receipt](data/drex-jevk5-2026-11-08.json).

Current Jev readouts have a maximum absolute probability difference of
0.0551452 from the older sealed reference. The corrected sealed reference and current
build both score 202/231 public and 12/12 synthetic cases, but this does not
establish identical individual decisions or bitwise numerical equivalence.
The prior [source-bound parity qualification](../validation/SCORING_PARITY.md)
remains separate. This run's numerical repeatability was exact within each
model: maximum probability delta 0.0 across both complete passes.

## Validation and reproduction

Eight model-free regression tests pass, including shuffled choice labels,
legacy boolean aliases, invalid readouts and paired metric accounting.
Ruff lint/format and repository policy checks pass. Independent source review
verified the 243-case answer mapping and the noncausal MLX attention path.
The new locked research dependencies are included in the existing security
workflow's dependency inventory; hosted checks have not been run for this
branch by this report.

Use the [harness instructions](../../scripts/drex_jevk5/README.md) and private
configuration template to reproduce. Model weights, fixtures and runtime
artifacts stay ignored. No prompts, responses, token vectors or individual
decision probabilities are published. This is additional evidence; it does
not supersede earlier engine or scoring comparisons.
