# Generation time estimates: 2026-10-31

## Scope and source

This run checks the built-in Chat playground's time-to-first-token display and remaining-completion estimates. Native streaming progress carries token counts and timings through the Python bridge to a request-fenced Swift view. Calibration retains bounded counts and durations in memory for comparable requests in the same model session. A cold request learns decode speed as tokens arrive; a later comparable request can predict first-token latency before its first sample.

The implementation was checked on `codex/openwork-secure-connection`, based on `30574579` (`docs: record window and agent streaming validation`). The following SHA-256 hashes identify the runtime sources used for these checks:

| Source | SHA-256 |
| --- | --- |
| `crates/stark-generate/src/generation.rs` | `edbf251c5bfc5546198217b1230379159c0d9e0a20387aa200a0a02e281548ab` |
| `crates/stark-generate/src/native.rs` | `bac8c8193918f9cabedbedcc12aa1924397fc66470f3c6c34def0e65fb04e1a5` |
| `crates/stark-generate/src/hybrid.rs` | `01f062be9e491ab0ced24302e9298560a2b50b66c562c2471226eca4dc81725b` |
| `tools/converter/stark_convert/generation_eta.py` | `42752d412c01cf2dae06239762a01110f29536531e03038b1a558948e8a1127d` |
| `tools/converter/stark_convert/chat.py` | `ede5672983c0f68636e4827187cba353f61299e521d3d2f2095557ce560bcfd4` |
| `tools/converter/stark_convert/desktop.py` | `b1099b8061725db9755614c1c1c4bd589e29dbef46639a56c19df3e9755a8bdd` |
| `apps/menu-bar/Sources/StarkApp.swift` | `94759da704dcbbf3026cad9f9f17cbb269e69246f2b60e39a1c37eceb43546e5` |

## Checks

| Check | Result |
| --- | --- |
| Full offline converter suite, Python 3.13.13 | 5,206 passed; 88 skipped |
| Generation worker default-feature library tests, Rust 1.98.1 | 69 passed; 2 existing benchmark tests ignored |
| Generation worker library tests without default features | 48 passed |
| Generation worker Clippy, all targets, warnings denied | Passed |
| Generation process boundary harness | 6 passed |
| Swift controller harness | 116 passed |
| Full production Swift view/window compile harness | Passed |
| Menu-bar build script tests | 35 passed |
| Python Ruff, repository policy and pinned Rust formatting | Passed |

Coverage includes EOS and stop-token accounting, context-clipped budgets, callback failure cleanup, unchanged nonstreaming replies, monotonic progress, cold estimates, live speed changes, output-length overruns, profile isolation, malformed events, cancellation and model/request replacement. The full-view check compiles the actual production views. Disposable view snapshots checked cold, predicted-prefill and measured-decode states at the minimum 500-point playground width; both timing rows fit without clipping. Connected-app activity keeps its existing status message and does not display a pending playground estimate.

## Model-bound smoke checks

The final-source release worker was built with Rust 1.98.1 and the locked dependencies. Its SHA-256 is `db16856ed0ffa218e69dd50a1d4a0b70cff3fc44929cdf1b94fa10fca2c5f89b`.

An existing converted Qwen3-0.6B artifact was used without downloading weights:

| Artifact provenance | Value |
| --- | --- |
| Source revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Source manifest SHA-256 | `e199732d889006d43907f5ebe0b09d136f3da4bb27482efd2b735a3128d0d5c7` |
| Converted `config.json` SHA-256 | `dfa5763dc220e871b86a6119536f5c1c2862a5137a473eaf9d812be3cc53924e` |
| `conversion.json` SHA-256 | `c15ceb81c6da20ea2e7bd48cd70b7995eefb8fbbcc55365fc537c72fedbd3048` |

A synthetic native streaming request used 14 input tokens and a 24-token output budget. It finished at the length limit and produced 25 progress frames: the initial zero count followed by all 24 sampled counts. Counts and elapsed times were monotonic, input/budget fields stayed fixed, and the first sampled progress time exactly matched terminal prefill time. Aggregate wall-clock observations were 0.021777875 seconds to first sample, 0.065478 seconds for decode, and 0.090970208 seconds for the generation operation.

A separate two-request smoke exercised the real Desktop-to-worker path with a disposable model registry. Each request produced 25 progress events through 24 sampled tokens. The cold request acquired live completion estimates on 23 updates. The next request had a historical first-token range before its first sample and completion ranges on all 25 updates. The worker was stopped and the disposable registry removed after the run.

Only aggregate results and hashes are recorded here. Prompts, responses, token vectors and private runtime artifacts remain excluded.

## Limits

These checks establish progress delivery, display behavior and estimator adaptation. Prediction accuracy across workloads remains unqualified. The ranges use empirical timing/length history with heuristic padding; they carry no statistical coverage guarantee. Generation timing excludes model loading, queueing, prompt tokenization and CPU image preprocessing. Vision-tower and decoder inference are included. Native image progress compiles through the same callback path and has model-free regression coverage; this run used text inference only.
