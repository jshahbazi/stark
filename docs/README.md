# Stark documentation

Reference pages are listed first. Records of past runs, milestones and experiments are under `archive/` and `benchmarks/` and are kept for provenance; they describe the builds named inside them, not the current code.

## Using Stark

| Page | What it covers |
| --- | --- |
| [Installation](INSTALLATION.md) | Install routes and what each verifies, source builds, prebuilt archives, updating, uninstalling, building an archive |
| [Menu bar app](../apps/menu-bar/README.md) | Building, updating and using the app, Connection Details, where the app keeps state |
| [Converter guide](../tools/converter/README.md) | `stark convert` usage, every option and limit, supported models, MLX and GGUF routes, offline replay, errors |
| [Trained Qwen checkpoints](QWEN_TRAINED_CHECKPOINTS.md) | Converting and testing fine-tuned Qwen3 4B and Qwen3.8 4B checkpoints |
| [Local API](LOCAL_API.md) | The loopback OpenAI-style chat API: authentication, endpoints, parameters, limits, error codes, tested clients |

## Integrating with the workers

| Page | What it covers |
| --- | --- |
| [Worker protocols](WORKER_PROTOCOLS.md) | Conventions shared by both workers: framing, readiness and version tags, error classes, privacy |
| [Text generation](GENERATION.md) | `stark-generate` requests, replies, streaming, stop sequences, cancellation, error codes, and the terminal chat client |
| [Native image implementation](MULTIMODAL_IMPLEMENTATION.md) | Native Qwen3.5 conversion, preprocessing, tower, fusion and image transport |
| [Vision model families](VISION_FAMILY_CONTRACT.md) | Merged BF16 Qwen3.5 admission, automatic installation checks and evidence levels |
| [Probability scoring](SCORING.md) | `stark-score` requests, replies, the consecutive cache, error codes and the digest probe |

## Developing Stark

| Page | What it covers |
| --- | --- |
| [Contributing](../CONTRIBUTING.md) | Toolchains, local checks, test layout, running a model-bound smoke test, where validation records go |
| [Technical guide](TECHNICAL_OVERVIEW.md) | Components, engine module layout, feature flags, Python package layout, the app control protocol |
| [CI and security checks](CI.md) | Workflows, what they cover and do not cover, dependency gates, caches, local equivalents |
| [Releasing](RELEASING.md) | Signed and prebuilt release workflows, verification on a receiving Mac, DMG handling |
| [Cargo troubleshooting](CARGO_TROUBLESHOOTING.md) | Repairing a corrupted registry cache |
| [Security policy](../SECURITY.md) | Reporting, supported code, trust boundaries and invariants |
| [Upstream provenance](../provenance/UPSTREAM.md) | Fork point, import, naming and compatibility notes |

## Evidence

| Location | What it holds |
| --- | --- |
| [benchmarks/](benchmarks/README.md) | Engine comparisons with a current-versus-superseded index, optimization records, and [conversion benchmarks](benchmarks/CONVERSION_BENCHMARKS.md) |
| [validation/](validation/) | [Scoring parity](validation/SCORING_PARITY.md) against frozen Uzu references and [ColGrep generation acceptance](validation/GENERATION_COLGREP.md), each bound to a named build |
| [archive/conversion/](archive/conversion/) | The converter replacement plan, milestone packets and their reviews |
| [archive/runs/](archive/runs/) | Dated validation and review write-ups, security baselines, CI measurements, website deployment and history-rewrite records |
| [archive/experiments/](archive/experiments/) | Protocols, raw results, patches and helper scripts behind the benchmark studies |
| [brand/](brand/README.md) | Logo provenance and colour tokens |

New validation runs belong in the pull request that made the change, or under `archive/runs/` when they must be preserved; reference pages describe current behaviour only.

The [Qwen3.5 family installation run](archive/runs/2026-10-27-qwen35-family-installation.md)
records changed-weight conversion and automatic local image checks through the
desktop, API and Swift controller.
The [config-derived Qwen3.5 family run](archive/runs/2026-10-27-qwen35-config-driven-family.md)
separates larger-model conversion/runtime evidence from metadata and operator
coverage for other sizes.
