# Manual Chat / Decision playground validation

Run date: 2026-10-22 (America/Los_Angeles).

This run covers the menu-bar app's manual mode selection, private desktop controller, and generic letter-choice scoring path. It records new evidence separately from previous scoring acceptance and generation qualification.

## Behavior

The playground retains a manual Chat / Decision selector. Decision mode accepts text, a question, and 2–16 comma-separated answer labels. The caller maps labels to distinct single-token letters, renders the selected model's converted chat template with thinking disabled, and requests next-token scoring at temperature 1.0. The interface displays the highest-scoring answer and the distribution across the supplied answers.

The generic prompt does not select or implement a model-specific decision profile. The readout has no task accuracy or calibration qualification. Model-specific published benchmark results do not transfer to this prompt or app.

Changing modes unloads and reaps the previous native worker before loading its replacement. Each decision is serialized as a complete operation. Inputs, generated replies, token IDs, and decision probabilities remain in memory and private pipes; only model registration and the manual mode preference are persisted. The scorer starts without the optional consecutive cache.

## Checks

| Check | Result |
| --- | --- |
| Swift arm64/macOS 26.4 type-check | Passed |
| Rust 1.98.1 release build, locked dependencies, `stark-score` and `stark-generate` | Passed |
| Full converter test suite | 4,002 passed, 88 skipped |
| Focused decision/desktop protocol and lifecycle tests, included in full suite | 117 passed |
| Menu-bar packaging checks | 21 passed |
| Swift controller event/lifecycle regression checks | 7 passed |
| Ruff on changed Python backend/tests | Passed |
| App packaging, embedded imports, worker self-checks and local signing | Passed |
| Packaged native round trip | 7 aggregate checks passed |

The native smoke used the existing converted Qwen3.5-2B Q4 model through the packaged Python controller and newly built workers. It checked model registration, chat loading, a chat request, decision loading, a three-answer decision request, chat reloading, and worker shutdown. It verified answer ordering, normalized finite probabilities, and termination of each replaced worker. It did not measure decision correctness, calibration, throughput, or compatibility with other models. Inference inputs and outputs were not written to validation artifacts.

Review subsequently found a Swift state-recovery edge case: a backend validation rejection could leave the form busy. The final Swift source handles that error, passed seven controller event/lifecycle checks, and was type-checked again; the app was rebuilt. The native smoke's Python sources and worker binaries are unchanged. The native smoke and final rebuild manifests are identified separately below.

Commands for the model-free checks and build:

```sh
xcrun swiftc -typecheck -target arm64-apple-macosx26.4 apps/menu-bar/Sources/StarkApp.swift
PATH="$HOME/.cargo/bin:$PATH" cargo +1.98.1 build --release --locked -p stark-score -p stark-generate
uv run --project tools/converter --locked pytest tools/converter/tests -q
uv run --no-project --python 3.13.13 python scripts/test_build_menubar.py
uv run --project tools/converter --locked python scripts/test_menubar_controller.py
```

The app was built with `scripts/build_menubar.py`, an extracted, manifest-verified 0.1.0 CLI runtime, and the fresh workers. The installed application was not replaced. The ignored local build is `artifacts/menu-bar/Stark.app`.

## Source and binary binding

The build was made from base revision `04a5fd1a896362849cc4219540e5b6afa63a5ace` with the implementation changes present. Its menu-bar manifest records a dirty source snapshot and file hashes. The following hashes bind the tested implementation and binaries independently of later documentation changes.

| Input | SHA-256 |
| --- | --- |
| Final `apps/menu-bar/Sources/StarkApp.swift` | `793fe5e8263c373231c8dbcd0898a3f891632a41a905e8f13a274109947c80b1` |
| `tools/converter/stark_convert/desktop.py` | `c105fe3fa342829f2d1be0df68c0c724f5bbfc26d825782eb5ecedeeafdf97a8` |
| `tools/converter/stark_convert/decision.py` | `ac763971f7e5cf1096dd640feb333f25cb22a13eb6c0db505fef6a8c90ac103a` |
| `scripts/build_menubar.py` | `8184818ff460564f978c120102a9f532bb9b195d3d6e80789b25065d746c2f50` |
| Packaged `stark-generate` | `5d81d0a55aeda2d7b81be793396a6c23dbaca597c6efe2b2a94570b5b51ce2f9` |
| Packaged `stark-score` | `f422ccbb7983a4cf015146fb12c5dfaccf3a358b6ed271f011dd5ce6a582ecfa` |
| Native smoke `menu-bar-manifest.json` | `2f8a49373958dd6291c1ed6a26a23e48f12f2e83da3cfec92ba533c4c0469d51` |
| Final rebuild `menu-bar-manifest.json` | `9bad9f6a03ffc25b71c5d10d3018d1e25ec35f15432d8831c5bc425b001387a4` |

These checks establish package and runtime interoperability for this smoke. Native screen interaction was not exercised in this run.
