# Menu bar application validation — 2026-10-21

## Scope and build identity

Validated the native Stark menu bar controller on an Apple M5 Max with 128 GB memory, macOS 26.7.1. The application targets native arm64 and macOS 26.4 or later. The source starts from `ea668f80` with the menu bar changes recorded on `codex/menu-bar-app`.

The installed application is self-contained at `~/Applications/Stark.app`. It copies the existing standalone CLI 0.1.0 runtime and preserves that distribution's manifest and notices. It adds the current desktop/chat modules, the development generation worker, and exactly locked Jinja2 3.1.6 / MarkupSafe 3.0.4 wheels with licenses. The builder checks wheel hashes, original runtime hashes, imports, worker startup behavior and the final local app signature. It records app overlays separately from the base distribution.

| Component | SHA-256 |
| --- | --- |
| Development generation worker | `f12fd9849d81610170c2b893e98360b25ab553310cd5276e266f6d5e583e320f` |
| `generation.rs` | `3d1243c06c186d076140f7a0a664cc95420c92ca7cf00ab5da221c0922144c14` |
| `native.rs` | `2998967405c172b1f32ba46fc8dd3dac05fdd7d8ea3ddd0b9843e80a3e2502a5` |
| `StarkApp.swift` | `3de286f68cee6291222d95d4d62ed45a11d93b55c62f60f45d4194a33d7ba3e7` |
| `desktop.py` | `caff322eb43d8270f71702f6f9f389dbdc9a5aafe20f2fb95f7e244261d7a9b4` |

## Results

| Check | Result |
| --- | --- |
| Desktop/chat model-free suite | 29 passed |
| Packaging, dependency pinning and safe installation tests | 10 passed |
| Rust generation model-free tests | 13 passed |
| Generation process boundary checks | 6 passed |
| Changed Python lint/format and Rust nightly formatting | Passed |
| Swift type checking, optimized compilation, app signature verification | Passed |
| Repository policy gate | Passed |
| Converted local model registration and native generation | Passed |
| Positive finite input/output throughput fields | Passed |
| Native Stop/Run restart and backend shutdown | Passed |
| Real `Qwen/Qwen3-0.6B` Hugging Face acquisition and conversion | Passed |
| Load and generate with the newly converted Hugging Face model | Passed |
| GUI local folder picker and installation | Verified interactively |

Native workflow checks ran through the installed app's private Python runtime and desktop backend using a temporary private model library. The test retained aggregate outcomes and hashes only. Temporary converted models were removed after validation. No prompts, responses or token vectors were written to the validation report. The app's separate model library has an explicitly registered local converted Qwen3 folder.

The original cat JPG remains unchanged. The delivered transparent menu bar image was edited in Pixlr Express using cutout and color controls, preserving the original outline, whiskers and pupils. Its source hashes are recorded in the [app guide](../apps/menu-bar/README.md).

## Limits

A fresh `cargo +1.98.1 build --release --locked -p stark-generate` could not resolve the workspace's existing `rustls-webpki 0.103.16` dependency from the available index. No lockfile or dependency version was changed. The development worker compiles the changed generation/native source with Rust 1.98.1 and reuses previously compiled Metal engine libraries. Its private build record captures compiler commands, source/lock hashes and hashes for 105 reused libraries; that record is embedded in the app. This is a local development build, not a newly qualified release engine.

Input tok/s is a first-token wall-clock prefill proxy that includes setup and scheduling. Output tok/s measures sampled tokens after the first token, including a terminal stop sample. These last-request values do not establish benchmark parity or broad model qualification. The UI receives complete replies rather than partial streamed text.

The Mac locked before the final dropdown interaction check. The installation dialog and local folder registration were verified before that lock; full end-to-end dropdown interaction still requires an unlocked session. The app is locally ad-hoc signed, without Developer ID signing or notarization. These results do not extend earlier source-bound model qualification records.
