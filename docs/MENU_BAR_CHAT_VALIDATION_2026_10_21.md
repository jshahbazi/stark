# Menu bar chat validation — 2026-10-21

The small chat interface adds visible user/model messages, a multiline composer, model selection and Run/Stop, New Chat, automatic scrolling, and last-request input/output tok/s. Command-Return sends a message. Messages and drafts stay in memory. Switching models or stopping clears conversation history; a failed or cancelled pending message returns to the composer for retry.

## Source and runtime

Validated `apps/menu-bar/Sources/StarkApp.swift` SHA-256 `30d1d6322816c3c689cdea9ed11d2e4b5a10cde3d71b8f05aea279c0cd8954ab`. The optimized application was rebuilt and installed at `~/Applications/Stark.app`; its local signature verifies.

The generation worker remains SHA-256 `f12fd9849d81610170c2b893e98360b25ab553310cd5276e266f6d5e583e320f`. The [initial menu bar validation](MENU_BAR_VALIDATION_2026_10_21.md) records its cached-engine development build, dependency-resolution limitation and timing semantics. This UI change introduces no engine or dependency changes and does not extend engine qualification.

## Results

- Swift type checking and optimized app compilation passed.
- All 10 packaging checks passed; repository policy and diff checks passed.
- A headless integration harness compiled the exact application source above, replaced only its application bootstrap, and drove the actual `EngineController` against the bundled backend and an existing local Qwen3 model.
- All 18 controller checks passed: startup/library loading, two complete conversation turns, user/assistant transcript order, positive speed measurements, New Chat, Stop/Run restart, oversized-message rejection, cancellation/draft recovery, and suppression of stale replies.
- The second request's input count grew from 19 to 43 tokens; New Chat restored the first-turn count to 19, confirming that both the visible transcript and backend conversation reset.
- The model registry stayed unchanged. The harness removed its test preferences and left no child processes or stderr output.

The harness retained aggregate outcomes and hashes under ignored `artifacts/menu-bar/chat-validation/`; it did not retain conversation content. Final interactive appearance verification was blocked by the locked Mac. Existing source-bound validation records remain unchanged.
