# OpenWork first-request authentication recovery

Date: 2026-11-06

## Source and scope

- Stark baseline: `b263442b2e896303e2d86a87925359b927a5cdd0` plus the connector lock update recorded with this document.
- Previous connector: `jshahbazi/stark-openwork` revision `a5534db0526e08c9b9b4df12825c9da7117cae86`.
- Fixed connector: revision `23f0c850ed204ffdd83247ead17ad6bcccf1a1d2` on `codex/startup-auth-hook`.
- Installed `connector.js` SHA-256: `09dd60ccd1dd788c3ef5ad28b7c6b23033b4f05077a86b77dd4e0a450443eac9`.
- OpenWork: signed version/build `0.18.65`, from `runtime/desktop-agent-validation/openwork-unpacked/OpenWork.app`.
- Model: existing `qwen3-4b-4bit`, in Chat mode.

This record qualifies startup authentication for that local OpenWork copy. It
does not qualify tool execution, OpenCode Desktop, or other client versions.

## Failure and cause

A cold OpenWork launch followed by a message as soon as the composer became
available produced a new HTTP 401 API-key error. Later requests succeeded. The
earlier successful warm requests did not establish reliable recovery.

OpenCode batches provider transforms during plugin startup. The transform
callback can run after the plugin's `setup()` returns. The previous connector
set `providerActive` inside that callback, but used its initial false value to
decide whether to register the authentication hook. The boot batch then
published the selectable model without its hook. The first five-second poll
installed the missing hook, explaining the timing-dependent result.

The fixed connector registers its provider-scoped request hook before the
provider transform and keeps it until teardown. Foreign providers are left
unchanged while the connector does not own the published provider. Hook
registration failure aborts setup before provider registration; provider
registration failure disposes the hook.

Upstream contract references, pinned to OpenCode `v2.0.26`:

- [State transforms and batching](https://github.com/anomalyco/opencode/blob/v2.0.26/packages/core/src/state.ts)
- [Promise plugin adapter and boot batching](https://github.com/anomalyco/opencode/blob/v2.0.26/packages/plugin/src/promise/adapter.ts)
- [Immediate hook registration](https://github.com/anomalyco/opencode/blob/v2.0.26/packages/core/src/plugin/hooks.ts)

## Validation

- The new deferred-boot regression fails against the immutable previous
  connector: a model is published with zero active authentication hooks.
- Fixed connector suite: 16 tests passed with Node `24.15.0`. Coverage includes
  first-request authentication before any poll, late connection, foreign
  provider passthrough, registration failures, cleanup, model transitions,
  endpoint restrictions, session rotation, and broker protocol handling.
- Stark repository policy check passed.
- App build tests: 46 passed.
- Built and installed with `scripts/build_menubar.py --install --open
  --openwork-connector ../stark-openwork`. The local source matched the complete
  immutable connector pin; the build manifest records no development connector
  override. The installed connector hash matches the lock above.
- Installed app passed `codesign --verify --deep --strict` with its Developer ID
  signature, hardened runtime, and secure timestamp.
- Live UI replay: three complete quit/reopen cycles of the specified OpenWork
  copy, each followed by a first message sent as soon as its startup controls
  allowed it. All three completed with model replies and no authentication
  error. One cycle required refreshing the accessibility tree because the
  initial workspace controls were replaced during startup.

The application build manifest records the Stark baseline as dirty because the
new connector lock had not yet been committed when building. The connector
itself is bound to the revision and hash above. No keys, capabilities, prompts,
responses, or token data are included in this record.
