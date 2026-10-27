# Qwen3.5 Developer ID worker execution qualification

This record extends the [native image execution qualification](2026-10-26-qwen35-image-execution.md)
to the Developer ID signed generation worker prepared for the Swift application.
It does not qualify the application bundle or claim notarization. Application
installation and Swift integration have separate evidence.

## Source and exact tuple

The engine and generation-worker source are unchanged from
`c9d24754f553cf7714ddd3a99ecc9544493c407e`. Rust 1.98.1 produced the release
generator separately from the optimized scorer. Signing used the existing
distribution signing helper, Developer ID team `S5BT2PRN5B`, hardened runtime,
and a secure timestamp. Models and raw execution artifacts remain private.

| Identity | SHA-256 or value |
| --- | --- |
| Model | `Qwen/Qwen3.5-0.8B`, BF16 |
| Source revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| Model weights | `7c5960e7225b3b36844fa397a85c5e20d281d732e382475062f552c0737c79d4` |
| Configuration | `3f0cecb7cb3633bc271eabf69b06ac9f74b377b9ec17b441f5ceac331a71f42a` |
| Tokenizer | `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4` |
| Processor | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| Template | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| Original qualified worker | `5fc92516c9f803df90316599a3930c1e2104cf320ae23361ff807e48bde08ec6` |
| Developer ID worker | `741a3ea7b39a80fcd9c9a6fe9deb0cee1d1b383bd40618479ec49a8a58d91ede` |
| Runtime | Metal, Apple M5 Max, Apple GPU family 1010, 40 GPU cores, MXU enabled |
| macOS | `26.7.1` |
| Transport | `mediaref-v1`, `owner-private-file-v1` |

## Delivered executable comparison

Both signatures were removed from private copies with `codesign`. The resulting
files differ in the `__LINKEDIT` segment's allocation `vmsize` field at byte
offset 1712: 2,523,136 versus 2,359,296 bytes. Clearing that single eight-byte
allocation field makes every remaining byte equal. The normalized SHA-256 is
`919ad88725ecb1048479208117c7bb96112ae9a19ad82f11bac23bd47c5fad02`.
This comparison preserves code and data; it does not assert that the raw stripped
files are identical. Their raw hashes are
`29101508b6219ab2e048fb4872d81d7e2302cf80b0b897d2e2bacff0764e0d37`
and `e1c3ec0fdc2512ef579b6adc426e224d6d4dd03d75088a910be774c15ac4b9d3`.

The private comparison result has SHA-256
`d88d07f0091adfc16e0f0516ab61ec7f477581407ad4e2af19180f083678eb95`;
its audit script has SHA-256
`ef87dd84cec7477566f638bd63ba09751b2a83cea6b4c242e04554a5870708be`.
The unchanged native computation retains the earlier bit-exact three-fixture
logit result and its unchanged 2% RMS gate. Those logits were not re-exported
from the production worker during this signing check.

## Execution replay

The signed worker completed all eight frozen public synthetic visual fixtures
through the actual native image protocol. No host qualification override was
used: the low-level worker API admits its exact artifact and reports its native
contract independently of the public host's execution catalog.

- Normalized answers match the pinned reference on **8/8** fixtures.
- Source and native answer accuracy are **6/8**; counting and OCR limitations
  remain present in both.
- Text-plus-image input counts match on **8/8** fixtures.
- Staged files after each completed request: **0**.
- The worker reports the exact five artifact hashes and runtime tuple above.

The pinned reference report has SHA-256
`c62855b797ff8a0d45d153131d38966d65bc9fa5e6621a170c5e3a770a99bea7`;
the fixture AST hash under Python 3.12 is
`87c7947de54266ca02f7287ae80d8632b8bb8cf9b34312c309dbe19a163551b8`.
The native comparison report has SHA-256
`f6919ac60ce64eda2589c403ed9fa96e35fc54326ed738843ef489aa9a5ec53d`;
the unchanged comparator has SHA-256
`6c2db8787801554ad3c275e28098c489e1fb3b7029a28ef1d87f009b86b3c605`.
An initial invocation with a nonexistent reference-report path failed at startup;
its failure record was retained separately before this successful replay.

## Catalog boundary

These results support one additional exact execution-catalog record for the
signed worker. The original record and evidence remain unchanged. Public image
capabilities still require all five artifact hashes, the launched executable's
hash, the device class, and the exact OS version to match. Re-signing the worker
changes its identity and requires a new record. Resealing only the enclosing app
does not change the qualified worker bytes.
