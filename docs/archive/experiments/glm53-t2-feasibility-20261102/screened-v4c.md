# GLM-5.3 T2 Per-Pair Host-Screened Attempt v4c

Date: 2026-11-02. Status: **partial screened attempt; performance unqualified**. V4c adds a per-pair idle-admission host screen to a same-loaded-model pair harness over the exact dual-input candidate. It used an 1,800-second active-work budget followed by bounded cleanup.

## Correctness and setup gates

The initial candidate gates passed: eight ordinary-API alignment rows, four pooled pairs across all four residues, 192 complete hidden/logit comparisons, 192 compact-choice/hidden comparisons, and 12 scheduled full-row checks. The v4c source also passed 72 model-free Python protocol cases, two model-free Rust gates, and three GPU component gates. The release no-run build succeeded in 115.73 seconds. Those component and validation receipts are separate from the deadline-screened model result.

The screen required six accepted A-B-B-A matched blocks. It observed 25 prepared samples; 13 executed compact validation and 12 were included in the screened attempt. Ten individual samples met their per-pair host screen. Of six complete matched-block attempts, one was accepted and five were excluded; one additional block attempt remained incomplete. Twelve quiet-admission timeouts cancelled prepared samples. Their timing metrics were not retained, so their underlying admission causes are unavailable. One executed sample was excluded for external CPU at a screened maximum of 161.720516932545%; another was excluded after paging, with three pageouts across the screened session. Sample 25 was interrupted when the active-work budget expired, before its host screen.

## Accepted block and limits

The single accepted block had these aggregate A/B means:

| Metric | A legacy | B dual-input | B/A |
| --- | ---: | ---: | ---: |
| Wall time (ms) | 89.724688 | 95.2672495 | 1.0617729815900836 |
| GPU time (ms) | 61.03162500949111 | 58.75683334306814 | 0.9627276569144404 |

Within that block, A2/A1 wall time drifted by 22.65% (`1.2264696913020365`). B was 6.18% slower by wall mean and 3.73% faster by GPU mean in that one accepted block. The required six-block screen was not met, and the control drift is large. These values establish neither a performance win nor a regression magnitude.

The active-work budget expired while sample 25 was being prepared, before its host screen; bounded cleanup followed. Cleanup reported `stream_reader_failed=true` and native return code 101. The full-session final counters were not returned. Source, helper, binary, model, and weight identity post-guards passed; raw native output was not persisted. No request-level gain, runtime admission, rollback behavior, or production enablement follows from this partial attempt.

The initial v4c candidate proof recorded 80,740,352 bytes graph-scratch high-water; 108,692,065,600 bytes guarded required peak under a 109,676,289,225-byte cap; and 105,772,580,864 bytes sampled Metal peak. These values match the v3 gauges. They are probe observations, not runtime admission evidence.

## Source and receipt provenance

V4c is frozen from base `b4f1836d97c9fcfb154d803ba93f96b8255d18e4`. Its 17-file freeze hash is `68282efd08fb11cb75a4cca332ab8f0c53e052abf44be28be394b2a5c11394c0`. [The full source patch](screened-v4c.patch) has SHA-256 `9d8bad3c0a841fd18352e9f7d8db615f43ba7ae0fd1f4478f1800fb1f5fbe4bd` (144,533 bytes). It declares 16 changed paths, including ten new-file sections; all 17 freeze-recorded hashes matched after default full-context application from the pinned base, and reverse application restored the clean base. The source archive SHA-256 is `6591c656c528cdbd483931d31066aaa76e3a312d6367069edc9a0a678e8069cb` (152,419 bytes). The archive and runtime outputs are not included.

The release no-run build receipt is `b8e03e1fceda1b2cfadc811893de40c0f91edea2e14bde04c9967b5b56d1a1f7`; its sealed binary hash is `ca4bec5de3022e4ab2b508f160696a89a07842d1793e910e70ffda0f88d4d1db`. The Python protocol receipt is `d6c7b5c3e1750f18f209c1832b7323d696ced4fe3bfc87ccc6177e81c7654d01`; the Rust model-free gate result is `877e08257ca69a29397fa65f55499d22c7d06e9e20903194f58a4ac7221da786`; the GPU component gate result is `0adf708d21ddbe401029ee0058eca528ee0abcb2df624c0b5750f1a389b4d907`.

The screened run result SHA-256 is `55945d0e7cc12c8595b70e1b5acc7ca67a95a8ce7c60ee72db0326f2bf89204a`; the selected-safe aggregate SHA-256 is `404fe3816aa6a9a6449936e63c995f289f1ead1bcb6ebd0124ee879aee9b4202`. These are aggregate receipts; hashes, source-file hashes, counts, and timing aggregates are in [screened-v4c.json](screened-v4c.json). Prompt/response payloads, token IDs, vectors, logits, raw host rows, process names/PIDs, and raw native output were not persisted. The prototype source includes authored fixed test fixtures.

This is test-only evidence. No candidate is promoted or enabled by this attempt.
