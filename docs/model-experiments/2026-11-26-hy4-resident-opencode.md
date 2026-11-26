# 20261126-HY4OC-02 · Resident AngelSlim Q3_K through OpenCode

## Identity and change

This run repeats the three OpenCode 2.4.2 synthetic coding turns from the
[tuned streaming study](2026-11-26-hy4-opencode-defaults.md) using the same
Stark backend, generator, AngelSlim Q3_K checkpoint, BF16 projector and NextN.
It compares the existing measurements with one new resident run on the same
128 GiB M5 Max and macOS 26.7.2. The two runs took place in separate sessions.

The resident run set `STARK_HY4_LOAD_PROFILE=resident-decode` and
`STARK_HY4_PREFILL_CHUNK=4096`. The context limit stayed unset. The worker
reported mapped expert residency, the Q5_0 draft head and 118,784 loaded
positions. The earlier tuned streaming run fitted 221,184 positions. Both
runs used greedy depth two, a 96-token output budget and the identical
synthetic prompts and tool schema.

## OpenCode timing

| Turn | Total input | Reused | Prefill tok/s, streamed → resident | Decode tok/s, streamed → resident | OpenCode seconds, streamed → resident |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh 16K | 16,000 | 0 | 444.62 → 484.59 | 18.79 → 38.67 | 41.791 → 36.171 |
| Continued 27K | 26,950 | 12,288 | 476.24 → 500.06 | 20.79 → 35.74 | 36.001 → 32.864 |
| Fresh 27K | 27,000 | 0 | 482.51 → 495.62 | 19.12 → 40.77 | 61.566 → 57.563 |

The resident run matched every rendered input and generated output digest.
It matched the draft counts and reused token counts across all three turns.
The worker reported zero software expert reads and zero decode host cuts
on all resident turns. Startup including desktop image qualification took
97.502 s, compared with 16.938 s on the earlier streamed run. All 29 pressure
samples reported level 1, with swap at 33.69 MiB.

The fresh 27K resident decode GPU-span counter returned more than 20,000 s
for a 2.33 s decode. The table uses the worker's wall-clock phase timings
and OpenCode invocation wall time. Exclude that GPU-span sample. The
prefill differences include possible session drift. This is a three-turn
decode improvement on the measured fixture with a larger startup cost
and a smaller memory-fitted window.

## Evidence

| Item | SHA-256 |
| --- | --- |
| Shared release worker | `c0c89f4c37c5c47ecb56c205dfdecab7d28e18599aed675e283aa04a3f60b588` |
| Resident aggregate | `139aeedf00125184abef17d374869716a81695a152e7549e58cf5510868be51f` |
| Resident host pressure samples | `ebc9eced03c66853aef5f656f256f7c3d515c6f7d2f579924cd80c7edea470ab` |
| Harness with aggregate stage counters | `815bfdcc60559097133b1b667581690f7b00c4f3f2717495379d878d440ba0b4` |
