# 20261122-INKLONGPOS-01 · Inkling 2 Mini verifier diverged from serial decode above about 13,000 positions

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date | `20261122-INKLONGPOS-01` / 2026-11-22 / 2026-11-22 |
| Kind | Correctness defect and fix |
| Attempt state | Completed |
| Result and acceptance criterion | Passed after the fix: drafting on equals serial output and committed state at every tested length through N = 16,000, and stock 4K parity is unchanged. Before the fix the same check failed at N = 14,000 and above. |
| Evidence scope | Full-model self-consistency (verifier against serial decode), full-model numerical parity on two pinned stock 4K captures, offline float32 arithmetic |
| Promotion decision | Fix adopted within the source-path scope; the long-position form remains unqualified against a reference |
| Disposition | Adopted on branch, uncommitted on top of `f1666090`. Not merged, released or installed. |
| Predecessor / related attempts / PR | Found by [20261122-INKDEFAULT-01](2026-11-22-inkling-mini-default-fast-paths.md). The fix build also carries [20261122-INKFUSE-01](2026-11-22-inkling-mini-launch-fusion.md). No PR. |

## Symptom and bisect

With drafting on, the committed state differed from serial decode while the emitted tokens matched. The `--measure-verify-transaction-schedule` check, which verifies 96 outputs from position N to N + 94, bisected the defaults-only build (`stark-generate` beginning `2da8222d57cf5080`, `inkling_slice` example beginning `b45cbce0d2dbff70`):

| N | Context | Last verified position | Result |
| ---: | ---: | ---: | --- |
| 8,800 | 16,384 | 8,894 | Pass |
| 10,000 | 16,384 | 10,094 | Pass |
| 12,000 | 16,384 | 12,094 | Pass |
| 14,000 | 16,384 | 14,094 | Fail |
| 15,000 | 16,384 | 15,094 | Fail |
| 16,000 | 16,384 | 16,094 | Fail |
| 16,000 | 32,768 | 16,094 | Fail |

The first state mismatch followed the two-row round committed at position 14,031. It appeared in visit 5, the first global-attention layer, while keys and values were still identical.

## Cause

From position 8,192 the global layers scale the query and relative head by 1 + 0.1 × log((position + 1) / 8192). The serial path (`ink_packed_heads_source` in `packed_attention.metal`) had no contraction pragma on that line, and the compiler fused it to a single multiply-add. The verifier (`ink_verify_qr_pack` in `verify_cache.metal`) computed it under `#pragma clang fp contract(off)` and rounded twice.

Offline float32 arithmetic gives the positions where the two forms differ in each check window:

| Window | Differing positions |
| --- | --- |
| 8,188, 8,800, 10,000 and 12,000 | None |
| 14,000 | 14,003 and 14,029 |
| 15,000 | 15,038, 15,054 and 15,073 |
| 16,000 | 16,023, 16,049, 16,057 and 16,089 |

The share of differing positions grows with position: about 0.2 percent just above 8,192, 0.8 percent near 10,000, 1.1 percent near 12,000 and 2.5 percent from 13,000. The passing and failing windows agree with this list. A difference only changes committed state when a verifier row lands on one of these positions, which explains why the earlier windows passed.

## Fix

One shared helper, `ink_long_position_gain` in the new [`long_position.metal`](../../crates/stark-engine/src/engine/inkling_slice/long_position.metal), computes the gain as an explicit `fma`, so the result no longer depends on the caller's contraction pragmas. It is used at `packed_attention.metal`, `serial.metal` and both sites in `verify_cache.metal`, and it is prepended to each translation unit that uses it.

With only the verifier changed to the single-rounding form, N = 14,000 passed. The serial path already used the single-rounding form, so its arithmetic is unchanged by the fix. The incumbent singleton kernel in `serial.metal` (`ink_heads_and_cache`) also had no pragma on that line; its switch to the helper is expected to be neutral for the same reason, but that kernel was not checked separately.

## Verification after the fix

The fix build's binary hashes are not recorded.

| Check | Inputs | Result |
| --- | --- | --- |
| `--measure-verify-transaction-schedule` | N = 14,000, 15,000, 16,000 at context 16,384; 16,000 at 32,768 | Pass |
| Same | N = 73, 511, 4,150 at context 4,352; 8,800 at contexts 9,216 and 16,384; 12,000 at 16,384 | Pass |
| `--full-target-batched` | `s4096-00` and `g4096-02` | Exit 0 with exact logits and hidden state |
| `--measure-verify-transaction` | `g4096-02` | Pass |
| Independent rerun by a second party on a fresh build | 4K parity on `g4096-02` and `s4096-00`; transaction-schedule checks at N = 16,000 and N = 73 | All exit 0 |

## Limits and open items

- The single-rounding form has never been compared with the pinned reference past position 8,192, because no reference capture reaches that far. Both paths now agree with each other; whether they agree with the reference above 8,192 is unknown.
- No timing was taken; the change is not a performance attempt.
- No audit of other expressions that differ only in contraction between the serial and verifier kernels is recorded.
- Performance playbook: not applicable.

Keep prompts, completions, token vectors and probabilities out of this document; the private reports stay ignored.
