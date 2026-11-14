# RAM-scaled Speed recommendation check — 2026-11-11

Source base: `67e14db0` on `codex/beginner-guided-client-choice`; this record describes the Speed policy change applied after that commit. No model was loaded.

## Policy

Speed requires a cached parent-model score >=40 for General Work or >=35 for Coding and exact export weights <=25% of physical RAM. Among candidates passing those limits and all normal resource checks, the highest score wins; ties prefer lower runtime memory, then weight bytes. All alternatives meet the same limits. Missing evidence does not waive a limit.

The score is a parent reference, not a quantization-specific accuracy result. The budget counts the actual export, including its precision. No speed measurement or bit-width quality penalty is fabricated.

## Scenario

Fresh library with no installed artifacts, 32K guide context, free RAM equal to physical RAM minus 2 GiB, and 4,000 GiB free disk/cache space. Synthetic unified-memory Metal budgets use 67% of physical RAM through 32 GiB and 75% above. Real devices can differ. Candidate discovery is a bounded, incomplete metadata scan; the lists are an illustrative source-bound snapshot, not exhaustive model coverage.

Metadata scan: 18 families, 270 entries, 36 planned candidates, incomplete=True. Input SHA-256: `57f5875fc41041551b80c1ef597d580edbf86c3920a5802fedb7ed4c4a7ed3cb`. Cached leaderboard captured 2026-11-11.

## Selected primary recommendations

| RAM (GiB) | Weight budget (GiB) | General Work Speed | Coding Speed | General Work Quality | Coding Quality |
|---:|---:|---|---|---|---|
| 8 | 2.0 | None qualifies | None qualifies | None qualifies | None qualifies |
| 16 | 4.0 | Qwen2.5-Coder-1.5B-Instruct | Qwen2.5-Coder-1.5B-Instruct | Qwen2.5-Coder-1.5B-Instruct | Qwen2.5-Coder-1.5B-Instruct |
| 24 | 6.0 | Qwen3 4B | Qwen3 4B | Qwen3 4B | Qwen3 4B |
| 32 | 8.0 | Qwen3 4B | Qwen3 4B | Qwen3 4B | Qwen3 4B |
| 48 | 12.0 | Qwen3 4B | Qwen3 4B | Qwen3.8-27B-oQ4e-mtp | Qwen3.8-27B-oQ4e-mtp |
| 64 | 16.0 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | Qwen3.8-27B-oQ4e-mtp | Qwen3.8-27B-oQ4e-mtp |
| 96 | 24.0 | Qwen3.8-27B-oQ4e-mtp | Qwen3.8-27B-oQ4e-mtp | Qwen3.8-27B-oQ4e-mtp | Qwen3.8-27B-oQ4e-mtp |
| 128 | 32.0 | Qwen3.8-27B-oQ4e-mtp | Qwen3.8-27B-oQ4e-mtp | Qwen3.8-Flash-Next-lily-q4-xl | Qwen3.8-Flash-Next-lily-q4-xl |
| 192 | 48.0 | Qwen3.8-27B-oQ4e-mtp | Qwen3.8-27B-oQ4e-mtp | Qwen3.8-Flash-Next-lily-q4-xl | Qwen3.8-Flash-Next-lily-q4-xl |
| 256 | 64.0 | Qwen3.8-27B-oQ4e-mtp | Qwen3.8-27B-oQ4e-mtp | Qwen3.8-Flash-Next-lily-q4-xl | Qwen3.8-Flash-Next-lily-q4-xl |
| 512 | 128.0 | Qwen3.8-Flash-Next-lily-q4-xl | Qwen3.8-Flash-Next-lily-q4-xl | Qwen3.8-Flash-Next-lily-q4-xl | Qwen3.8-Flash-Next-lily-q4-xl |

## Variants and evidence

The 27B choice is Jundot/Qwen3.8-27B-oQ4e-mtp (mixed 4/5/6/8-bit). Qwen3 4B selects mlx-community/Qwen3-4B-4bit. Flash Next selects the Lily q4-xl export. The selected Qwen2.5 1.5B/7B and Qwen3 8B catalog exports are BF16.

These recommendations preserve the exact source revision binding, but do not establish that a quantization retained the published parent score. Quality results remain unchanged by the Speed filter.

Candidate identity digest: `647333e8d9dbed377878c2fca739ff31a20c4e164b085f46665cc44bc7ee59e9`.

The current candidate pool limits growth: Speed can plateau at Qwen3.8 27B over several RAM tiers. Additional supported, scored models can improve that without changing the policy.

## Selected source identities

| Repository | Revision | Quantization |
|---|---|---|
| Jundot/Qwen3.8-27B-oQ4e-mtp | `7dd276b77fcb27c982ebd8d8a434539345713fea` | 4/5/6/8-bit |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | `2e1fd397ee46e1388853d2af2c993145b0f1098a` | BF16 |
| Qwen/Qwen2.5-Coder-7B-Instruct | `c03e6d358207e414f1eca0bb1891e29f1db0e242` | BF16 |
| Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | BF16 |
| fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl | `0f94842e3682b49acace4daeb4cf2e65a6f94f83` | Lily q4-xl (Q4 GDN; Q8 attn/shared/head/embed) |
| mlx-community/Qwen3-4B-4bit | `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` | 4-bit |

## Verification

- 222 Python recommendation, resource-profile and benchmark tests passed, including RAM growth, inclusive budget/score boundaries, absent evidence, unknown alternatives, hard admission and same-parent BF16/quantized export separation.
- 19 native beginner checks, 12 native window checks and 46 packaging checks passed.
- Repository policy, changed-Python lint and diff whitespace checks passed.
- Empty-report recovery was traced through the existing EngineController mapping of report.message to beginnerChoicesError and the host/view choicesError detail.
- This run used synthetic hardware and metadata only; no new model quality or throughput measurement was performed.
