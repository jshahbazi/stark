# Coding benchmark priority and local installation

Date: 2026-11-08. Implementation source:
`f3da1edf29ed90af539b796a481e98c78025dea1` on
`codex/beginner-hardware-ranking`.

## Corrected policy

The previous policy preserved unscored models' fallback positions. That left
unscored Flint ahead of Qwen3.8 despite the request to prioritize coding
benchmark evidence. The user explicitly requested correcting this behavior.

Coding with Balanced now applies hard admission first, then selects one coding
cohort with at least one eligible covered parent. Cohort selection uses distinct
eligible-parent coverage, publisher authority and ID; it does not cherry-pick
score magnitude. Candidates covered by that cohort rank first by descending
score. Equal scores use the existing deterministic fallback. Uncovered models
follow in fallback order, with explicit unknown-quality wording.

A single covered parent is sufficient to prioritize evidence. Multiple exports
of the same parent count once for coverage. A valid zero is recorded evidence;
absence is not converted to zero. Soft memory-headroom tiers cannot override
coding evidence after hard RAM, GPU, storage and route admission. If no eligible
reference exists, the guide explicitly labels its resource-based fallback.

This is an evidence-availability policy. It does not prove that an unscored
model has worse coding quality. Public upstream references remain distinct
from measurements of Stark quantizations. No scores, aliases, native memory
plans, inference paths or model qualifications were added. Other task rankings
and explicit Lighter model / More model capacity preferences are unchanged.
MaxxBench remains deferred.

## Validation

| Check | Result |
| --- | --- |
| Benchmark and guided model tests | 107 passed |
| Complete converter suite at the implementation source | 6,741 passed; 96 skipped; 262.89 seconds |
| Converter lint and format | Passed; 181 files formatted |
| Repository policy and diff whitespace checks | Passed |
| Independent read-only policy and patch reviews | No correctness gap found |
| Separate locked generation, optimized scoring and template release builds | Passed with Rust 1.98.1 |
| Self-contained app and packaged runtime checks | Passed |
| Installed-bundle full catalog ranking replay | Passed; 228 entries; Qwen Flash exports 91.9, then Qwen 27B 90.3 |
| Deep strict signature verification before/after install | Passed |

Regressions cover an unscored first-place candidate, a lone covered parent,
zero vs missing, aliases not inflating cohort coverage, evidence across soft
memory tiers, hard GPU/physical/storage exclusion, deterministic ties, separate
cohorts, all-unknown fallback wording, explicit resource preferences and unchanged
non-coding behavior. The actual native profiles are replayed with a 128 GiB
host, its Metal budget and installed Qwen copies; reducing GPU budget excludes
those Qwen candidates despite their scores.

## Actual resource and catalog replay

The production desktop pipeline checked the existing library on this Mac:
128 GiB physical memory, Apple M5 Max, unified memory, Metal recommended budget
115,448,725,504 bytes. The bounded catalog returned 228 entries. Flint still
passed admission; its missing coding result no longer preserved first place.
The actual top three became:

| Choice | Published coding reference | Existing copy |
| --- | --- | --- |
| Qwen3.8-Flash-Next MLX-Serve mixed | 91.9, Qwen developer LiveCodeBench v6 table | Yes |
| Qwen3.8-Flash-Next Lily Q4-XL | 91.9, same parent and same table | Yes |
| Qwen3.8-27B oQ4e MTP | 90.3, same table | No |

All three report `used_for_ranking=true` and cohort
`qwen38-card-livecodebench-v6`. The two Flash exports share one parent result;
their tie order does not establish a quality difference. The native-only
packaged replay likewise placed both Qwen exports before Flint. The full
catalog includes the additional scored 27B export, moving Flint beyond the
three displayed choices. No weights were downloaded and no model or task
benchmark was run.

## Installed bundle

The signed development build was created from a clean source tree using the
current builder, then installed through its staged signature verification and
atomic replacement helper at `~/Applications/Stark.app`.

- Source revision: `f3da1edf29ed90af539b796a481e98c78025dea1`
- Development build: `20261108205314508578`
- Signed executable SHA-256: `0d91b805b77580cdcbcaf1225896027efc3f1b271a33fd9e9020a6679643193e`
- Manifest SHA-256: `307d21d2753f2c746244f9f7770f377edafb259fe7a705caee1ec10013360328`

The installed app displayed the matching development build and opened Guided
Setup. Coding and Balanced were selected and discovery requested. The Mac
locked during final visual verification; Computer Use reported that manual
unlock was required. The installed bundle's complete desktop-pipeline replay
independently confirmed the three scored Qwen choices in the table above with
`used_for_ranking=true`.

After the user manually unlocked the Mac, visual verification completed in the
installed Guided Setup window. Coding / Balanced displayed the already-installed
Qwen3.8-Flash-Next MLX-Serve mixed export first, with no new download required.
Its explanation states that published coding evidence drives priority among
models that fit. Expanded model details showed LiveCodeBench v6 (Qwen developer
table), 91.90/100 for the upstream Flash Next model, explicitly marked
"Used to prioritize coding recommendations." The details retain the exact
candidate revision and distinguish the upstream result from artifact evaluation.
The window was left open with those details expanded; no model was started.
