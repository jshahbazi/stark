# Hardware-aware Beginner recommendations

Date: 2026-11-05. Branch: `codex/hardware-model-recommendations`.
Source: the commit containing this record, based on
`d57ece4441fb9d6f191f2bfba264a47ccd44dde1`.

## Behavior covered

The native app supplies its Metal device name, unified-memory support,
recommended working-set budget and processor count. The backend independently
reads RAM and library/download-cache storage. GPU information can only narrow
the backend resource checks. The recommendation never adds GPU memory to
physical memory on Apple silicon.

The guide scans bounded catalog candidate windows asynchronously. Exact source
revisions, complete dense memory geometry, source-bound chat/tool metadata and
the guided 32,768-token context are required. Unknown recurrent/expert layouts
remain in Advanced until they have their own planners. The two recorded Qwen3
starters remain available as fallback candidates under the new hardware checks.

The policy reserves physical memory and Metal headroom, then ranks by
comfortable fit, explicit task metadata and the selected priority. Balanced,
lighter-model and capacity priorities have different deterministic objectives.
Parameter capacity and weight traffic are heuristics; they do not establish
answer quality, latency or task competence. Installed status is a tie-breaker.
Artifact identity, resources and native readiness are rechecked at install/start.
Generation state and prefix-reuse behavior are unchanged.

## Native and live metadata checks

Host: macOS 26.7.1, Apple M5 Max, 128 GiB physical memory. Native Metal reported
a 107.52 GiB recommended working-set budget. These are observations from this
run, not eligibility claims about other machines.

The live metadata scan returned 180 variants across 14 family windows with no
failed families. The windows were incomplete by design. Seven catalog variants
had complete plans and recognized tool metadata. After including and
deduplicating the recorded starters, eight candidates were considered. The
observed first choices were Qwen3 4B 4-bit for Balanced, Qwen3 0.6B BF16 for
Lighter model, and Qwen3-14B BF16 for More model capacity. No candidate weights
were downloaded and no model was run during these metadata checks.

The actual Python recommendation report was parsed by the production Swift
`BeginnerModelReport`: three choices, GPU summary and selection reasons passed.
The complete native app sources compiled with optimization for arm64 macOS
26.4. The executable was built in ignored runtime storage; it was not installed
or launched.

Private local artifact SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| Native app executable | `a4a8750a172cac3fdc609101e8c2ca401e1c3f76f4b72ea37c9ef83f70747ddd` |
| Live balanced recommendation report | `091f9dd3d5c11d119f8810686e64f091f003914ba8929ace70826e8462f517bd` |

## Automated validation

- Native Beginner state/view checks: 9 passed.
- Native controller checks: 128 passed.
- App builder tests: 46 passed.
- Focused catalog, recommendation, discovery and stop/stream regression checks:
  201 passed.
- Full offline converter suite: 6,094 passed, 89 skipped in 230.26 seconds.
- Ruff lint and format, locked dependency resolution, repository policy and
  whitespace checks passed.

An earlier full-suite run exposed missing discovery state on a deliberately
minimal controller fixture. Stop now tolerates that fixture while retaining
cancellation fencing. Candidate fixtures were updated alongside the completed
chat/tool and query-head metadata contract before the final run.

## Limits

This validates recommendation plumbing and deterministic policy behavior.
It does not qualify newly discovered artifacts through conversion, native
loading, long-context generation, performance or OpenWork task execution.
Discovery covers a bounded candidate window in each family; it is not an
exhaustive search of every repository or variant. Memory reserves remain
planning estimates and need measurements across more Mac configurations.
