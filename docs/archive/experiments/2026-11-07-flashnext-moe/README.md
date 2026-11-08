# Flash-Next MoE projection source patches

Each patch applies independently to engine baseline
`72b5aaef96c939809bc906d10d45a0eef70386c0`. They retain the original upstream
copyright and licenses. The pinned Lily source revision and local provenance
are in [UPSTREAM.md](../../../../vendor/lily-core/UPSTREAM.md).

| Patch | Implementation | SHA-256 |
| --- | --- | --- |
| [Candidate 1](candidate1-fixed-projections.patch) | Fixed Q4 expert gate/up and Q8 shared-down dimensions; original dispatch order | `ae704e6611811868054bbcad01427af49723bf6fff9a7199e053abd2a101dee8` |
| [Candidate 2](candidate2-shared-first.patch) | Fixed Q4 expert gate/up; ordinary shared down queued first in the existing dependency level | `34640e336d7ad228fc0368d5790f1df726af7ccb0ba3651ef39d22394b601c19` |
| [Candidate 2, final controls](candidate2-final-controls.patch) | Same production sources as candidate 2; test-only addition of repeated expert routes | `a897e69e365721bae1bc56e9975b53d3bfb29d968ca125439b5cac215f27c723` |

Candidate 2 with the final controls is retained on the experimental branch.
The original two patches bind the measured component and worker receipts.
The final patch changes only the ignored numerical test, expanding its
12 permutation cases to 16 cases with explicit repeated routes. The release
generator rebuild retains the same executable hash. All patches include
their source-bound synthetic controls. Candidate 1's
separate shared-down specialization is retained only as an experiment.
Apply one patch only.

The [report](../../../benchmarks/2026-11-07-flashnext-moe-projections.md) and
[aggregate receipts](../../../benchmarks/data/flashnext-moe-projections-2026-11-07.json)
record the executable identities, numerical checks, component measurements
and incomplete/ineligible matched attempts. Whole-model performance remains
unqualified. A patch applying cleanly is not performance qualification.
