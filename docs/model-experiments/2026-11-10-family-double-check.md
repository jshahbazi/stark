# Model-family double-check before pull request

Attempt ID: `20261110-FAMILY-RECHECK-01`.
Baseline branch head: `cbd8b0224440ec7cd3bb2084c37ed0c5f26141fb`.
Status: local review and revised controls completed; production artifact qualification pending.
Registered before running revised native controls. No installation or merge.

Three independent reviews cover native execution/sampling/state, image handling
and memory admission, and converter/metadata compatibility. The converter review
found no concrete regression; its targeted tests pass 42 cases. Initial source
review found the following gaps before a PR was opened:

- The snapshot worker regression did not isolate several invalidation keys:
  context was tested after clearing the entry, and later divergence/scope misses
  also changed other inputs. Each revised gate must prime and publish a matching
  baseline, prove a 1,024-row plan, then change one condition. Cached/fresh output
  comparisons remain exact and content-free. Test both synthetic full-attention
  and sliding-ring targets, greedy and seeds 0/7/42.
- A finite reciprocal does not ensure finite temperature-scaled logits.
  Temperature 1e-38 with finite values can still overflow. The candidate must
  avoid silently choosing a lower logit and avoid a request-triggered model
  reload, while preserving existing arithmetic and draws in ordinary ranges.
  The precise repair and numerical controls will be recorded before adoption.
- The original tower estimate summed a loose block bound across every layer,
  despite scratch-pool reuse within a command. A square Gemma 4 E2B image was
  charged about 5.3 GiB of tower workspace. The revised inventory must cover
  allocator-rounded live scratch classes, persistent constants, patches and
  readbacks without multiplying reusable storage by depth. Compare the bound
  with observed production-path pool allocation on a synthetic tower; no speed
  or full production image-artifact qualification follows from that control.

Existing source-bound validation is preserved in
`../archive/runs/2026-11-10-model-family-triage.md`. Revised receipts will bind
new source/executable identities separately. Prompts, responses, token vectors
and probability arrays stay private and ignored; publish aggregates/hashes.
The full image acceptance matrix and actual Gemma 2 reference replay remain
separate qualification gaps and are not closed by this review.

## Attempt history before the final candidate

The first integrated compile failed in the new allocation controls: private
allocator access, ambiguous backend inference and a missing trait import.
The control was changed to use the normal Metal command allocator and its
existing reservation probe, preserving production module visibility.

The first centered-temperature candidate failed the combined CPU/Metal control
with an invalid-token assertion (zero passed, one failed), and the actual dense
worker survival control also failed (zero passed, one failed). The original
kernel assertion did not identify its backend/dtype; that failed run alone does
not localize the failure. Revised assertions include content-free labels.
The next candidate reconstructs a positive subnormal temperature reciprocal
from its stored mantissa using normal-range arithmetic, then applies the same
centered retry. Exponent-nonzero reciprocals retain the existing operation.
The physical rerun is required before adoption. The first dirty source patch was
not sealed, so its failure receipt binds the executables and named mechanism,
not an immutable source commit; final candidate source will be committed.

The three allocation controls already pass on the first executable. Actual
constant/pool reservation remains within its inventory and pool reservation is
unchanged between one and three synthetic production-tower layers. Both revised
cache fixtures also pass the isolated-key controls. Exact allocation results
are in [the workspace record](../archive/runs/2026-11-10-vision-workspace-inventory.md).


## Final candidate and observed results

Runtime source: `77d6c50e`. The additional boundary-test commit is
`d92999f1dc73b2bbc93eed79020804554d20849a`; it does not change runtime code.
Host: Mac17,7, 128 GiB RAM, macOS 26.7.2 (25G332).
Rust 1.98.1 builds the release worker; nightly-2026-10-01 builds engine tests,
with dev/test debug information disabled and incremental compilation disabled.
The worktree owns its target directory; worker and optimized scorer builds are
separate. Python uses locked uv 3.13.13.

The corrected centered/subnormal control passes on both CPU and physical Metal,
F32/BF16, over the registered filters, masks, equal maxima, seeds and a 2,061-row
vocabulary. The added boundary control rejects reciprocal-overflow bits
`0x00200000`, accepts `0x00200001`, and handles the minimum normal F32
temperature. Normal-temperature and omitted-temperature controls preserve draws.
Only the overflow arm changes sampling arithmetic; weight, activation, KV and
recurrent precision policies are unchanged. Sampler state is reseeded per
request and does not alter the retained prefill boundary or its reuse key.

| Final check | Observed result |
| --- | --- |
| CPU/Metal native regression set | 70 passed, six image-artifact controls ignored |
| Separate CPU/grammar sampling and cleanup set | Four sampling controls and one cleanup control passed |
| Release generator suite | 144 passed, five ignored |
| Revised isolated snapshot test | Passed on full-attention and sliding-ring synthetic artifacts; every invalidation proves an eligible baseline hit before changing one condition and compares output with fresh execution |
| Actual dense worker temperature control | Accepted 1e-38 and the following greedy request both succeed in the same loaded Qwen3-4B Q4 worker |
| External CLI protocol recovery smoke | 1e-40 is recoverably rejected; 1e-38 emits four tokens; the next greedy request emits four tokens with the unchanged baseline output hash |
| Both release scorer configurations | 34 tests passed each |
| Generation process boundary | Six cases passed |
| Release Clippy, all generator targets, warnings denied | Passed |
| Formatting, repository policy and diff checks | Passed |
| Independent converter slice | 42 passed; unchanged Python source retains the original full-suite 6,936 passes / 94 skips |

The early revised-cache execution overlapped another functional control process.
It was repeated in isolation after that process completed; the final cache
receipt uses only the isolated run. No timing or performance inference is made
from either functional run.

The actual worker smoke uses the unchanged source-bound artifact documented in
the [original record](../archive/runs/2026-11-10-model-family-triage.md), context
512, output budget four, seed 42 and MTP off. Its subsequent greedy output hash
remains `1735f922da391a9e850e42fa617a2b77406211dbe9d06b348dd12a1ad817529d`.
The actual model is used for bounded process recovery only; no broad quality,
image-loaded text/MTP or Gemma 2 reference claim follows.

The revised image workspace bound is allocator-rounded stage inventory, including
the separate constant allocator and explicit margins. Square Gemma 4 E2B charges
412,221,440 bytes (393.125 MiB), rather than the prior roughly 5.3 GiB. The
synthetic production-path control observes identical 125,952-byte pool reserves
at one and three blocks, within its 475,136-byte inventory. This proves that
bounded allocation path and source inventory; each production image artifact
still needs its own default-image acceptance matrix. See the
[workspace record](../archive/runs/2026-11-10-vision-workspace-inventory.md).

| Sealed artifact | SHA-256 |
| --- | --- |
| Release generator | `856dd91287fc10a2c557751651ceea58a897893c8db9fb40786f2fdee8937cb3` |
| Generator test executable | `7247c6d25e5b4f0146190e5c3f50e2cb9d4518bd3469d454bab82c6d400d5082` |
| CPU/Metal engine test executable | `112b8f84dbeb1dcf52805e6197d60b08b350fa2e48e38c5bd2d845423a03e84b` |
| Final native result receipt | `de778694f90b428e3a59bd279ccbf71b1bd5e2797532de9e882cdb7bde64d8d3` |
| Isolated snapshot result receipt | `7cde1d212d464ed7531844ae528d1bd0996a87cf045200cff32d0650faa1b94a` |
| Actual worker recovery receipt | `e84182ffd7d1fdc06443d0e6a14ddca444f2848e5a037a5d82428277bcdea3b2` |

The first failed centered candidate is preserved separately: its engine test
executable hash is `2e9a84e18e85739ab6f6eb687aaa20c8e56e9632e3b6698175e84bd23e26ab23`
and worker-test hash is
`598d22825aea3de38bffaf8e4846853520b3c13a938198dd5c6cbc9bf76b51cc`.
These are failure receipts, not accepted numerical evidence.

## Disposition and limits

The corrections are adopted on the branch with the tested scope above, ready
for pull-request review. The complete native Metal suite, complete CPU/grammar
suite, actual Gemma 2 reference replay, production image qualification, quality,
peak RSS and speed remain open. This work does not merge, release, install or
change user models. The original source-bound records remain intact.
