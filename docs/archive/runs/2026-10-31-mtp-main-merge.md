# MTP integration with current main, 2026-10-31

Validated merge: `a83d01be6325888b3ebaf2829442233cacf9a50b`.
Parents: MTP `5b05123658d21a1dfb66e007e4d4f024df144cfe` and main
`a759fad34dfa6cb1687969ce5fa533faf3e84ab6` (desktop backend PR #58).
Exact tested tree: `25bede0dc72f6087c0c7583a9cf652031d55a71d`.
Private check receipts identify the pre-merge HEAD and this staged tree;
the committed merge has the identical tree.

Eight files had textual conflicts. Resolution retains main's persistent
desktop connections, function tools and model-specific context limits with
MTP capability discovery, per-model preferences and request overrides.
Twelve new Python cases cover MTP with tools and unknown context capacity.
Documentation describes both features and the growing initial scratch reserve.

The host MTP draft and both verifier routes now pass the active state capacity
to the updated prefill scratch allocator. The MTP iterator reserves its admitted
continuation before borrowing scratch. Committed recurrent, convolution,
n-gram, hidden and cache state survives growth. Target arithmetic, quantization,
sampling probabilities and RNG logic are unchanged. Independent review found
no additional integration defects.

## Validation

- Full converter suite: **5,212 passed, 88 skipped**.
- Generator release suite: **88 passed, two ignored**. Model-free generator:
  **52 passed**. Release build and all-target Clippy with warnings denied passed.
- Physical-host Lily default suite: **77 passed, 52 ignored**.
- Full converter Ruff lint/format, Rust formatting, repository policy and
  whitespace checks passed. Dependency locks are unchanged.
- Official mixed-4/8-bit checkpoint: **13 native generation requests passed**.
  Boolean/count selectors, repeated on/off/on and invalid-selector recovery
  retain all pre-merge per-route sampled-output hashes. A 262,144-token request
  budget retains the short workload's respective on/off hashes; an oversized
  budget is rejected and the following request succeeds. This checks admission
  with a small workload, not full-length numerical accuracy or memory use.

The first full Python run inherited the private runner's `umask 077`: its
public-directory fixture became private, producing one failure and 5,211 passes.
The unchanged test and production source match main. All three related cases
pass with `umask 022`; only the public case fails with `077`. The successful full
run sets the child process mask to `022`, retaining private parent receipts.
No product or test-fixture change was made for this harness error. An initial
vendor command named an excluded package from the root workspace; the corrected
manifest-path invocation produced the passing vendor result above.

The native correctness check ran with one existing installed-app model worker
left running. The validator rejected additional competing workers, closed its
own worker on exit, and retained only aggregate results and hashes. It is not a
throughput benchmark. Existing source-bound benchmarks and frozen workers are
preserved; no new speed result is claimed.

| Artifact | SHA-256 |
| --- | --- |
| Rebuilt generation worker | `d7462886917e2548abd602c5344c5a157b80205ae31b3c4d167ca8ae4046129d` |
| Native smoke validator | `9f6868c3b9f30f1a24ed7f4f88ba1e9d1069a71285f87dec69afcc24c99806ea` |
| Corrected aggregate test runner | `43fd13f0bcc3932ff998a502eac4aaeda33530cd4a308090364fe379b7f54036` |

The preceding [PR review](2026-10-31-mtp-pr-review.md) and
[model-free build follow-up](2026-10-31-mtp-model-free-build-fix.md) retain their
original source identities and scope.
