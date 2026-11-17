# Decider reference environment security update

Date: 2026-11-16. Attempt: `20261116-DF13SEC-01`. Dependency fix:
`f7f3c4bb664eef4e475d1d349bee3960019c8932` on `codex/decider-flash-v13`.
See the [attempt ledger](../../model-experiments/2026-11-16-decider-flash-v13.md#reference-dependency-security-update-20261116-df13sec-01).

The isolated reference project now locks Pillow 13.0.0 instead of 12.3.0.
The updated environment passed the existing formatter, full numerical and
lifecycle checks. Earlier Pillow 12.3.0 qualification records remain unchanged;
this is a separate result, with a new reference-lock identity.

## Security boundary and change

PR #169's original Security run
[43962410817](https://github.com/jshahbazi/stark/actions/runs/43962410817)
rejected the newly added reference lock. Both OSV and GitHub Dependency Review
reported Pillow advisories. A direct OSV version query reproduced 62 advisory
records for Pillow 12.3.0 and returned none for 13.0.0. Those records span
multiple codecs and advisory feeds; this does not mean 62 independently
demonstrated exploits in the reference runner.

The reference's caller-supplied file/base64/PIL image inputs reach
`pplx_decider.protocol.open_image`, then `PIL.Image.open`/RGB conversion and the
official processor. The native Rust/Metal image path does not load Pillow.
Updating the shared reference dependency covers every caller and platform in
that environment without relying on file extensions or filtering advisories.
Pillow's [13.0.0 release notes](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html)
identify the patched release, including JPEG decoder and other image-processing
fixes. Exploit reachability of each codec-specific advisory was not separately
demonstrated.

`uv lock --upgrade-package pillow` changed only Pillow's resolved version and
artifacts. All other package versions remained unchanged. `uv lock --check`
and `uv sync --locked` passed; the environment reports Pillow 13.0.0. The
official snapshot/source, model artifact, native worker and validation Python
code were unchanged. The README documents the deliberate security override of
upstream's 12.3.0 pin. No scanner, severity policy, allowlist or CI gate changed.

## Environment and identities

Host: Apple M5 Max, 128 GiB, macOS 26.7.2 (`25G332`). Reference: Python 3.12.13,
MPS FP32, separate questions, `fast=False`, `locate=False`, eight CPU threads,
4,096 positions. Native: existing Rust 1.98.1 release/default-feature BF16
worker, unchanged precision and fresh state per forward. Reference and native
models ran sequentially. Incidental latencies are not a performance result;
other host activity was not screened for timing.

| Binding | SHA-256 or revision |
| --- | --- |
| Official checkpoint revision | `d06fb3f2d97e0a52325a0d463c450cf0338f44fe` |
| Reference source aggregate | `6a1555c5998ffcaba54a299599b45591002a7941b5d383bcef5bf35bf4d49481` |
| New reference lock | `13d1743023d04ab6997161b300699d406561cec50b03759f5cb20a834019dfd5` |
| Reference runner | `2e94e02c84feea14186e57ee70a734f35f7f3072c63d4191795eb1f85c03f54c` |
| Qualification script | `5f0d8298786b76d6ca34b08fc40a6e8bd8fd702244adead04043d6222e3fd074` |
| Native worker | `ca58a20531797dfca877aef26f2986798a24a75539ac7a9a95643af34ca23579` |
| Native configuration | `d859a249a3089252f164c755d2a584e41c1041ab91dc32751a742ddad29e8337` |
| Native weights | `c3ab2961a63c3342ff3390b41036c6029d62095414db4cffbd59a8944c754c97` |
| Private fixture | `3c302223fcd3491aaa47389dc91743a11627f0b8f61b555ffce6877e59969d4c` |
| New aggregate receipt | `1c186f3aee118463cf21a5fb35b46573778efde0ff19bfdf27d0ba9afde1f681` |

The aggregate receipt is retained privately through the security artifact
store. Inputs, images, token vectors and response/probability traffic were kept
private; raw responses were not saved. No historical model/source file or
receipt was modified.

## Results and integration checks

- Formatter differential: all 6,460 tokens across 15 sequences matched exactly,
  including five image sequences; no marker, segment or option-count mismatch.
- Full reference/native corpus: 12 requests, 15/15 top choices, including five
  image decisions. Maximum absolute probability error: 0.00595570393684; mean:
  0.0000553653162528. Both meet the unchanged fixed gates.
- Token usage, repeat, reset, validation-error recovery, invalid-image worker
  closure, cancellation while pending and restart checks passed.
- Focused protocol/qualification tests: 38 passed. OSV gate tests: 17 passed.
  Repository policy and diff checks passed. Lock consistency and actual installed
  Pillow version were checked independently of the advisory count.

The dependency push exposed conflicts with the newer API-worker changes on
main, preventing GitHub from starting PR checks. The feature branch was updated
from main without merging PR #169. Resolutions preserve automatic Chat/API
worker selection for generic language models and reject native Decider Chat/API
requests without unloading its Decision worker. Focused checks passed: 54
native/API/input-budget tests, four automatic-worker tests, 11 Swift Decider
checks, five API/controller checks, a connection regression including native
Decider exclusion, and full Swift typecheck. Final CI security results are
recorded on [PR #169](https://github.com/jshahbazi/stark/pull/169).

This qualifies the named artifact and bounded corpus under the updated
reference dependency. It does not establish arbitrary-image equivalence,
per-task calibration, speed, another artifact or an installed application
update. No app was installed and no PR was merged for this security fix.
