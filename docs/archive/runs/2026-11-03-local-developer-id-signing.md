# Local Developer ID signing validation — 2026-11-03

Implementation: `d001b0890af9f9da789686a82bb5ebae197122ec`.

## Problem and change

The local menu-bar builder always used an ad-hoc signature. Its designated
requirement depended on its code hash, so changing the executable could require
renewed approval to read the existing development Keychain credential.

Local builds now require the unique available Developer ID Application identity
by default. `--sign-identity` and `--keychain` select an identity explicitly.
Missing or ambiguous certificates and signing failures stop the build before
publication or installation. `--ad-hoc` explicitly selects the previous behavior;
unsigned distribution staging passes that flag before the separate release
signing step.

The builder reuses the distribution signer's native-image inventory and
signature verification. It signs each embedded native image, records delivered
hashes separately from pre-sign build evidence, signs the canonical app and
verifies the complete seal before replacement. Local builds retain their
development version and channel.

## Checks

| Check | Result |
| --- | --- |
| Builder and distribution suites | 67 tests passed |
| Signing, Apple credential setup and DMG suites | 45 tests passed |
| Isolated API credential suite | 2 tests passed |
| Repository policy, Ruff checks and formatting, diff whitespace | Passed |
| Fresh default build and installation | Passed |
| Delivered host signature and complete app seal | Passed |
| Embedded native runtime signatures | 37 images verified |
| Signed overlay hashes | 428 files verified |
| Prior and new signed app designated requirements | Identical, with different executable hashes |
| Rebuilt signed executable reuses a disposable Keychain credential | Passed with interaction disabled |

The native credential check compiled two different binaries from the production
credential store and a small private harness. Both used Developer ID team
`S5BT2PRN5B`, the same validation signing identifier, hardened runtime and secure
timestamps. The first created a synthetic item in a unique test service; the
second reused it with credential creation disabled. Both legacy Keychain user
interaction and the store's authentication-context interaction were disabled.
The test removed the disposable item. It did not access the production API item.
Harness SHA-256:
`f876aefd0dad56815ed174a8cf51b8376948635f9540bb47eef06cbf763a9ce4`.

## Installed artifact

- Destination: `~/Applications/Stark.app`; reopened from that exact location.
- Signing team: `S5BT2PRN5B`.
- Development build: `20261103181949858884`.
- Executable SHA-256: `12ae40f8b1d1dbb8f2f443a6460da50c89b940d792402a298016b3bd41438b05`.
- Embedded menu-bar manifest SHA-256: `aa46703d1daa886077138ec094830c6d2c7686fc1ca354218186d2d2f46a515b`.

The source manifest identifies the implementation commit and reports
`dirty: true` because an existing untracked research document was retained.
Every recorded source-file hash matched the checkout after the implementation
commit. Original build hashes and signed-runtime hashes remain separate.

This run covers signing continuity and disposable credential reuse. Switching
from an ad-hoc build selects the existing installed credential namespace and may
require one approval or reconnecting copied-key clients. A locked Keychain or an
intentional signing-team change can still require user action. The local app
was not notarized, and no model execution or performance qualification was run.
