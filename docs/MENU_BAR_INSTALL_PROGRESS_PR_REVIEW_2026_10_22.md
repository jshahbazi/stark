# Install progress PR review — 2026-10-22

The independent review found one UI failure case: an install command rejected because the backend is unavailable changed app status to Error but left the download bar Running. Revision `703c1d62592ab0b8730b2d94281fd4d82dc7c3e0` marks unfinished stages Failed when command delivery fails. Local downloads remain skipped. A native test compiles the actual controller and covers remote/local failures, retry and counter reset. It fails against the previous controller and passes with the fix. The existing macOS CI job now runs these menu bar tests.

Converter and packaging reviews found no further concrete defects. The branch incorporates main revision `62b7c72230f4154a633514370cd74384abfd0272`; those upstream changes affected CI only.

## Repeated verification

- Locked converter suite: 3,855 passed, 88 skipped.
- Installer, CLI and menu bar checks: 44 passed, including all 15 menu bar tests and the native controller regression.
- Converter and packaging Ruff checks, formatting, repository policy and Swift typecheck: passed.
- Optimized, locally ad-hoc signed app build from clean revision `703c1d62`: passed.
- Packaged runtime harness: all 17 checks passed. The cached Hugging Face install reported six download events ending at 1,514,733,679 bytes and 13 conversion events ending at 1,192,276,800 bytes. Intermediate conversion progress, validated registry publication, native generation and positive input/output token rates were confirmed. Temporary data and child processes were removed; the user's registry was unchanged.

The review bundle and private aggregate-only results remain ignored under `artifacts/menu-bar/pr-review-build/`. Earlier source-bound reports and artifacts remain intact. The [original validation report](MENU_BAR_INSTALL_PROGRESS_VALIDATION_2026_10_22.md) describes the progress contract and scope limits, including cached-download coverage and the reused development worker. The review run did not replace the user's installed app.

| Review artifact | SHA-256 |
| --- | --- |
| Embedded app manifest | `a9c6f11c0b677f3abbcad53b75cb2adf7606d6640498763fcc5853eb06dc6f8f` |
| Signed app executable | `20cbb67194816e919db3a356307bed1b8699df684f566d811741621df3e30140` |
| Converter wheel | `2531efc00bc46d28be321e65a03fe16bf12308c2e1d35e399d8391a8f904b8f7` |
| Canonical converter toolchain provenance JSON | `607bd099ead2d2a2a072698122f33935d859db89e354b7ac4fd5d64057d85b1a` |
| Generation worker | `f12fd9849d81610170c2b893e98360b25ab553310cd5276e266f6d5e583e320f` |
