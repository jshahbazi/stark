# Website installer update — 2026-10-22

The public [Stark website](https://starkengine.app/#start) now documents the signed v0.1.1 installer published on 2026-10-21. The install section includes the Apple silicon/macOS 26.4 minimum, native arm64 terminal requirement, versioned download, PATH setup and model-free `stark doctor` check. The command tabs use the installed `stark generate`, `stark score` and `stark convert` launchers.

Terminal chat and the menu bar app require newer source builds. The page links those build instructions separately. Installation content was checked against the v0.1.1 release assets and launcher, with documentation links pinned to Stark revision `534acd3add687fab9958c88708ed101cd577c61f`.

## Publication receipt

| Field | Value |
| --- | --- |
| Site project | `appgprj_6ad8f59df1f4819185403a1dff22a0fb` |
| Published Site source | `6fcab1f4fc3c963c3b4f23c2fdf86fc6b53196d1` |
| Saved version | `appgprj_6ad8f59df1f4819185403a1dff22a0fb~appgver_90d27acc3f688191a31c9a6d60a66d58` (version 8) |
| Deployment | `appgdep_6ada6db0aa0c8191af11e998ffea42b2` |
| Result | `succeeded`, 2026-10-22 13:11:03 PDT |
| Archive content hash | `sha256:00dc574931e3d7d71d1b2ab437f8e8dffb821fc60e705ae7f2fe513bb7bc7d8a` |
| Public addresses | [starkengine.app](https://starkengine.app/), [generated Sites address](https://stark-inference.arcanepine32.chatgpt.site/) |

The existing public audience was retained. Publication used a separate ignored checkout containing website source only.

## Verification

- Static build, JavaScript syntax, whitespace and locked repository policy checks passed.
- Browser checks confirmed installation and worker command copying, Convert selection, and keyboard navigation to Score. The published installation copy button reported success.
- Anonymous HTTPS downloads confirmed the custom-domain HTML contains the updated source content; Cloudflare appends its browser check script. JavaScript, CSS and logo bytes match the local source exactly.
- The release installer download returned HTTP 200, and GitHub confirmed the pinned documentation revision exists.

This run verifies website content and publication. It does not rerun installation, native inference or benchmarks. Prior installation and benchmark evidence remains recorded separately.
