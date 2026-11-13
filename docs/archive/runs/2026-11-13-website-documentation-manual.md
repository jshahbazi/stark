# Website documentation manual — 13 November 2026

The publication source was opened from Sites project `appgprj_6ad8f59df1f4819185403a1dff22a0fb`, source commit `0d2ac07eac8f2a38b6fac047add2e70eda057735`. The Stark branch starts at `cae79c68`. Publication preserves the newer homepage and technical content in the Sites source rather than replacing it with the older Stark checkout.

Fourteen Markdown pages cover installation, model selection and management, Playground, three application connection routes, first API request, API and worker references, troubleshooting, compatibility, updates and privacy. Astro/Starlight generates navigation, page contents, highlighted examples and static Pagefind search. Dependencies are pinned and locked. The legacy docs page forwards named section links to the new guides and retains links without JavaScript.

Validation: `npm run build` passed in the publication checkout; `python3 scripts/check_docs.py` checked 624 generated documentation links, fragments and assets. The repository policy check passed in the Stark worktree. Browser QA covered the introduction at 1280px and the API guide at 390px, mobile navigation and search for Keychain returning relevant troubleshooting/privacy sections.

App installation, model loading, managed-client behavior and authenticated inference were not replayed during this documentation change. Product instructions were checked against the owning source references listed in `website/docs-site/README.md`. Development-flow labels and release/artifact capability differences remain explicit. No new numerical or performance qualification is claimed.


Publication completed as Sites version 66, source commit `8940409233a20dd7335275f3b1110e6bc675f32e`, deployment `appgdep_6af7051f210c819197e6c1fbbd87b9ce` (succeeded). The published introduction was verified at https://starkengine.app/docs/. A clean `npm ci --ignore-scripts` and full build in the Stark worktree also passed, followed by the same 624-link check. The dependency audit reported zero vulnerabilities.

## Pre-PR review

Merged current main and preserved its website changes. Corrected homepage and technical-page links to removed documentation anchors; the link checker now covers incoming docs links and fails when docs output is missing. Added a path-filtered documentation CI job with pinned actions, locked npm installation, static build and link check. Documented Node/npm requirements. Independent source review corrected image guidance to include both matching reference-verified execution evidence and locally tested evidence, using runtime capabilities for availability. Final Stark build and 640-link check passed; publication checkout check covered 639 links because its homepage differs. Repository policy and npm audit passed (zero reported vulnerabilities). Functional app/inference replay remains outside this documentation review.
