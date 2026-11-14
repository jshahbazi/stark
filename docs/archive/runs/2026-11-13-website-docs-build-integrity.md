# Documentation build integrity — 13 November 2026

The reported `/docs/first-request/` route and other nested documentation pages returned HTTP 404. Sites version 68, source `119c20672feed53057e1752423d229bc75c54382`, published only 20 files. Version 67 contained 75. The source retained the Markdown manual, but the legacy `node build.mjs` entry point removed `dist/` and copied static assets without running Astro.

The build entry point now checks for installed dependencies before modifying output, copies the static pages, runs Astro, propagates failure and verifies generated output for each Markdown page. `npm run build` delegates to that same command. Both entry points therefore build documentation. Existing favicon changes from the subsequent publication remain intact.

Sites version 69 restored the complete website with 76 packaged files, from source `98859da1caa3b3a12505ea369698b3fe8a5fdddf`. Deployment `appgdep_6af71170ea4481919bcf6226d4eb2096` succeeded.

Validation:

- The formerly static-only `node website/build.mjs` now builds all 14 documentation pages in the Stark checkout. Repository policy and generated link/fragment/asset checks pass (640 links in the Stark checkout, 639 in the publication checkout).
- Live GET checks of 58 routes/files at `starkengine.app` passed, including all 14 clean documentation routes, home, technical page, legacy docs page, search index, JavaScript, CSS, image and sitemap assets. HTML titles matched the build; non-HTML content hashes matched the local output.
- The reported first-request route was also opened and verified in the browser.
- External-link audit checked 19 documentation URLs and 23 homepage/technical-page URLs; every check returned HTTP 200. These totals count the separate audit sets and may overlap.

This verification concerns website content and delivery, not model execution or API behavior.
