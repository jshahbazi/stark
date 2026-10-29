# Stark website

A static landing page: HTML, CSS and a small browser script, with no package installation, server-side code, analytics, external fonts or model execution. The logo is the owner-supplied image recorded in [brand provenance](../docs/brand/README.md).

## Build and preview

From the repository root, with Node installed:

```sh
node website/build.mjs
python3 -m http.server 4173 --bind 127.0.0.1 --directory website/dist
```

Open `http://127.0.0.1:4173`. `public/` is the source and `dist/` is ignored generated output. From a standalone site checkout, run `node build.mjs`.

Release copy is evergreen. Download and release-note links use GitHub's stable `releases/latest` URLs, so publishing a release requires no website rebuild. Update the site when installation steps, compatibility requirements or feature claims change. Benchmark, validation and provenance references stay pinned to the source revisions behind their evidence.

## Content and interactions

- The original logo is retained byte-for-byte in `public/assets/stark-engine.jpg`; [brand provenance](https://github.com/jshahbazi/stark/blob/feee52b4c7b07e29b4849684cd3dfba8690472d4/docs/brand/README.md) records its source and hash.
- The social preview card `public/assets/stark-social-card.jpg` (1200 by 630) is rendered from the original logo by `scripts/render_social_card.swift`, which places the unmodified logo in the hero's framed card beside the headline. The Open Graph and Twitter card tags reference it; the original logo remains the favicon and hero image.
- The footer introduces Stark, the owner's cat and the app's namesake, with the supplied photo in `public/assets/stark-cat.jpg`, resized to 480 by 360 for its 160 by 120 display box. The full-resolution original remains in Git history at commit `c21f1b74`. The photo loads lazily below the main content.
- The hero footnote and installation intro link to the latest release notes without displaying a release version or date.
- The hero identifies the menu bar app, terminal tools and experimental Uzu-derived Rust/Metal engine. The install section precedes the benchmark and labels two alternatives: the Mac app DMG, and the app plus terminal tools installer. Both use the latest signed, notarized release. The terminal path includes PATH setup, a model-free `doctor` check and the app launch command.
- The model directory features concise JevK5 Decision and Qwen3.8-Flash-Next text-generation cards. The inventory shares a compact area with Text, Images and Decision tabs. Text includes GLM-5.3-Flash, Qwen3.8-Flash-Next and Flint alongside the converter families; their setup details remain inside their entries. Decision lists JevK5. Each tab has its own panel, selected state and roving keyboard focus, with arrow keys, Home and End navigation. Image and other group fragment links select the matching panel. Without JavaScript, all panels remain readable. Text families expand to show the root README's named checkpoints and sizes; compatibility, validation and preparation details are linked.
- The Image models tab covers one PNG/JPEG through Playground, terminal chat and the Local API for admitted dense BF16 Qwen3.5 vision artifacts after local checks. Qwen3.8-Flash-Next is labelled for text input. Detailed family constraints and exact validation scope remain in the linked documentation.
- The optional local API subsection covers native/backend client requests on the same Mac, the Connection Details location, port/key lifecycle, defaults and unsupported features. It links to the scoped client-validation report for API behavior, settings and limitations. This optional subsection stays within Get started.
- The developer section covers the menu bar app's OpenAI-style text-chat API and JSON-lines worker integration over private stdin/stdout pipes. Direct worker integrations own their templates/tools; no SDK packages are published. Scoring copy describes candidate token scores and temperature scaling without claiming empirical calibration.
- Chat, generation, scoring and conversion tabs provide installed CLI command references. They do not execute a model in the browser. Tabs support arrow keys, Home and End; both command panels support copying with a selection fallback.
- The benchmark defaults to tokens/second and offers an overall view plus six workload selections across six configurations from five engines. Relative mode normalizes MLX-LM to 1.000× from the unrounded recorded rates; rounded display values are not used to reconstruct the speedups.
- A semantic results table remains usable without JavaScript. Bar lengths compare engines within each workload. Narrow screens scroll the table within its own region.
- Expandable methodology notes describe four independent loads, repeated workloads, complete request timing, worker-pipe versus local HTTP transports, quantizer differences and qualification limits. Exact engine revisions and answer-format checks remain in the linked immutable benchmark report. The page leads with installation and scopes the 12.2% result to the measured Qwen3.5-2B comparison.
- Color, focus indicators, image descriptions, reduced-motion handling and keyboard controls are implemented directly in the source. The page declares a light color scheme only; there is no dark theme.
- `robots.txt`, `sitemap.xml` and `404.html` ship beside the page. The site is one document, so the sitemap lists one URL. The 404 page uses absolute asset paths and is marked `noindex`; whether unknown paths return a 404 status depends on the hosting platform, which served the home page for unknown paths before this file existed.

When the current benchmark changes, update `public/benchmark-results.json`, the pinned report links in `public/index.html`, and the [benchmarks index](../docs/benchmarks/README.md) together.

## Hosting

The site identity is in `.openai/hosting.json` (a deployment project identifier, not a secret), with `dist` as the static directory. The public URL is [starkengine.app](https://starkengine.app/), which is also the canonical URL; [starkengine.io](https://starkengine.io/) serves the same site. Publication mirrors only the website source into a separate ignored checkout, builds it, and uploads `dist/` with the hosting manifest; credentials are supplied through the workflow's stdin and never stored. The domain setup and the first deployment are recorded in [the deployment record](../docs/archive/runs/WEBSITE_DEPLOYMENT_2026_10_21.md).
