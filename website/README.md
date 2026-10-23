# Stark website

A static landing page built around Stark's supplied cat logo, native interfaces and source-bound Qwen3.5-2B results. HTML, CSS and a small browser script; no package installation, server-side application, analytics, external fonts or model execution.

## Build and preview

From the Stark repository root, using the existing Node runtime:

```sh
node website/build.mjs
python3 -m http.server 4173 --bind 127.0.0.1 --directory website/dist
```

Open `http://127.0.0.1:4173`. Stop the server when finished. `public/` is the source; `dist/` is ignored generated output. From a standalone Site checkout, run `node build.mjs` instead.

## Content and interactions

- The original logo is retained byte-for-byte in `public/assets/stark-engine.jpg`; [brand provenance](https://github.com/jshahbazi/stark/blob/feee52b4c7b07e29b4849684cd3dfba8690472d4/docs/brand/README.md) records its source and hash.
- The footer introduces Stark, the owner's cat and the app's namesake, with the supplied photo in `public/assets/stark-cat.jpg`. The photo loads lazily below the main content.
- The hero identifies the menu bar app, terminal tools and experimental Uzu-derived Rust/Metal engine. The install section precedes the benchmark and labels two alternatives: the Mac app DMG, and the app plus terminal tools installer. Both use the latest signed, notarized release. The terminal path includes PATH setup, a model-free `doctor` check and the app launch command.
- Model compatibility links go to the converter's catalog and qualification guide. Copy distinguishes converter coverage from verification of exact checkpoints and formats.
- The optional local API subsection covers native/backend client requests on the same Mac, the Connection Details location, port/key lifecycle, defaults and unsupported features. It links to the scoped client-validation report for API behavior, settings and limitations. This optional subsection stays within Get started.
- The developer section covers the menu bar app's OpenAI-style text-chat API and JSON-lines worker integration over private stdin/stdout pipes. Direct worker integrations own their templates/tools; no SDK packages are published. Scoring copy describes candidate token scores and temperature scaling without claiming empirical calibration.
- Chat, generation, scoring and conversion tabs provide installed CLI command references. They do not execute a model in the browser. Tabs support arrow keys, Home and End; both command panels support copying with a selection fallback.
- The benchmark defaults to tokens/second and offers an overall view plus six workload selections across six configurations from five engines. Relative mode normalizes MLX-LM to 1.000× from the unrounded recorded rates; rounded display values are not used to reconstruct the speedups.
- A semantic results table remains usable without JavaScript. Bar lengths compare engines within each workload. Narrow screens scroll the table within its own region.
- Expandable methodology notes describe four independent loads, repeated workloads, complete request timing, worker-pipe versus local HTTP transports, quantizer differences and qualification limits. Exact engine revisions and answer-format checks remain in the linked immutable benchmark report. The page leads with installation and scopes the 12.2% result to the measured Qwen3.5-2B comparison.
- Color, focus indicators, image descriptions, reduced-motion handling and keyboard controls are implemented directly in the source.

## Hosting

The Site identity is stored in `.openai/hosting.json`, with `dist` as its static directory. The project owner has requested public access without ChatGPT sign-in. The primary public URL is [starkengine.app](https://starkengine.app/), also used as the page's canonical URL. The generated [Sites URL](https://stark-inference.arcanepine32.chatgpt.site/) remains available. [starkengine.io](https://starkengine.io/) is active and serves the same site and benchmark data as `starkengine.app`; the repository's custom-domain receipt records the final verification.

The source is mirrored into a separate ignored publication checkout. The Sites workflow opens that checkout, synchronizes only website source, builds it, commits and pushes its exact source state, then packages `dist/` with the hosting manifest for deployment. It does not publish the Stark repository, benchmark runtime data, models or local secrets. Credentials remain in session memory and are supplied only through the workflow's stdin.

No inherited Stark cloud, signing, benchmark or release workflow is enabled. See the [presentation and launch notes](https://github.com/jshahbazi/stark/blob/feee52b4c7b07e29b4849684cd3dfba8690472d4/docs/PRESENTATION.md).
