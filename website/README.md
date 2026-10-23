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
- The hero and install section link directly to the latest signed, notarized DMG through the stable Stark.dmg release asset. The install section provides copyable latest-stable release commands for the menu bar app and terminal CLI, the Apple silicon/macOS 26.4 minimum, PATH setup, a model-free `doctor` check and the app launch command. Model installation and chat steps cover both interfaces.
- Chat, generation, scoring and conversion tabs provide installed CLI command references. They do not execute a model in the browser. Tabs support arrow keys, Home and End; both command panels support copying with a selection fallback.
- The benchmark defaults to tokens/second and offers an overall view plus six workload selections across six configurations from five engines. Relative mode normalizes MLX-LM to 1.000× from the unrounded recorded rates; rounded display values are not used to reconstruct the speedups.
- A semantic results table remains usable without JavaScript. Bar lengths compare engines within each workload. Narrow screens scroll the table within its own region.
- Expandable methodology notes describe the four independent loads, repeated workloads, complete request timing, model and engine versions, quantizer differences and quality checks. The main page prioritizes benchmarks and signed installer setup. Evidence links point to immutable Git snapshots.
- Color, focus indicators, image descriptions, reduced-motion handling and keyboard controls are implemented directly in the source.

## Hosting

The Site identity is stored in `.openai/hosting.json`, with `dist` as its static directory. The project owner has requested public access without ChatGPT sign-in. The primary public URL is [starkengine.app](https://starkengine.app/), also used as the page's canonical URL. The generated [Sites URL](https://stark-inference.arcanepine32.chatgpt.site/) remains available. [starkengine.io](https://starkengine.io/) is active and serves the same site and benchmark data as `starkengine.app`; the repository's custom-domain receipt records the final verification.

The source is mirrored into a separate ignored publication checkout. The Sites workflow opens that checkout, synchronizes only website source, builds it, commits and pushes its exact source state, then packages `dist/` with the hosting manifest for deployment. It does not publish the Stark repository, benchmark runtime data, models or local secrets. Credentials remain in session memory and are supplied only through the workflow's stdin.

No inherited Stark cloud, signing, benchmark or release workflow is enabled. See the [presentation and launch notes](https://github.com/jshahbazi/stark/blob/feee52b4c7b07e29b4849684cd3dfba8690472d4/docs/PRESENTATION.md).
