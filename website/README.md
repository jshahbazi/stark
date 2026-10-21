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
- Generation, scoring and conversion tabs provide command references. They do not execute a model in the browser. Tabs support arrow keys, Home and End; copying has a selection fallback.
- The benchmark defaults to tokens/second. Relative mode normalizes MLX-LM to 1.000× using the recorded paired percentage gain. Rounded displayed rates are not used to reconstruct those gains.
- A semantic results table remains usable without JavaScript. Bar lengths compare engines within each workload. Narrow screens scroll the table within its own region.
- Qualification notes preserve the four-of-six accepted pairs, whole-campaign rejection and quality limitations. Evidence links point to immutable Git snapshots.
- Color, focus indicators, image descriptions, reduced-motion handling and keyboard controls are implemented directly in the source.

## Hosting

The Site identity is stored in `.openai/hosting.json`, with `dist` as its static directory. A new Site starts owner-private. Changing its audience or selecting an official public homepage is a separate launch decision.

The source is mirrored into a separate ignored publication checkout. The Sites workflow opens that checkout, synchronizes only website source, builds it, commits and pushes its exact source state, then packages `dist/` with the hosting manifest for private deployment. It does not publish the Stark repository, benchmark runtime data, models or local secrets. Credentials remain in session memory and are supplied only through the workflow's stdin.

No inherited Stark cloud, signing, benchmark or release workflow is enabled. See the [presentation and launch notes](https://github.com/jshahbazi/stark/blob/feee52b4c7b07e29b4849684cd3dfba8690472d4/docs/PRESENTATION.md).
