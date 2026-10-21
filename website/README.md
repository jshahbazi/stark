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
- Benchmark tabs separate the Oct-20 updated Stark/MLX-LM comparison from the Oct-19 stock multi-engine pilot and JevK5 BF16 scoring. The multi-engine workload selector covers all six fixtures across MLX-LM, vLLM Metal, llama.cpp and both TensorFold modes. Older Stark rates stay bound to that earlier pilot; MetalRT remains unmeasured. Tabs support arrow keys, Home and End.
- The stock chart scales its decorative bars to the highest recorded rate within the selected workload. Its numbers are medians from the original report; no new cross-campaign speedup is calculated. JevK5 latency remains separate from generation throughput and displays the low public accuracy alongside the timing result.
- A semantic results table remains usable without JavaScript. Bar lengths compare engines within each workload. Narrow screens scroll the table within its own region.
- Without JavaScript, all three comparison sections remain visible and the stock table shows the Fibonacci fixture with its selector disabled; the linked report contains every workload.
- Expandable methodology notes retain the accepted-pair, exclusion and quality details. The main page prioritizes benchmarks and source-build setup. Evidence links point to immutable Git snapshots.
- Color, focus indicators, image descriptions, reduced-motion handling and keyboard controls are implemented directly in the source.

## Hosting

The Site identity is stored in `.openai/hosting.json`, with `dist` as its static directory. The project owner has requested public access without ChatGPT sign-in. The primary public URL is [starkengine.app](https://starkengine.app/), also used as the page's canonical URL. The generated [Sites URL](https://stark-inference.arcanepine32.chatgpt.site/) remains available. The requested `starkengine.io` alias is configured with the same hosting targets; its new registry delegation must finish propagating before it can serve traffic. The repository's custom-domain receipt records the observed status separately from configuration.

The source is mirrored into a separate ignored publication checkout. The Sites workflow opens that checkout, synchronizes only website source, builds it, commits and pushes its exact source state, then packages `dist/` with the hosting manifest for deployment. It does not publish the Stark repository, benchmark runtime data, models or local secrets. Credentials remain in session memory and are supplied only through the workflow's stdin.

No inherited Stark cloud, signing, benchmark or release workflow is enabled. See the [presentation and launch notes](https://github.com/jshahbazi/stark/blob/feee52b4c7b07e29b4849684cd3dfba8690472d4/docs/PRESENTATION.md).
