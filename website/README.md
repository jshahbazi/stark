# Stark website

A static landing page: HTML, CSS and a small browser script, with no package installation, server-side code, analytics, external fonts or model execution. The logo is the owner-supplied image recorded in [brand provenance](../docs/brand/README.md).

## Build and preview

From the repository root, with Node installed:

```sh
node website/build.mjs
python3 -m http.server 4173 --bind 127.0.0.1 --directory website/dist
```

Open `http://127.0.0.1:4173`. `public/` is the source and `dist/` is ignored generated output. From a standalone site checkout, run `node build.mjs`.

## Content

The page introduces the app and terminal tools, offers the DMG and the installer script as the two install routes, links model compatibility to the converter guide, describes the local API briefly, and shows the Qwen3.5-2B benchmark with tokens-per-second and relative views across six workloads. Benchmark data comes from `public/benchmark-results.json`, which is a copy of the results in the linked report; relative values are computed from the unrounded rates. The results table works without JavaScript. Links to documentation point at `main`, except the benchmark report and evidence pages, which are pinned to the commits that produced them so the numbers on the page always match their source.

When the current benchmark changes, update `public/benchmark-results.json`, the pinned report links in `public/index.html`, and the [benchmarks index](../docs/benchmarks/README.md) together.

## Hosting

The site identity is in `.openai/hosting.json` (a deployment project identifier, not a secret), with `dist` as the static directory. The public URL is [starkengine.app](https://starkengine.app/), which is also the canonical URL; [starkengine.io](https://starkengine.io/) serves the same site. Publication mirrors only the website source into a separate ignored checkout, builds it, and uploads `dist/` with the hosting manifest; credentials are supplied through the workflow's stdin and never stored. The domain setup and the first deployment are recorded in [the deployment record](../docs/archive/runs/WEBSITE_DEPLOYMENT_2026_10_21.md).
