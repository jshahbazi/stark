# Stark website

A static product website built with HTML, CSS and a small browser script.
The Markdown documentation uses Astro/Starlight. There are no analytics, external fonts or model execution.

## Build and preview

Use Node.js 22.23.4 or newer and npm 9.6.5 or newer. From the repository root:

```sh
npm --prefix website ci
npm --prefix website run build
uv run --project tools/converter --locked --python 3.13.13 python -m http.server 4173 --bind 127.0.0.1 --directory website/dist
```

`website/public/` is the source; `website/dist/` is ignored build output. A
standalone Site checkout also builds with `npm ci` and `npm run build`.

Check the build, local links, keyboard navigation and narrow-screen layout when
changing the pages. Run the repository policy check before committing.

## Content

- Ripple-65B Preview appears in the technical page's Text model list and the
  documentation page, with the pinned Q4_K_M and Q8_0 downloads and a link to the
  native support guide. Copy identifies the tested M5 Max 128 GB host and requires
  a source build containing Ripple support.

- The homepage introduces the app, shows its three product screenshots, gives a
  short setup path and explains the OpenWork/OpenCode connections and local
  processing. The new beginner flow offers General Work → OpenWork or Coding →
  OpenCode, followed by Quality (the default) or Speed.
- Technical details covers the local API, featured models, memory examples,
  context reuse and three benchmark studies. Its tabs and copy control work
  without external services; content remains readable without JavaScript.
- Documentation provides an on-site quick start and organized guides for app
  connections, model compatibility and developers. Full reference material stays
  in the linked documents under `docs/`. The setup guide explains RAM-aware
  recommendations, manually refreshed offline rankings and the distinction
  between parent-model scores and quantized-export quality. Label the new flow
  as development work until a release contains it. The owner-supplied setup
  screenshot shows the new flow; it does not establish release availability.
- The Images panel retains existing dense Qwen3.5 BF16 guidance and links to a
  development update for the added Qwen dense/MoE, retained Lily and Gemma paths.
  The on-site image coverage section distinguishes recorded complete-artifact
  execution from component checks and unverified larger checkpoints. It labels
  the new paths as source-build work under review, without making release or
  image-performance claims.
- Keep the homepage and technical overview concise. Setup trivia, troubleshooting,
  API contracts, template coverage and validation matrices belong in Documentation
  or its linked references.
- Scope performance figures with a short description and a link to the original
  report. Preserve exact checkpoints, conditions and limitations in the source
  records. A timing result does not establish model quality or a universal
  speedup.
- Download and release-note links use GitHub's stable `releases/latest` URLs.
  Model compatibility depends on the artifact and installed app version.

The three `benchmark-qwen38*.json` files in `public/` are byte-identical copies
of canonical aggregate records under `docs/benchmarks/data/`. When changing benchmark data, update the
source record, website copy and [benchmark index](../docs/benchmarks/README.md)
together.

The owner-supplied logo and screenshots retain their original bytes. See
[brand provenance](../docs/brand/README.md) and the
[latest screenshot replacement record](../docs/archive/runs/2026-11-11-website-screenshot-replacements.md).
Keyboard-operable tabs, focus indicators, reduced-motion handling and the skip
link remain part of the site. `robots.txt`, `sitemap.xml` and `404.html` ship
alongside the three pages.

## Hosting

`.openai/hosting.json` retains the existing Site identity and declares `dist` as
the static directory. The canonical URL is [starkengine.app](https://starkengine.app/);
[starkengine.io](https://starkengine.io/) serves the same Site.

Publication opens the current Site source in a separate ignored checkout,
mirrors only website files, builds and packages the static output, and publishes
through Sites. Credentials are passed through the workflow's stdin and never
stored. Preserve the Site's audience. See the
[initial deployment record](../docs/archive/runs/WEBSITE_DEPLOYMENT_2026_10_21.md).

## Documentation site

The Markdown manual lives in `docs-site/src/content/docs/`. Run `npm ci` and `npm run build` from `website/` to build the full site, then `uv run --no-project --python 3.12.13 python scripts/check_docs.py` from that directory to check generated docs links and assets. `node build.mjs` alone copies static assets only. See [documentation maintenance](docs-site/README.md). Preserve newer published homepage changes when synchronizing with the Sites repository.
