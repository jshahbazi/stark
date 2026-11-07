# Stark website

A static product website built with HTML, CSS and a small browser script.
There are no package dependencies, analytics, external fonts or model execution.

## Build and preview

From the repository root:

```sh
node website/build.mjs
uv run --project tools/converter --locked --python 3.13.13 python -m http.server 4173 --bind 127.0.0.1 --directory website/dist
```

`website/public/` is the source; `website/dist/` is ignored build output. A
standalone Site checkout builds with `node build.mjs`.

Check the build, local links, keyboard navigation and narrow-screen layout when
changing the pages. Run the repository policy check before committing.

## Content

- The homepage introduces the app, shows its three product screenshots, gives a
  short setup path and explains the OpenWork/OpenCode connections and local
  processing.
- Technical details covers the local API, featured models, memory examples,
  context reuse and three benchmark studies. Its tabs and copy control work
  without external services; content remains readable without JavaScript.
- Documentation provides an on-site quick start and organized guides for app
  connections, model compatibility and developers. Full reference material stays
  in the linked documents under `docs/`.
- Keep the homepage and technical overview concise. Setup trivia, troubleshooting,
  API contracts, template coverage and validation matrices belong in Documentation
  or its linked references.
- Scope performance figures with a short description and a link to the original
  report. Preserve exact checkpoints, conditions and limitations in the source
  records. A timing result does not establish model quality or a universal
  speedup.
- Download and release-note links use GitHub's stable `releases/latest` URLs.
  Model compatibility depends on the artifact and installed app version.

Benchmark JSONs in `public/` are byte-identical copies of canonical aggregate
records under `docs/benchmarks/data/`. When changing benchmark data, update the
source record, website copy and [benchmark index](../docs/benchmarks/README.md)
together.

The owner-supplied logo and screenshots retain their original bytes. See
[brand provenance](../docs/brand/README.md) and the
[screenshot replacement record](../docs/archive/runs/2026-11-06-website-screenshot-replacements.md).
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
