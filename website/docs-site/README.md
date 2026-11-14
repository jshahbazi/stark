# Public documentation

Author pages in `src/content/docs/` as Markdown. `astro.config.mjs` defines the sidebar. Run `npm ci` and `npm run build` from the website root; output goes to `dist/docs/`. The existing homepage and technical page remain static assets. Search is a locally served Pagefind index.

Each task guide must state prerequisites, exact steps, a success check, recovery steps and a next action. Use task titles and concrete UI labels. Label development-only controls and verify release availability before promising a capability. Do not publish private request data or credentials.

## Owning references

| Public pages | Stark repository source |
| --- | --- |
| Installation, updates | `docs/INSTALLATION.md`, `apps/menu-bar/README.md` |
| Models, Playground, app connections | `apps/menu-bar/README.md`, `docs/MODEL_SUPPORT.md` |
| Model recommendations | `docs/MODEL_RECOMMENDATIONS.md` for builds containing that policy |
| First request, API, connection recovery | `docs/LOCAL_API.md` |
| Worker integration | `docs/WORKER_PROTOCOLS.md`, `docs/GENERATION.md`, `docs/SCORING.md` |
| Compatibility | `docs/TOOL_CALLING.md`, `docs/MODEL_IMAGE_STATUS.md`, `docs/VISION_FAMILY_CONTRACT.md` |

Detailed protocol schemas stay in their owning references. Update public task steps alongside changes to those behaviors. Validation records describe their named builds; keep them outside the first-run path.

## Review and publishing

Check generated internal links and fragments, desktop and narrow layouts, keyboard navigation and search after building. Follow install, connection and request guides against the target release before labeling them release-verified. A website build alone does not verify inference or client behavior.

The Sites repository is the publication source. Preserve current published homepage changes when synchronizing from the Stark repository. `public/docs.html` retains old section links and forwards to the new routes. Keep the project identity and audience unchanged.
