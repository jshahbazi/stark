# OpenWork automatic setup live validation

Date: 2026-11-11. Source: `482973995752d95d21fdc5ccf689480d80585653`.

## Missing model diagnosis

The running `/Applications/Stark.app` showed a selected model with the worker
stopped. OpenWork's authenticated `models.list` query returned only its
`openwork-free` provider. Starting the selected model made the exact Stark model
appear as available without changing the connector or OpenWork configuration.
The absence observed during this check was explained by the stopped worker.
This does not establish when the worker stopped relative to the original user
screenshot, which showed a running model.

The active Stark installation was also older than the candidate installed in
`~/Applications`. The active app reported build `20261111150156547340`.
Installing a second copy had not updated that running installation. Rebuilt and
installed the candidate explicitly to `/Applications`, reopened it, and started
the previously selected model again after replacement stopped the old worker.

## Live results

- Official OpenWork: 0.18.66; bundled OpenCode v2: 2.2.6.
- Running model API ID: `qwen38-flash-next-lily-q4-xl-0f94842e`.
- OpenWork catalog: provider `stark`, exact model ID, `available: true`.
- A standalone invocation of the production `OpenWorkBeginnerHandoff` passed:
  catalog match, new empty task, saved binding, displayed task readback.
- The updated native beginner guide advanced from Connect through automatic
  configuration to Ready. Its final state reported the connector check and
  automatic task configuration. No workspace or model confirmation was used.
- The running app was verified at `/Applications/Stark.app`, build
  `20261111165448821240`. The model was left running.
- No prompt was submitted and no model inference or OpenWork tool action was
  executed as part of the handoff check. Workspace paths, task IDs, UI-control
  tokens and response content are omitted.

Measured executable SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| Installed Stark executable | `71c513e7abc6a37ffb9d9622d6dfd20dc35dbb34b9c32710ccada9a62846b2f0` |
| Installed OpenWork `opencode2` sidecar | `d2f56080ce92996ba470fc8e20d5be22500b438d764c025b399b82a7761cb275` |

This qualifies the live configuration handoff for these artifacts. It does not
measure task quality or qualify generated output. The prepared task's binding
does not change OpenWork defaults for subsequent tasks.
