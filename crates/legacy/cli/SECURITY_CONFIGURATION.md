# Inherited server configuration

The inherited CLI server uses one operator-owned model and admits one inference
request at a time. Concurrent requests receive HTTP 429 before their bodies are
read. Request bodies are limited to 64 MiB and generation to 4096 output tokens;
smaller client limits are preserved. Response channels apply backpressure.
The complete request body must arrive within 30 seconds, or the server returns
HTTP 408 and releases admission. Each wait for response-channel capacity is
limited to 30 seconds; a stalled or disconnected consumer cancels the active
turn and releases admission after the session finishes cancellation cleanup.
These deadlines apply to client I/O and do not cap model inference duration.

Set `STARK_SERVER_API_KEY` in the server environment and send it using the
`Authorization: Bearer <key>` header. This key authorizes use of the configured
model and its operator resources. Both completion aliases and model-listing
aliases require the key. Browser Origin requests are rejected.

For a trusted native client on the same machine, the explicit
`--allow-unauthenticated-local` option permits a loopback binding without a key.
It cannot enable unauthenticated access on a public binding. The default bind
address remains `127.0.0.1`.

Diagnostics contain counts, timing and fixed categories. Request bodies,
generated replies and token vectors are excluded, including with
`--verbose-file-log`. On Unix, newly created log directories use mode 0700 and
files use mode 0600. Existing permissive log directories or symlinks are
rejected; the operator must secure an existing directory before enabling file
logging. Each run creates a fresh file. Expired files are pruned at startup;
there is no guaranteed expiry for files belonging to a continuously running or
stopped server.

JSON grammar schemas use bundled supported JSON Schema drafts. External
meta-schemas and file or network references are rejected. Internal references
remain supported.

Inherited Engine telemetry is disabled by default. Rust callers may explicitly
enable it with `Engine::new_with_telemetry(config, true)`. Inference failure
events contain only the fixed `inference_failed` category.
