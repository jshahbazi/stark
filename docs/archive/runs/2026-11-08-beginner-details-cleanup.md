# Beginner details cleanup

Implementation source: `8839cc466c126133d1e8ecf13f2b6db1ea475bf3`.

Beginner now projects a small display from structured recommendation metadata.
It shows a short reason, estimated total memory, one benchmark score and a brief
published-model caveat, with a source link. Revisions, raw source URLs, harness
settings, qualification text, unknown license labels and repeated caveats are
no longer rendered in the recommendation flow. The backend report and ranking
are unchanged. Reference scores that did not influence a choice are labeled as
reference information. Measured memory pressure keeps a short actionable notice.

Existing copies no longer show new setup-space requirements. Fresh downloads
retain their download size and free-space requirement. The device line shows
chip and RAM; it no longer repeats the internal GPU budget or disk inventory.

Validation: 14 Beginner UI/report checks, 136 controller checks and 46 builder
checks passed. Swift typechecking, the final full app compilation, repository
policy and whitespace checks passed. New checks cover short presentation of
structured benchmark evidence, omitted technical metadata, recorded/unused and
missing scores, malformed score handling, HTTPS source links and installed-copy
storage presentation. The Python ranking implementation was not modified.

The signed app was rebuilt from a clean tree and atomically installed at
`~/Applications/Stark.app`; deep strict signature verification passed.
Development build: `20261108213716778462`. Signed executable SHA-256:
`4fecc8e2795ca0f8cf8f0ebb970c617183b7249b184754f6d91335bc723f7376`.

Visual verification completed in the installed Coding / Balanced flow. Qwen3.8
remained first. Expanded details showed only "Estimated memory: 111 GB",
"LiveCodeBench: 91.9/100", the short published-model caveat and a "Benchmark
source" link. The device line showed Apple M5 Max and 128 GB memory, and the
existing-copy card showed no setup-space requirement. The window was left open
for review. No weights were downloaded and no model was started for this change.
