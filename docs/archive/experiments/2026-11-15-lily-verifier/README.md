# Lily verifier experiment sources

Base: `2ac1b713d893f85b97111c1279465cfaa12d7bbe`.
Attempts: `20261115-LILY-VERIFY-01` through `-04`.

The [benchmark report](../../../benchmarks/2026-11-15-lily-small-row-verifier.md)
and [public aggregates](../../../benchmarks/data/lily-small-row-verifier-2026-11-15.json)
define the workload, builds, result and decision. The tested candidate failed
the practical performance gate and is not production code. Two drafts remain
recommended. No build was installed or released.

## Source variants

| Artifact | Execution and purpose |
| --- | --- |
| `diagnostic-stage-attribution.patch` | Built and run; fixed GDN/MoE timestamp categories only. SHA `3b8214dd51c46b08588fdadaccc54a5c93249f99ae2e890d153f33197de26f4c`. |
| `head-only-prepared-unbuilt.patch` | Prepared but never built or tested. Rejected on measured whole-request ceiling before implementation execution. |
| `projection-reuse-v1-unbuilt.patch` | Prepared but never built/run; repeated environment lookup replaced before execution. |
| `projection-reuse-v2-as-run.patch` | Built and tested against the base without the finer timestamp patch. Frozen experimental selector, generic Q4/BF16 and Q8/BF16/F32 reuse, finite component test. SHA `9f406ddd07961d1d529622d13699cb119aacdbf9dbeba33fef0d6e4e5e4b0768`. |

The projection patches add original Stark wrappers around the existing exact
Lily-derived arithmetic. Existing upstream copyright and licenses remain
unchanged. A source patch's presence is not numerical, throughput or lifecycle
qualification. Unbuilt variants must not inherit v2's results.

## Harnesses and analysis

- `run-invalid-selector-as-run.py`: retained initial protocol failure.
- `run-screen-as-run.py`: completed off/stage/stage/off coarse profile.
- `run-detailed-as-run.py`: completed two-session finer profile.
- `run-comparison-as-run.py`: completed before/candidate/candidate/before
  comparison, four measured responses per prompt per session.
- `analyze_screen.py` and `analyze_compare.py`: offline aggregate analysis,
  route/input/hash/work checks, dispersion, paired changes and host summaries.
- `analyze_screen_initial.py`: preserved initial analyzer with reversed
  off/stage labels in the second coarse-profile equality pair. The corrected
  analyzer uses session four as off and session three as stage. This changes
  labels only; timing, phase attribution and all eight equality results are
  unchanged. `correct_publication.py` binds this correction and preserves the
  initial publication privately by SHA-256. The original private
  `screen-v2/analysis.json` remains unchanged; the corrected analysis has its
  own filename and hash in the public data.
- `prepare_head_candidate.py`, `prepare_projection_reuse_v1.py` and
  `prepare_projection_reuse_v2.py`: source-generation provenance; the patches
  above are the immutable implementation artifacts.
- `publish.py`: aggregate-only publication and byte-identical source copying.

The public JSON binds every archived source SHA-256, executable/patch/lock
identity, private receipt hash and all whitelisted per-request observations.
Native aggregate profiles remain private. Prompts, responses, token arrays
and probabilities were never saved.

The as-run harnesses expect to live under ignored
`runtime/lily-verifier-study/` in a checkout and reference the already sealed
baseline worker, pinned model and private identity receipt described in the
report. They reuse unchanged helpers from the prior natural-response and
sampling-policy studies; their exact source hashes are recorded in the
configuration. They require the native template helper and locked converter
environment. Copying them directly into this archive changes their computed
workspace root; this directory is an immutable record, not their execution
location. Input arrives on private stdin with PTY echo disabled, never as
shell arguments or files.

Apply each source patch independently to the named base in a separate
checkout/build target. Build workers with Rust 1.98.1 and
`cargo +1.98.1 build --release --locked -p stark-generate`. Enable
`STARK_LILY_VERIFY_PROJECTION_REUSE=1` only in candidate processes. For the
component test, use the same environment flag with
`cargo +1.98.1 test --release --locked --manifest-path vendor/lily-core/Cargo.toml projection_reuse_matches_ordinary_bytes -- --ignored --nocapture`.
The finite component result is 60 cases / 9,375,336 compared bytes / zero
unequal bytes; other tests were filtered out.

The discarded visual-renderer diff export remains private with its hash. It
was not used to build an executable or generate a measurement. Production
source was never mutated by these experimental builds. Earlier MTP studies
and archived sources are retained unchanged.
