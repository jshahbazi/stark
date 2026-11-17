# Very-long-context Lily performance sources

Attempt: `20261115-LILY-LONGCTX-01`.
Source checkout: `a4a1cc3021281a329cd42f0f7bf350a14a25bee1`.

The [benchmark report](../../../benchmarks/2026-11-15-lily-long-context-infinitebench.md)
and [public aggregates](../../../benchmarks/data/lily-long-context-infinitebench-2026-11-15.json)
describe the exact input source, 88,398-token request, fixed run, metrics,
host observations and evidence limits. This is one InfiniteBench-derived
performance case, not a suite score or summary-quality qualification.

## Immutable sources

- `prepare.py`: fetches the pinned author dataset in RAM, verifies its full
  LFS SHA-256, extracts the pinned upstream prompt as literal data, and chooses
  the first unmodified whole record in the declared native token range.
- `run-as-run.py`: exact executed B/C/C/B harness; explicit two drafts, images
  loaded, full-context warmups, fresh requests, streaming, deadlines and
  aggregate-only output/work checks. No prompt, response or vector is saved.
- `analyze.py`: standalone offline source/integrity checks and distributions,
  same-repetition pairing, per-half changes, counts and host summaries. It
  applies no automatic promotion gate and does not exclude busy observations.
- `publish.py`: copies these sources unchanged and publishes only whitelisted
  aggregate fields and private-receipt hashes.

The public JSON binds all four source hashes and nine private aggregate
receipt hashes. The actual internet corpus and generated summary are absent
from this archive and were never persisted. Dataset and repository license
declarations, pinned source URLs and their hashes are in the public record.

The harness expects these files under ignored `runtime/lily-long-context/`
in the recorded checkout; its computed workspace root depends on that
location. It references the sealed baseline, candidate, native template
helper and model identity receipt from the earlier verifier study. Their
exact hashes, artifact revision, sampling and context settings are recorded.
Place the archived `run-as-run.py` at `runtime/lily-long-context/run.py` for
the recorded layout. This archive is an evidence record, not its execution
directory. Use the locked converter environment with `uv`, Python 3.13.13.

The kernel candidate and its build provenance remain in the
[previous source archive](../2026-11-15-lily-verifier/README.md); it was not
rebuilt or modified for this case. Reusing its historical component result
does not broaden numerical or lifecycle qualification.

Completed outcome: 16 measured responses plus four warmups, all matching
output/work identities. Median paired decode throughput increases 5.40%
(8/8); total request time decreases 2.39% (7/8). Source, two-draft default,
installed app and earlier benchmark data remain unchanged. The candidate is
experimental and no PR, installation or release was performed.
