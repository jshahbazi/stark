# Long-response comparison sources

As-run, content-free benchmark sources for the
[coding response](../../../benchmarks/2026-11-15-lily-long-coding-response.md)
and [planning response](../../../benchmarks/2026-11-15-lily-long-planning-response.md).
Worker, model, source and input/output hashes bind each aggregate to its run.

| Snapshot | Executed location |
| --- | --- |
| `coding_run.py` | `runtime/lily-ttt-long/run.py` |
| `coding_initial_stdin.py` | `runtime/lily-ttt-long/run_initial_stdin.py`; the first stdin preflight source |
| `coding_publish.py` | `runtime/lily-ttt-long/publish.py` |
| `coding_syntax_checker.py` | `runtime/lily-ttt-long/check_program.py` |
| `planning_run.py` | `runtime/lily-slack-long/run.py` |
| `planning_publish.py` | `runtime/lily-slack-long/publish.py` |

Restore the exact names/locations to preserve path semantics. Native binaries,
weights, original receipts, prior identity/patch files and frozen earlier
benchmark material remain ignored/private. Another environment needs its own
recorded path/source adaptation and matching identities.

The runners consume the request as a single JSON line on stdin. A tool PTY
keeps stdin open; echo is disabled before reading so the instruction enters
neither terminal output nor process arguments. The initial plain-pipe
invocation ended at EOF before receiving a request, creating results or loading
a worker. Its source and a zero-request preflight receipt are retained.

The actual requests and generated responses are never written or echoed.
The coding runner keeps unique responses in memory only through its deferred
syntax phase; the planning runner retains only structure counts after timing.
Do not add the request as a CLI argument or write generated programs to disk.

The coding checker's as-run `check_response` is strictly syntax-only. Earlier
sandbox-probe helpers remain in that source snapshot for provenance, but no
generated code is executed; runtime isolation was not qualified. A syntax pass
does not establish a playable or correct game.

All native requests and host observations are retained. Natural EOS and length
termination remain distinct. Both user requests are short inputs asking for
longer answers; these are not long-context prefill studies. Stochastic draft
depths may generate different-length answers, so report token rate and task
time separately and preserve that limitation.
