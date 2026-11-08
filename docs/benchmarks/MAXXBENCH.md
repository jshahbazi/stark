# Local MaxxBench evaluations

Stark's MaxxBench runner evaluates coding-agent task completion with the actual
prepared artifact, native worker, chat template and local HTTP API. It uses the
upstream task environments, reference solutions and separate-container verifiers.
An artifact fitting in memory or passing a tool-format smoke test does not predict
its score on these engineering tasks.

## Pinned inputs

- MaxxBench: `unlap-labs/maxxbench`, commit
  `1a0df5e9c5d4bccbae131b01830d478088fea947`, 89 tasks.
- Pier: `datacurve-pier==0.3.1`.
- mini-swe-agent: `2.4.6`, with its packaged `mini.yaml` prompts, bash function
  schema, observation formatting, submission marker and `DefaultAgent` loop.
- Python: 3.12 for the pinned evaluation runner; 3.13.13 and the locked converter
  environment for Stark's API bridge.

The runner's PEP 723 lock is `scripts/run_maxxbench.py.lock`. Upstream sources
and their licenses remain in the separate checkout. The release validator
checks all released files and Docker contexts before a run. Keep generated
locks, jobs and other local files outside that checkout: its release validator
rejects extra files.

This is a local evaluation configuration. The published leaderboard uses
provider-hosted models and reasoning settings that Stark's HTTP API does not
expose. Scores from local quantized artifacts must identify their complete
configuration and must not be presented as reproductions of those rows.

## Prepare the runner

Docker with Compose must be running. A single task commonly requests two CPUs
and 8 GiB of container memory. Agent and verifier environments retain the
task's network restrictions. Model requests originate from the host, so the
task container needs neither a model credential nor network access to Stark.

From the Stark checkout:

```sh
git clone https://github.com/unlap-labs/maxxbench.git runtime/maxxbench-upstream
git -C runtime/maxxbench-upstream checkout 1a0df5e9c5d4bccbae131b01830d478088fea947
cargo +1.98.1 build --release --locked -p stark-generate -p stark-template
uv sync --project tools/converter --locked --python 3.13.13
uv run --locked scripts/run_maxxbench.py --upstream runtime/maxxbench-upstream --output runtime/maxxbench-preflight --preflight-only
```

The API bridge runs the converter environment's Python directly. This avoids
leaving an extra package-manager process between the runner and the bridge
during cancellation. The native worker and template helper must come from the
same qualified build.

## Configuration and execution

Create a private JSON matrix under `runtime/`. Each entry in `profiles` has:

```json
{
  "profiles": [
    {
      "id": "artifact-name",
      "path": "/absolute/path/to/prepared/model",
      "context": 32768,
      "max_tokens": 4096,
      "thinking": false,
      "mtp": "off",
      "temperature": 0,
      "step_limit": 250
    }
  ]
}
```

The baseline above uses production HTTP template behavior (`thinking: false`),
greedy sampling with seed zero, and MTP disabled. A model's template can still
require reasoning regardless of the optional thinking flag. `thinking: true`
is an explicitly different benchmark configuration implemented by the isolated
bridge; it is not a new public HTTP request field. Output requests above the
production API's limit fail admission instead of being silently clamped.

The local client allows 600 seconds per completion and 30 seconds per command.
Command timeouts return fixed feedback to the agent so it can recover. The
upstream task normally allows three hours for the agent and 30 minutes for its
verifier. Context overflow is a recorded budget outcome. Unexpected adapter
faults abort the attempt as infrastructure failures and cannot become model
scores.

First establish that a reference solution passes the installed grading path:

```sh
uv run --locked scripts/run_maxxbench.py --upstream runtime/maxxbench-upstream --output runtime/maxxbench-oracle --oracle --task tomlkit-toml-table-converters
```

Before reporting a full score, run the oracle for every selected task on the
same Docker platform. A passing oracle for one task qualifies only that task's
reference execution. Omitting `--task` from the oracle command checks all 89.

Then run the same task for every configured artifact:

```sh
uv run --locked scripts/run_maxxbench.py --upstream runtime/maxxbench-upstream --output runtime/maxxbench-pilot --matrix runtime/maxxbench-matrix.json --task tomlkit-toml-table-converters
```

Omitting `--task` selects all 89 tasks. Repeating `--task` selects an explicit
subset. A subset is a pilot and must not be reported as a MaxxBench score.
Use a separate output directory for a different selection, configuration,
source/build or runner revision. `--fingerprint-only` hashes model files and
exits before loading a model. It is useful for preparing an exclusive GPU window.

The runner takes an advisory `fcntl.flock` on `--gpu-lock` while hashing, building
oracle environments or keeping models loaded. Cooperating tasks must use the
same absolute lock path. It serializes complete task attempts through one
model and releases the lock only after model cleanup. Do not compile or run
another model during a reserved measurement window.

## Evidence and privacy

The host-side adapter preserves mini-swe-agent's task loop while keeping its
conversation only in memory. Trajectory saving and conversation/exception
logging are disabled. Credentials are generated per loaded model, passed to
the bridge over a private pipe, and retained only in process memory. They do
not enter task containers, arguments, environment variables, JSON or receipts.
The separate verifier still receives the task's normal patch artifact.

Private `runtime/` receipts bind results to the upstream manifest, complete
artifact file hashes, worker/helper hashes, Python/dependency lock, source
commit, host platform, runner/adapter/bridge hashes and declared settings.
Task image IDs are captured for every artifact/task attempt and checked when
resuming that attempt. Upstream's `--force-build` also disables Docker's build
cache; independent builds can have different image IDs. These receipts identify
the actual environments and do not assert byte-identical builds across models.
Per-trial `progress.json` files expose
only elapsed time, phases and aggregate model-call/token/error counts so long
attempts can be monitored without conversation logs.
Per-file hash caches are reused only while device, inode, size, modification
and change times remain identical. File-set/identity changes during a run abort
the campaign. Resuming with a different identity is rejected.

Publish counts, verifier rewards and hashes only. Keep task instructions,
generated patches, verifier logs, raw evaluation directories, models and binaries
ignored. Report model failures separately from infrastructure failures and
context/output/step limits. Do not remove failed attempts from the denominator
or infer an 89-task score from a partial run. Full task success requires the
upstream `reward == 1`; its partial metrics do not establish full success.

## Provenance

[MaxxBench](https://github.com/unlap-labs/maxxbench) is Apache-2.0 and adapts
[DeepSWE](https://github.com/datacurve-ai/deep-swe). Its `PROVENANCE.md` preserves
the individual upstream project licenses. The runner imports the pinned
upstream image-handoff hook and does not copy or modify task instructions,
reference patches or test expectations. mini-swe-agent remains an external,
locked dependency; the adapter loads its packaged templates at runtime.
