# Inkling 2 Mini DFlash 2 pilot

`stark-generate` can verify live DFlash 2 proposals on the original Inkling 2 Mini checkpoint on an Apple M5 Max. The worker runs the target, vision tower and verifier in Stark's Rust/Metal engine. A resident local Python process runs the BF16 draft on MLX Metal. The profile is an explicit worker option. A worker launched without it follows the existing Inkling draft configuration.

## Stage the runtime

- Use the `inkling2-mini-original` artifact with the original target weights and vision tower. The checkpoint revision is `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, and the target SafeTensors SHA-256 is `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`. [Inkling source preparation](../tools/converter/stark_convert/inkling_prepare.py)
- Place `config.json` and `draft.safetensors` from the DFlash 2 checkpoint in one local directory. The admitted revision is `4f9088ea66c21f2b99ed657d1941c306b03e84d2`, and the draft SafeTensors SHA-256 is `217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee`. The worker checks the pinned config and weight bytes before starting the process. [IncoAI checkpoint](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/tree/4f9088ea66c21f2b99ed657d1941c306b03e84d2)
- Use a clean local checkout of `z-lab/dflash` at `07ebd93db9f472af339b644bb70221ad8428328a`. The companion imports the local MLX model implementation directly. The reference code carries the MIT license and the Z Lab copyright notice. [Pinned source and license](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a)
- Use an Apple Silicon Python environment with MLX 0.31.2 and MLX-LM 0.31.3. Both are MIT-licensed and carry Apple's copyright. The pilot used CPython 3.12.13, NumPy 2.2.6 (BSD-3-Clause), and `huggingface-hub` 1.11.0 (Apache-2.0). The companion checks both MLX package versions and requires the GPU device at startup. [MLX](https://github.com/ml-explore/mlx/tree/v0.31.2) · [MLX-LM](https://github.com/ml-explore/mlx-lm/tree/v0.31.3)

Run these commands from the checkout root to create and lock a dedicated Python environment.

```sh
uv venv --python 3.12.13 runtime/inkling-dflash2-python
printf '%s\n' 'mlx==0.31.2' 'mlx-lm==0.31.3' 'numpy==2.2.6' 'huggingface-hub==1.11.0' |
  uv pip compile --python runtime/inkling-dflash2-python/bin/python --generate-hashes \
    --output-file runtime/inkling-dflash2-python/requirements.lock -
uv pip sync --python runtime/inkling-dflash2-python/bin/python \
  runtime/inkling-dflash2-python/requirements.lock
```

## Launch the worker

Save a version 1 profile with absolute paths. Point `companion` to the script in your Stark checkout and `python` to the dedicated environment's interpreter. This example selects the seven-proposal setting used in the live greedy benchmark. It uses the same target, draft and runtime revisions listed above.

```json
{
  "version": 1,
  "python": "/absolute/path/to/stark/runtime/inkling-dflash2-python/bin/python",
  "companion": "/absolute/path/to/stark/scripts/inkling_dflash2_companion.py",
  "draft": "/absolute/path/to/Inkling-2-Mini-DFlash2",
  "dflash_repo": "/absolute/path/to/z-lab/dflash",
  "greedy_drafts": 7
}
```

The optional `greedy_drafts` field sets the profile's greedy default to an integer from 1 through 7. Omit it to use 3. It also sets the depth selected by `"mtp": true` for greedy text.

Launch the worker with the profile, keeping the existing JSON-lines protocol. The profile adds a 4 GiB draft reserve to memory admission. `--prefix-cache` enables scoped text prefix reuse. `--vision-staging-root` enables the existing private image transport when the host supplies an empty, owner-private staging directory.

```sh
cargo +1.98.1 build --release --locked -p stark-generate
target/release/stark-generate \
  --model-path /absolute/path/to/inkling2-mini-original \
  --inkling-dflash2-profile /absolute/path/to/inkling-dflash2.json \
  --prefix-cache \
  --vision-staging-root /absolute/path/to/private-image-staging
```

## Drafting, images and prefix reuse

The profile selects DFlash 2 as its text draft head and leaves the DSpark graph unloaded. The draft computes one anchor-plus-seven-mask block against up to 2,047 preceding committed target feature rows. Its selector chains the seven predictions greedily over the trained top-16 lattice with F32 corrections and the checkpoint's post-correction softcap. Stark draws each emitted token from the target using the request's sampling settings and random seed. It accepts a proposal when it matches the corresponding target draw. A text request that omits both MTP fields uses the profile's greedy default or two proposals for sampled generation. Explicit `mtp_drafts` accepts 1 through 7 for greedy generation and 1 through 3 for sampled generation. Set `"mtp": false` or `"mtp_drafts": 0` to request serial generation. [Worker sampling and MTP protocol](GENERATION.md#multi-token-prediction)

The native verifier checks up to four rows per call. With four or more proposals it runs a second call only if the first call accepts its last prediction.

Streaming uses the same target draws and stop rules as buffered generation. For matching loaded artifacts, build, input and sampling settings, greedy requests are deterministic and sampled requests use the supplied seed. [Streaming protocol](GENERATION.md#streaming)

The vision tower stays loaded. Image requests use the existing image path with drafting disabled and invalidate the retained text prefix. A matching text prefix snapshot includes the committed feature window, so the worker can rebuild the draft's K/V at the restored absolute position. The companion resets between requests. During a request it keeps only committed target K/V in chronological order and receives new feature rows as they become available.

The profile must initialize successfully before the worker can advertise DFlash 2. A failed draft transaction invalidates retained state and stops the worker. The worker exchanges feature words and token IDs with the companion over private pipes. The companion disables core dumps, keeps request state in memory, and emits only fixed failure statuses in diagnostics.

The [trace experiment](model-experiments/2026-11-24-inkling-mini-dflash2-traces.md) records the original feasibility controls. The [live worker benchmark](benchmarks/2026-11-24-inkling-mini-live-dflash2.md) records measured worker results and their qualification scope.
