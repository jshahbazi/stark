# Website copy and structure review — 2026-11-06

## Scope

Local review changes against source commit
`30ee6d73df9f4e04e4446971a2483986e4a6708f`:

- Replace the broad hero privacy promise with Stark's local processing and
  the specific statement that nothing is transmitted to us. Link the connected
  app/tool data boundary beside it. Remove repeated wording and “LLM”.
- Keep the JevK5 feature explanation in the Decision catalog panel. Remove
  the separate dark band and the repeated explanation in the Playground caption.
- Put Qwen3 4B 4-bit first at 16 GiB and add an explicitly estimated 32 GiB
  planning example. Keep the memory and tool-support guidance adjacent.
- Consolidate Beginner/Advanced setup guidance in Get started. Keep the
  recommendation disclosure and the shared-model/mode-switch explanation there.

This change does not alter model admission, recommendations in the app,
conversion, inference, or existing validation records. The user reviewed the
local preview and approved pushing and publishing these changes on 2026-11-06.

## Memory evidence and limits

The existing Qwen3 4B 4-bit recipe has recorded native chat and bounded
OpenWork evidence on an M5 Max with 128 GiB memory. That run explicitly does
not establish fit on a smaller Mac. Its current 32K-context planner estimate
is 11 GiB; Qwen3 0.6B BF16 is estimated at 9 GiB. See the
[OpenWork run](2026-11-04-openwork-beginner-e2e.md) and
[planner review](2026-11-05-hardware-model-recommendation-review.md).

The new 32 GiB card names Qwen3 14B 4-bit as an estimated starting point.
The example calculation uses
[`mlx-community/Qwen3-14B-4bit`](https://huggingface.co/mlx-community/Qwen3-14B-4bit/tree/a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4)
at revision `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4`.
Hub metadata was checked on 2026-11-06 without downloading weights:

| Weight shard | Bytes | SHA-256 |
| --- | ---: | --- |
| `model-00001-of-00002.safetensors` | 5,354,381,380 | `5795efcfc7c96fd273e600562e8b111bfcc427415de9001d0a07e70cd99cff19` |
| `model-00002-of-00002.safetensors` | 2,953,517,134 | `2814562d654fe2d541fd4682804a0ccaa400e79701872c8e9f5998cf9481fdf8` |

The exact payload totals 8,307,898,514 bytes. Geometry from the
[Qwen3-14B configuration](https://huggingface.co/Qwen/Qwen3-14B/blob/40c069824f4251a91eefaf281ebe4c544efd3e18/config.json)
is 40 layers, 8 KV heads, 128 head dimensions and 40 query heads.

Instantiating the current
[`Recipe.runtime_memory`](../../../tools/converter/stark_convert/guided_models.py)
with that payload and geometry gives:

| Context tokens | Rounded runtime planning estimate |
| ---: | ---: |
| 8,192 | 13 GiB |
| 32,768 | 17 GiB |

The calculation includes 128 MiB of installed-output overhead, BF16 key/value
cache budgeting for context plus a 1,024-row attention suffix, and the larger
of 3 GiB workspace or the fallback attention scores plus 1 GiB. The final
division rounds upward to whole GiB; it does not add another GiB reserve.

This is a metadata-based calculator example, not measured peak memory, native
artifact qualification, a newly approved app recipe, or a run on a 32 GiB Mac.
Available memory, Metal's budget and context length still affect fit. The site
labels this card “Estimated fit” and directs users to Guided setup for their
Mac. No generic 32B-on-32-GiB claim is added.

## Verification

- Static build, browser-script syntax and repository policy checks passed.
- Browser inspection at 1,280, 774 and 390 CSS pixels found no horizontal
  page overflow. Memory cards use four, two and one columns respectively.
- The page has no duplicate IDs, broken local fragment links or missing
  ARIA label/control targets.
- The Decision tab reveals the consolidated JevK5 explanation; the moved
  recommendation disclosure opens and closes in Get started.
- The existing artwork, screenshots and benchmark result files are unchanged.
  No inference or new performance/quality validation was run.
