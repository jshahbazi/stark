# Inkling 2 Mini repeated tool calls with and without prefix reuse

| Field | Value |
| --- | --- |
| Attempt, date, owner | `20261124-INKCTX-03` · 2026-11-24 · Codex |
| Kind and state | Prefix reuse correctness diagnostic, completed |
| Result | A synthetic repeated tool call reproduced with zero reused tokens. The matched warm request produced the same token IDs and finish reason. |
| Evidence scope | Installed affine8 artifact, signed Stark worker, isolated managed OpenCode HTTP route |
| Disposition | Diagnostic only. The incident's original request was unavailable for an exact replay. |
| Predecessor | [Request-sized context allocation](2026-11-24-inkling-mini-request-context.md), `20261124-INKCTX-02` |

## Inputs and controls

The report concerned repeated tool calls across requests in OpenCode. Test whether the same rendered request yields a different result after clearing the prefix. Use a synthetic tool result, supply it through the normal streamed `/v1/chat/completions` route, and leave the running desktop worker alone. No tool commands were executed in these controls.

Both arms used the signed worker SHA-256 `594c68d74304fe29847eeb1ba6d4ee9d524167f46acfc9e30651116d03d54f60` from Stark source revision `b92055b983b3c5d6887abf63422d04849544af27`. The installed model revision was `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, affine8 artifact `c9e14848ad017467dc0deb91c6aa67022e181dbcdfe02a1b04024434bf9acddd`, with DSpark artifact `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`. Both runs loaded vision on an M5 Max with 128 GiB unified memory, macOS 26.7.2 (`25G332`). Hardware policies and backend provenance are shared with the predecessor.

Start each pair with a synthetic prompt that generates a tool call. Feed an assistant tool call plus its linked textual result back through the managed API. For the long probe, add four more copies of the call and result to the history, with distinct call IDs. Run the follow-up with prefix reuse, clear the worker's snapshot, and send the identical follow-up again. Compare the native generation parameters, output and public terminal finish. Use greedy sampling and a 1,024-token output limit for every arm. The short seed was 6,471 input tokens and the long seed was 49,991. The long probe's private harness SHA-256 was `e93be6e7cf902123e8774b899f5b22e645e1ee189e596af2038e081cacdcccae`.

## Results

| Tool history | DSpark | Initial physical window, seed → follow-up | Warm reused | Cold reused | Warm and cold behavior |
| --- | --- | ---: | ---: | ---: | --- |
| Single result | Off | 8,192 → 8,192 | 6,144 | 0 | Same decoded output and `stop`. Both answered correctly. |
| Single result, longer output | Off | 8,192 → 9,216 | 6,144 | 0 | Same decoded output and `stop`. Both answered correctly. |
| Single result | On | 8,192 → 8,192 | 6,144 | 0 | Same decoded output and `stop`. Both answered correctly. |
| Single result, longer output | On | 8,192 → 9,216 | 6,144 | 0 | Same decoded output and `stop`. Both answered correctly. |
| Five identical completed calls with the same result and distinct IDs | On | 51,200 → 60,416 | 49,152 | 0 | Same generated token IDs, decoded output and `tool_calls`. Both repeated the tool call. |

The first short probe exited during setup with an invalid 240-second grant lifetime, and its cleanup error masked the observation. Correcting the cleanup exposed the grant rejection. The successful probes used 120-second grants on the isolated loopback API. The renderer's 12 Inkling template and response tests also passed. Timing and peak memory were not measured.

This reproduction establishes that prefix reuse is unnecessary for the synthetic repeat and that it preserved the output on the tested prompts. The tests cover one warm restoration per paired request. The original request, a long repeated-hop warm chain, and a long repeated-tool case with DSpark off were not replayed. OpenCode v2 documents that `doom_loop` is not a current Core permission action, leaving repeated calls without that built-in stop. [OpenCode permissions](https://opencode.ai/docs/permissions/#actions)
