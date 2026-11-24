# 20261124-INKCHAT-01 · Inkling 2 Mini Playground generation

## Identity and state

Executed and registered on 2026-11-24. A report of an apparently stuck
Playground code-review follow-up prompted this correctness and lifecycle
diagnostic. The matched synthetic requests passed. The original incident
remains inconclusive. Evidence is specific to the installed build and recorded
inputs. No runtime or default change was evaluated. See the
[model experiment ledger](../../model-experiments/README.md) and the
[earlier native worker checks](../../model-experiments/2026-11-21-inkling-mini-worker.md).

## Artifact, source and environment

- Stark's default `inkling2-mini-affine8-dense` vision-language artifact,
  including the DSpark head, retained its
  [qualified source and target hashes](../../model-experiments/2026-11-23-inkling-mini-affine8-default.md).
  The installed download used the publisher's public `1fb4134b9a74` upload.
- Signed local development app `20261124142452346400`, Stark `6c61016fe733`.
  Installed `stark-generate` SHA-256
  `381fc87968ab98415f91b42d3a7d89f2736ae9c6c5b951385b660854c95028d2`.
- Apple M5 Max, 128 GiB RAM, macOS 26.7.2. Peak memory was not measured.

## Workload and controls

The installed desktop Python and native worker ran in a temporary application
data directory with the installed model loaded via `Desktop.run`. A synthetic
coding turn was followed by a code-review request. Both follow-up arms used the
same rendered history, greedy sampling, seed 0, minimal template reasoning
effort (`none`), 32,768 context tokens, a 2,048-token output limit, and separate
prefix-cache scopes. Drafting was enabled for the first arm and disabled for
the second. The preceding turn used the app's default drafting policy and a
768-token output limit. These were text requests on the image-loaded worker.
On a fresh load, `Desktop.generate` reran the same synthetic exchange with a
512-token follow-up cap. The worker planned Auto with the tokenized follow-up
before that cap was applied, without starting an uncapped request.

## Correctness and lifecycle

| Request | Prompt tokens | Output cap | Sampled tokens | Finish | Visible answer |
| --- | ---: | ---: | ---: | --- | --- |
| Initial coding turn | 38 | 768 | 725 | `eos` | Yes |
| Review follow-up, DSpark enabled | 780 | 2,048 | 1,766 | `eos` | Yes |
| Review follow-up, DSpark disabled | 780 | 2,048 | 1,766 | `eos` | Yes |
| Review follow-up through `Desktop.generate` | 780 | 512 | 512 | `length` | No |

The follow-up arms produced identical token sequences in memory. Neither
sequence ended with four consecutive copies of the same short token block in
the fixed inspection range of 1–128 tokens. This check does not assess
semantic repetition. Eighty-eight model-free tests covering Inkling templates,
output framing and selected desktop reasoning, progress and cancellation paths
also passed.

Auto planned a 31,988-token response budget within the 32,768-token context.
The 512-token request produced reasoning and no answer text, emitted its final
`length` reply, and returned the desktop controller to ready. This reproduces
the possibility of an empty visible answer at a real output limit.

## Evidence, decision and limits

The [model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini#7-bias-risks-and-limitations)
reports repetitive reasoning, particularly during code debugging. It recommends
an output limit and an agent-turn limit. Stark's Playground receives numeric
progress and displays reply content after the terminal result. Its elapsed
timer advances between progress events. The screenshot alone therefore does
not establish whether tokens were still arriving when it was taken.

The original conversation was not replayed. There is no failure trace for that
request and this sample does not qualify DSpark, prefix reuse or long-context
behavior. Performance and memory are unmeasured for qualification. No prompts,
answers or token vectors were retained.
