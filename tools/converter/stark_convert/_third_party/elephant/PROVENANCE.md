# Elephant 1 protocol provenance

Model source: https://huggingface.co/baselabs/elephant-1

Model revision: `b3f8a93462fb1051a9de031e94a7ff380e254072`.

Protocol source: https://github.com/OmniJev/OneJev

Protocol revision: `81ce62f1597c91e46767d4d02d6ac2e18534fe94`.

Pinned source file SHA-256:

- `qev/prompt.py`:
  `17dfae09e4dff47cb8ce80e172eba116f188b82b70c2777bd14fd9b8a71680de`
- `qev/schema.py`:
  `bce4b7d353ac115a9e50aef284975ddeed9a3bb452e3fd928a556668705bc2f3`
- Elephant `serve.patch` Git blob SHA-1:
  `7a432e8d254a9838b04ca3a155547ec3b3c70e78`

`elephant_protocol.py` is a bounded Python port of `qev/prompt.py`,
`qev/schema.py`, and the response assembly behavior from `qev/answers.py`.
The port keeps the frozen `qev-labels-v2` text layout, 256 answer labels,
OneJev confidence formulas, and native block-attention contract. It delegates
chat-template evaluation, image preprocessing, state execution and transport
to Stark. It serializes one question branch at a time and rejects Locate and
generative use.

OneJev is Apache-2.0. Its license text is included at
`../decider/LICENSE.txt` in the packaged converter resources. Elephant's model
card at the pinned revision declares MIT licensing; that declaration and the
original model README are retained in prepared artifacts.

This provenance does not establish checkpoint correctness, performance, or
installed-build qualification.
