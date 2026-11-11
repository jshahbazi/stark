# Gemma 4 named image profile controls

The five `*.config.json` and `*.processor.json` files are public metadata from
the official Google checkpoints, copied byte for byte from already cached
immutable snapshots. They contain no weights or request data. Configuration
SHA256 values match the catalog's `profile_source.files_sha256` records.

| Size | Official repository | Immutable revision | Config SHA256 |
| --- | --- | --- | --- |
| E2B | `google/gemma-4-E2B-it` | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | `1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330` |
| E4B | `google/gemma-4-E4B-it` | `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c` | `33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4` |
| 12B | `google/gemma-4-12B-it` | `43c9d014b231474fa7212dbb48a762566c2b255d` | `478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9` |
| 26B-A4B | `google/gemma-4-26B-A4B-it` | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | `ed0c1eb3633de771906e9ba004a44cc5635bcc06ee2062077c3d2e88a50707d3` |
| 31B | `google/gemma-4-31B-it` | `842da3794eaa0b77d5f08bae87a17459d91ff475` | `e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1` |

Source links use `https://huggingface.co/<repository>/blob/<revision>/config.json`
and the corresponding `processor_config.json`. These files describe Google's
Gemma 4 checkpoints and retain their published model terms. Architecture
equations and Transformers provenance are recorded in `provenance/UPSTREAM.md`.

`decoder_controls.json` freezes the canonical decoder configuration emitted by
the converter at `b5ef8164eeb9c136483d2e7b413342bb143b6e49`, with these source
configs and complete header-only BF16 decoder tensor descriptors. No payloads
were opened or allocated. Repeated layers are stored as `layer_templates` and
`layer_order`; reconstruct them before comparing. Python verifies the planner
against these controls, and native tests independently validate the same controls
against the runtime's named profile contracts. The controls establish structural
admission compatibility and do not establish model execution or numerical parity.

The retained E2B and E4B decoder controls were extracted from previously
converted image artifacts with complete configuration SHA256 values
`9acc540130bd576271650c3805467141136570ffc4111c186e317442cc3dc762` and
`977693a5a4bb1378e26db4e4eaff457eef788240c57c5f0c94a369d57b3f302e`.
Only their decoder configuration is retained here. E2B predates the embedding
materialization selectors; its original pair of false defaults is an explicit
legacy admission policy. Its artifact identity and qualification remain separate
from newly converted canonical artifacts.
