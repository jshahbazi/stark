# Edge0 Phase 0 license and provenance intake

Date: 2026-11-12. Attempt: `20261112-EDGE0-01` (compatibility).
Scope: [Phase 0 of the Edge0 plan](../../EDGE0_PLAN.md#phase-0-records-provenance-and-license).

Status: primary license and origin text inspected; owner license decision
pending. This is an engineering intake of published terms, without a legal
conclusion about Stark's intended use or the publisher's rights. Phase 1
has not been authorized by this record. No model weight payload, executable
or source port was acquired; only public text and source-tree metadata were
read. No runtime or model validation was performed.

## Immutable sources and observed scope

- Code: [`Edge0-AI/Edge0` at `b7900cbb322926ccebdcd0c0c5a1280497ecff22`](https://github.com/Edge0-AI/Edge0/tree/b7900cbb322926ccebdcd0c0c5a1280497ecff22).
  The [root license](https://github.com/Edge0-AI/Edge0/blob/b7900cbb322926ccebdcd0c0c5a1280497ecff22/LICENSE)
  is Apache-2.0; the [root notice](https://github.com/Edge0-AI/Edge0/blob/b7900cbb322926ccebdcd0c0c5a1280497ecff22/NOTICE)
  identifies Copyright 2025 edge0 contributors for other project files and
  lists bundled works with their separate licenses.
- Strata: the [fork license](https://github.com/Edge0-AI/Edge0/blob/b7900cbb322926ccebdcd0c0c5a1280497ecff22/edge0-strata/LICENSE)
  is MIT, Copyright 2026 Niko1221 and the Strata contributors.
  Its [origin/change record](https://github.com/Edge0-AI/Edge0/blob/b7900cbb322926ccebdcd0c0c5a1280497ecff22/edge0-strata/EDGE0.md)
  reports fork import `f1933b1e4993fecc10d2e9360d85569619a2c42d`, upstream
  merge base `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`, and additional
  uncommitted-at-origin files identified in a
  [path/hash manifest](https://github.com/Edge0-AI/Edge0/blob/b7900cbb322926ccebdcd0c0c5a1280497ecff22/edge0-strata/edge0/IMPORTED_UNCOMMITTED.txt).
  These are publisher assertions; the originating fork history was not
  independently reconstructed.
- Component revisions: Strata's build recipe pins llama.cpp
  `3cf03257f219afbe7334045ff7c6a06ac68c627d`. The framework's
  [GGML origin](https://github.com/Edge0-AI/Edge0/blob/b7900cbb322926ccebdcd0c0c5a1280497ecff22/edge0-infer/third_party/GGML_ORIGIN)
  and [tokenizer origin](https://github.com/Edge0-AI/Edge0/blob/b7900cbb322926ccebdcd0c0c5a1280497ecff22/edge0-infer/src/app/text/UPSTREAM)
  identify llama.cpp `7ab4ee7baad2d920464cbacfad4f4b07cf111fd2`.
  This intake does not equate those distinct trees or verify the recipe's
  fetched dependency.
- Weights: [`Edge0/Edge0-145B-q3k` at `d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`](https://huggingface.co/Edge0/Edge0-145B-q3k/tree/d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74).
  Its [license](https://huggingface.co/Edge0/Edge0-145B-q3k/blob/d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74/LICENSE),
  [notice](https://huggingface.co/Edge0/Edge0-145B-q3k/blob/d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74/NOTICE)
  and [scope record](https://huggingface.co/Edge0/Edge0-145B-q3k/blob/d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74/release-scope.json)
  designate Edge0 Community Model License 1.0 for Edge0-specific contributions,
  only to the extent Edge0 owns or may license them. Upstream material retains
  its own terms; inference/framework code retains its separate code licenses.

## Commercial scope requiring an owner decision

The following paraphrases identify the decision points; the linked complete
terms control.

The [Edge0 model license, section 4](https://huggingface.co/Edge0/Edge0-145B-q3k/blob/d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74/LICENSE)
requires a separate written Edge0 commercial license before:

- Selling, renting or licensing model access, weights, or covered adapters/deltas,
  including a paid bundle primarily offering the model.
- Offering third parties inference/fine-tuning as a capability through an API,
  endpoint or general model interface, with meaningful input control, funded by
  access/usage fees, subscriptions, advertising or service-linked sponsorship.
- Offering an independent coding or office-productivity assistant using the
  covered model, funded through those product-linked mechanisms; this includes
  inference on the user's device.

The section excludes qualifying community donations/cost reimbursement,
incidental features and domain/single-purpose tools as specified in the text.
Unrelated software able to run a model alone is excluded. Section 3 permits
personal/research use, qualifying internal use, paid work without model access
for customers, and output use, subject to upstream terms. A paid custom covered
adapter/delta still falls under section 4(a). This scope is limited to rights
Edge0 controls. Edge0 commercial contact: `info@edge0.ai`.

The inherited [Qwen Community License 1.0](https://huggingface.co/Edge0/Edge0-145B-q3k/blob/d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74/third-party/Qwen-LICENSE.txt)
has separate conditions:

- Preserve its copyright and permission notices in copies/substantial portions.
  Commercial products/services above 100,000,000 monthly active users **or**
  US$20,000,000 monthly revenue must prominently display the model name in
  their interface.
- If the licensee **or an affiliate** conducts a Model as a Service or AI Work
  Assistant business, obtain a separate Qwen license before any commercial use
  of the software/derivatives. The internal-use exception requires that the
  model, its outputs and its underlying capabilities all remain unavailable
  to third parties.
- Model as a Service covers third-party inference/fine-tuning access with
  meaningful control of inputs, parameters or training data; mere relaying to
  other third-party hosts is excluded. AI Work Assistant means an independent
  coding/office product, with the stated single-purpose, other-domain and
  incidental-feature exclusions. Qwen contact: `model-business@notice.qwencloud.com`.

Edge0 permission does not supply Qwen permission. Not redistributing weights
does not settle the service, product or affiliate-business conditions. The
owner must identify the contemplated integration/product, funding and external
access, and record a decision against both complete licenses before the plan's
license gate can pass. Record any required written permissions separately;
none were supplied or verified in this intake. An instruction to begin Phase 0
is not an owner license determination.

## Lineage and rights limits

The model scope record identifies `Qwen/Qwen3.8-Flash-Next` and pins its license
reference to `812d96e10a0dd82b69d749e68693698cf6c36daa`. Its shipped Qwen
license bytes match the independently fetched
[upstream license at that revision](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/LICENSE).
The scope record expressly leaves the actual training/conversion base revision
unverified and null. A matching license hash does not establish training
ancestry, ownership of all weights or a separate Qwen commercial authorization.
Its prospective designation also preserves prior valid grants; this intake
does not establish that Stark holds such an earlier grant.

## Preservation checklist for later phases

- Keep `LICENSE`, `NOTICE`, `release-scope.json` and
  `third-party/Qwen-LICENSE.txt` byte-for-byte in a prepared model folder; bind
  them to the artifact receipt by revision and SHA-256. Keep each license's
  scope explicit. The future catalogue row must link the pinned model license.
- Keep weights and runtime artifacts ignored. This record does not authorize
  their redistribution or a model download.
- Before adapting source, tables or a layout implementation, record the exact
  component revision, path, hash, attribution and Stark changes. Add applicable
  full Apache-2.0/MIT reference texts beside existing reference licenses and
  preserve relevant root/component notices. Inspect nested component licenses
  for anything actually used; this intake is not an exhaustive dependency audit.
- For Apache-derived files, preserve required notices, identify modifications
  and deliver the applicable license/NOTICE content. For MIT-derived material,
  retain the copyright and permission text with copies/substantial portions.
  Update the distribution notice inventory when an adaptation is introduced.
- For redistributed covered model modifications, retain both sets of terms
  and attribution, identify material changes and modifier, and explain which
  materials each license governs. No modification was made in this intake.

## Inspected-text identity and checks

All hashes below are SHA-256 of HTTP response bytes, without newline
normalization. Files were fetched from the immutable revisions above using
Python standard-library HTTPS. Only listed text and GitHub tree metadata were
fetched. The web reader also inspected the pinned root Apache license; its
Hugging Face text requests failed, so direct HTTPS supplied the model text.
No payload ranges, weight files, execution or numerical tests were involved.

`G` means a path at the pinned GitHub revision; `H` means a path at the pinned
Q3_K Hugging Face revision; `Q` means the pinned upstream Qwen license revision.

| Source/path | Bytes | SHA-256 |
| --- | ---: | --- |
| G `LICENSE` | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| G `NOTICE` | 2,075 | `370471bb77311c57318c803f96a47e6511214bac190cbe12cb1e32a6e1ab0747` |
| G `edge0-strata/LICENSE` | 1,093 | `949b9b0aada7476006469e58ee4ffb0e0f68c2d9137e90a356757b61a09d37f0` |
| G `edge0-strata/README.md` | 923 | `404f5022874db3be31ae24eb38d4fb7d6f64df5a07172073c5b6fa3dd6ede593` |
| G `edge0-strata/EDGE0.md` | 13,466 | `c081604554c578caabfcb696802fe3d7ae9fb63a01ba64691d0b08cf90bf5b67` |
| G `edge0-strata/edge0/IMPORTED_UNCOMMITTED.txt` | 3,156 | `dc36938aacfee87f830c0ae6986c2a1c24c23445b5896fdd31ddae8111d45f0c` |
| G `edge0-infer/third_party/GGML_ORIGIN` | 277 | `3cdca077de6239e9a3662602ddb5ca4e83288d2216360d5395c0d466e4b9641b` |
| G `edge0-infer/src/app/text/UPSTREAM` | 946 | `c7e8e0c4fb31fba00128bc7ae67082d13d9cd8651a880827f636247f64745ad2` |
| G `edge0-infer/third_party/GGML_LICENSE`; G `edge0-strata/third_party/ggml/LICENSE` | 1,078 each | `94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d` |
| H `LICENSE`; G `licenses/edge0-model/LICENSE` | 13,059 each | `b1f5016a9f27a770e0fdd98f37c588477154d07a96f50e203de8ac53cdbefa79` |
| H `NOTICE`; G `licenses/edge0-model/NOTICE` | 2,560 each | `f2f4feabdd5b75e3ed5b41870a022bfc1dfdbcd798b8009f648cc870edbef777` |
| H `third-party/Qwen-LICENSE.txt`; G `licenses/edge0-model/third-party/Qwen-LICENSE.txt`; Q `LICENSE` | 3,235 each | `a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08` |
| H `release-scope.json` | 2,534 | `9ab2d5f7bb8fef4caf664502a73a6b4371c56ad0244e81571df48587a69a11f4` |
| H `README.md` | 30,263 | `02f204eb03e3760be10d80ec5d2ecca57b1598f87c1446598058a3ff50650f0b` |
| G `licenses/edge0-model/release-scope.json` | 13,198 | `f1038e7601f3abfedd8a307fc9f5ee12bd740d27ced9db4bd40d3f83f6f39bc1` |
| G `licenses/edge0-model/upstream-reference.json` | 2,210 | `89f61ef0098fcd2806cf37983c3fe669e3db3caf6a9248d5c34371a63fcafbcc` |

Checks: identical license/notice copies and all three Qwen copies compared
byte-for-byte; the GitHub model aggregate scope and the artifact-specific
scope intentionally have different hashes and were not substituted for one
another. These hashes identify intake text only. They do not attest to model
payload integrity, source-port correctness, inference support or throughput.
