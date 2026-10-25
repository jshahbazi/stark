# ColGrep generation review

Reviewed on 2026-10-13 against implementation commit `6bd0b979`, the separate native build recorded in the reference report, and the final evidence artifacts. An independent reviewer inspected the worker, converter integration, validation tooling and disposition. The reviewer did not author the worker or validation tooling.

## Disposition

No remaining blocking implementation or evidence finding was identified for the scoped controlled functional acceptance. The reviewer independently verified all three evidence-report hashes, three validation-script hashes, 559 native source-file hashes, and all fifteen functional disposition checks.

The exact checkpoint, converted artifact and worker have controlled generation and tool-protocol evidence. The generated fixture carries actual calls into successive upstream-template histories and finishes spontaneously after search and a numbered read. It covers both the delay calculation and its application.

## Resolved findings

- Reject unknown fields in the greedy sampling object; a serde unit variant did not enforce that contract.
- Preserve modern nested Llama RoPE parameters, both EOS IDs and the exact checkpoint template. Reject contradictory or unsupported declarations.
- Require complete EOS stops in the strict reference gate, including an actual EOS rather than matching absent stop IDs.
- Verify source revision, Git/LFS hashes and upstream asset identities. Fail closed on malformed metadata.
- Require verified calculation and application lines in the controlled finish fixture.
- Reap malformed-readiness workers and close private pipes.
- Match upstream reasoning stripping before carrying assistant content into subsequent turns.
- Bind the numerical diagnosis to the original reference-report hash, source/model/worker hashes and reproduced native/reference token-sequence digests.
- Update the README from pending status to the scoped result.

## Preserved limits

The [strict reference report](../../../provenance/COLGREP_GENERATION_VALIDATION.json) remains failed at three of four exact greedy cases. The [diagnosis](../../../provenance/COLGREP_GENERATION_DIAGNOSIS.json) verifies identical valid finish locations in the differing case. Numerical behavior that depends on generation history is a supported explanation; the original incremental logits and exact native kernel cause remain unobserved. Native scoring comparison covers two candidates at one divergence.

The [functional disposition](../../../provenance/COLGREP_GENERATION_ACCEPTANCE.json) does not establish exact token parity, stochastic backend equivalence, broad context quality, maximum-cap native acceptance, other-checkpoint acceptance, a live search index or app integration. The [generation contract](../../GENERATION.md) describes these limits and the app-facing interface.
