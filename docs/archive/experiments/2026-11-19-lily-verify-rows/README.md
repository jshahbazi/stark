# Lily verify-row experiment sources (unadopted)

Base: `4d09b55045f24eab81da0a9b5b024b9c02318459`.
Attempts: `20261119-LILYROWS-01` (hyper-connection rows),
`20261119-LILYROWS-02` (GDN verify replay, projection follow-up only) and
`20261119-LILYROWS-03` (routed-expert rows).

These patches are unadopted experimental source. None of their code is on the
integration working tree of the Lily MTP speed project; that tree carries only
the adopted GDN replay rollback (`gdn-replay-v1`, not archived here) together
with the host round-trip, depth-policy and benchmark work. The experiment
records define each workload, result and decision:

- [20261119-LILYROWS-01](../../../model-experiments/2026-11-19-lily-hc-verify-rows.md)
- [20261119-LILYROWS-02](../../../model-experiments/2026-11-19-lily-gdn-verify-replay.md)
- [20261119-LILYROWS-03](../../../model-experiments/2026-11-19-lily-moe-verify-rows.md)

## Source variants

| Artifact | Attempt | Execution and decision |
| --- | --- | --- |
| `hc-rows.patch` | `20261119-LILYROWS-01` | Built and run. Shared-weight exact hyper-connection down kernel for verify rows, byte tests and timing tests; includes the attempt's draft record. Byte-exact and passed the eight-arm output-hash gate, but the gain is below measurement resolution; not adopted on risk against benefit. SHA-256 `8e39514c3ed1c7e4652422bcd342ccc80f2a24fd2d0ac92914ca20e70ad06308`. |
| `moe-rows.patch` | `20261119-LILYROWS-03` | Built and run. Default-off routed-expert sharing kernels across verify rows, a private selector, an overlap diagnostic and byte tests; includes the attempt's draft record. Byte-exact but not faster; rejected by its author. SHA-256 `49235c95fb57957830966f80efbc4ef6160a81f46f63d5881321c2963d8402cf`. |
| `gdn-replay-v2-projections.patch` | `20261119-LILYROWS-02` follow-up | Built and run. The GDN replay change plus switched-off, test-only GDN input-projection row kernels. The only faster restructuring was byte-identical in 288 synthetic cases but changed 3,476,470 real-model verify values and every MTP arm's output hash; all variants rejected. Only its replay part (identical in intent to `gdn-replay-v1`) is adopted, from the v1 patch. SHA-256 `6b6fb81a11b348ad375e9307c7514830b4076322a1a92f73e1d9c48dbcdde122`. |

Each patch is a `git diff` against the base above, saved by its workstream and
copied here byte for byte; `.gitattributes` disables whitespace handling for
these files so the hashes stay valid. Apply each one independently to the base
in a separate checkout and target directory. A patch's presence here is not
numerical, throughput or lifecycle qualification. The variants modify
Stark-local code in `vendor/lily-core`; existing upstream copyright headers
and licenses are unchanged. Run files, stage profiles and private receipts
stay under ignored `runtime/`; prompts, responses, token arrays and
probabilities were never saved.
