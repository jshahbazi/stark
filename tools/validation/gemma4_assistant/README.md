# Gemma 4 assistant selection validation

These source tools consume private tensor payloads on stdin and publish only
aggregate checks. Model weights, payloads and receipts remain ignored. The
reference environment is Python 3.13.13, Torch 2.14.1 at
`5c4886908584029761b579af026dcfb627c84070`, Transformers 5.15.1 and safetensors
0.8.0, installed through the experiment's pinned `uv` environment. Nothing in
these scripts downloads models or dependencies.

`gemma4_assistant_round_oracle.py` preserves the original raw comparison and
unchanged maximum absolute 0.5 / relative L2 0.02 limits. Its SHA-256 is
`be71c7475d16adaba90c8a0842ebad2ed0653944b2ff53df71a6cf3bd63fe3dd`.
Its 1,792-token raw masked-logit failure remains failed in the source-bound
record. Do not overwrite or relabel that result.

`gemma4_assistant_round_tie_contract_v2.py` applies the separately registered
`mathematical-topk-canonical-tie-v1` policy and retains every raw metric and
`raw_original_passed` flag. It binds the adjacent `gemma4_tie_contract_v2.py`
by hash. Both actual native and actual reference masked outputs must expose
4,096 non-sentinel tokens in 32 whole clusters, with valid finite values,
ordering permutation, strict BF16 sentinel gap and selected-minimum-minus-one
sentinel. Every non-tied support difference fails. A tied difference requires
exact centroid equality and top-k score multisets, plus an independently
computed canonical readout at the original numerical limits. Signed-zero
cutoff ambiguity fails. Original root, greedy, fixed-position, feedback and
centroid gates remain required.

The reference wrapper takes the pinned source-companion directory and target
weights file as arguments. The ignored native test
`private_e2b_assistant_wrapped_handoff_and_transactional_batch` sends its
ephemeral JSON payload over the child's stdin using
`STARK_GEMMA_ASSISTANT_ORACLE_PYTHON` and
`STARK_GEMMA_ASSISTANT_ORACLE_SCRIPT`. Set the latter to this tracked wrapper,
and set the target, companion, source and receipt environment paths described
in the test. Run the source-matching test executable only within the agreed
exclusive model slot and unchanged continuous memory/foreign-worker guard.
The ordinary native output and production centroid selection are unchanged.

Run public synthetic CPU controls without loading a model or Metal backend:

```sh
"$GEMMA_REFERENCE_PYTHON" tools/validation/gemma4_assistant/gemma4_tie_contract_cpu_controls_v3.py runtime/gemma-tie-controls-fresh.json
```

The receipt path must be fresh. Fourteen positive/negative controls cover
valid untied and tied selections, actual reference support that differs from
a second heap selection, invalid native/reference top-k choices even with a
forced raw-pass flag, missing strict-greater clusters, partial support, readout
errors, collapsed sentinel, signed zeros, invalid ordering, non-exact cutoff
scores and non-finite inputs. Only aggregate results are written.

See the source-bound [family ledger](../../../docs/model-experiments/2026-11-09-gemma-next-image-parity.md)
for immutable tool/binary/artifact identities, independent review and observed
outcomes. No serving or performance qualification follows from this oracle
alone.
