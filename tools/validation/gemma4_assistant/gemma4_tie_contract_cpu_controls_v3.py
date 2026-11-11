"""Public synthetic CPU controls; aggregate-only, no model or Metal access."""
import hashlib
import json
import sys
from pathlib import Path

import torch

from gemma4_tie_contract_v2 import POLICY, metrics, qualify_step, selection_evidence

torch.set_num_threads(1)
ordering = torch.arange(262144, dtype=torch.int64).reshape(2048, 128)
values = ((torch.arange(262144) % 31).float() / 8 - 2).to(torch.bfloat16)


def mask(scores, selected=None, unmasked=values):
    if selected is None:
        selected = torch.argsort(scores, descending=True, stable=True)[:32]
    tokens = ordering[selected].flatten()
    out = torch.full_like(unmasked, unmasked[tokens].min() - 1)
    out[tokens] = unmasked[tokens]
    return out


def run(name, native_scores, source_scores, actual=None, expected=False,
        source_ordering=ordering, unmasked=values, force_raw_pass=False, reference_override=None):
    if actual is None:
        actual = mask(native_scores, unmasked=unmasked)
    raw_reference = (mask(source_scores, torch.topk(source_scores, k=32).indices, unmasked)
                     if reference_override is None else reference_override)
    evidence = selection_evidence(native_scores, actual, source_scores, source_ordering,
                                  unmasked, raw_reference)
    original = {"feedback": {"passed": True}, "centroids": {"passed": True},
                "logits": metrics(actual, raw_reference)}
    if force_raw_pass:
        original["logits"]["passed"] = True
    result = qualify_step(evidence, original, True, True, True)
    return {"case": name, "expected_qualified": expected, "qualified": result,
            "control_passed": result == expected,
            "raw_logits_passed": original["logits"]["passed"],
            "support_tokens": evidence.get("observed_support_size"),
            "valid_native_topk": evidence.get("native_valid_topk"),
            "valid_actual_reference_topk": evidence.get("reference_actual_mask_topk_valid"),
            "actual_support_difference": evidence.get("actual_selected_token_support_difference"),
            "signed_zero_ambiguity": evidence.get("signed_zero_cutoff_ambiguity"),
            "canonical_max_abs": evidence.get("canonical_reference_metrics", {}).get("max_abs")}


untied = torch.full((2048,), -2.0, dtype=torch.bfloat16)
untied[:32] = 2
tied = untied.clone()
tied[:29] = 2
tied[29:33] = 1
rows = [run("untied_valid", untied, untied, expected=True),
        run("equal_cutoff_valid", tied, tied, expected=True)]
# Exercise the additional branch with an actual different Torch heap choice.
found_tied_difference = False
for seed in range(16):
    generator = torch.Generator().manual_seed(20261109 + seed)
    permutation = torch.randperm(2047, generator=generator) + 1
    spread = torch.full((2048,), -2.0, dtype=torch.bfloat16)
    spread[0] = 2
    spread[permutation[:28]] = 2
    spread[permutation[28:32]] = 1
    native = mask(spread)
    raw = mask(spread, torch.topk(spread, k=32).indices)
    if not torch.equal(native, raw):
        rows.append(run("equal_cutoff_different_raw_heap_choice", spread, spread, native, expected=True))
        found_tied_difference = True
        break
if not found_tied_difference:
    rows.append({"case": "equal_cutoff_different_raw_heap_choice", "control_passed": False})
wrong = torch.cat([torch.arange(31), torch.tensor([32])])
rows.append(run("non_tied_misselection_even_with_raw_pass_flag", untied, untied,
                mask(untied, wrong), force_raw_pass=True))
rows.append(run("actual_reference_non_tied_misselection_with_raw_pass_flag", untied, untied,
                expected=False, reference_override=mask(untied, wrong), force_raw_pass=True))
alternate_tied_reference = torch.cat([torch.arange(31), torch.tensor([32])])
rows.append(run("actual_reference_different_valid_tie_selection", tied, tied, expected=True,
                reference_override=mask(tied, alternate_tied_reference)))
missing_strict = torch.cat([torch.arange(1, 32), torch.tensor([33])])
rows.append(run("missing_strict_greater_cluster", tied, tied, mask(tied, missing_strict)))
partial = mask(tied)
partial[0] = partial.min()
rows.append(run("partial_cluster_support", tied, tied, partial))
wrong_readout = mask(tied)
wrong_readout[0] += 2
rows.append(run("canonical_readout_error", tied, tied, wrong_readout))
collapsed_values = torch.full((262144,), 512.0, dtype=torch.bfloat16)
rows.append(run("collapsed_bf16_sentinel", tied, tied,
                mask(tied, unmasked=collapsed_values), unmasked=collapsed_values))
signed_zero = torch.zeros(2048, dtype=torch.bfloat16)
signed_zero[:31] = 1
signed_zero[31] = -0.0
rows.append(run("signed_zero_cutoff_ambiguity", signed_zero, signed_zero))
duplicate_ordering = ordering.clone()
duplicate_ordering[-1, -1] = 0
rows.append(run("invalid_ordering_permutation", tied, tied, source_ordering=duplicate_ordering))
near_tie = tied.clone()
near_tie[31] = torch.tensor(0.9921875, dtype=torch.bfloat16)
rows.append(run("non_exact_centroid_cutoff", near_tie, tied))
nonfinite = tied.clone()
nonfinite[7] = float("nan")
rows.append(run("nonfinite_centroid", nonfinite, tied, mask(tied)))
out = {"policy": POLICY, "torch_version": torch.__version__,
       "torch_git": torch.version.git_version, "device": "cpu", "model_loaded": False,
       "helper_sha256": hashlib.sha256(Path(__file__).with_name("gemma4_tie_contract_v2.py").read_bytes()).hexdigest(),
       "controls": rows, "passed": all(x["control_passed"] for x in rows)}
with Path(sys.argv[1]).open("x") as handle:
    json.dump(out, handle, indent=2)
print(json.dumps(out))
sys.exit(0 if out["passed"] else 1)
