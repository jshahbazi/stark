"""Ephemeral stdin tensors; stdout contains numerical aggregates only."""
import json
import sys
import hashlib
from pathlib import Path
from gemma4_tie_contract_v2 import POLICY, selection_evidence, qualify_step

import torch
from safetensors import safe_open
from transformers import Gemma4AssistantForCausalLM

helper_hash = hashlib.sha256(Path(__file__).with_name("gemma4_tie_contract_v2.py").read_bytes()).hexdigest()
if helper_hash != 'd47525f9d46bb0a5051f3982cf88e3fb697751c141ff009bd994a351f1a2f597':
    print(json.dumps({"passed": False, "selection_contract_identity_valid": False}))
    sys.exit(1)
torch.set_num_threads(1)
payload = json.load(sys.stdin)
steps = payload["steps"]
head = Gemma4AssistantForCausalLM.from_pretrained(
    sys.argv[1], local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
).eval()


def tensor(values):
    return torch.tensor(values, dtype=torch.bfloat16)


def kv(values, rows, width):
    return tensor(values).reshape(1, rows, 1, width).transpose(1, 2).contiguous()


first = steps[0]
shared = {
    "sliding_attention": (
        kv(first["local_keys"], first["local_rows"], 256),
        kv(first["local_values"], first["local_rows"], 256),
    ),
    "full_attention": (
        kv(first["global_keys"], first["global_rows"], 512),
        kv(first["global_values"], first["global_rows"], 512),
    ),
}
hidden = tensor(first["hidden"]).reshape(1, 1, -1)
results = []
with safe_open(sys.argv[2], framework="pt", device="cpu") as target, torch.inference_mode():
    for index, step in enumerate(steps):
        embedding = tensor(step["embedding"])
        row = target.get_slice("decoder.embedding.embedding.weights")[step["root"] : step["root"] + 1].reshape(-1)
        expected_embedding = (row * torch.tensor(39.25, dtype=torch.bfloat16)).to(torch.bfloat16)
        joined = torch.cat([embedding.reshape(1, 1, -1), hidden], dim=-1)
        output = head(
            inputs_embeds=joined,
            position_ids=torch.tensor([[first["position"]]]),
            shared_kv_states=shared,
            output_hidden_states=True,
        )
        centroids = head.masked_embedding.centroids(output.hidden_states[-1])
        metrics = {}
        for name, expected in (("feedback", output.last_hidden_state), ("centroids", centroids), ("logits", output.logits)):
            actual = tensor(step[name]).float().flatten()
            expected = expected.float().flatten()
            delta = actual - expected
            maximum = float(delta.abs().max())
            relative = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected).clamp_min(1e-9))
            metrics[name] = {"max_abs": maximum, "relative_l2": relative, "passed": maximum <= 0.5 and relative <= 0.02}
        native_centroids = tensor(step["centroids"]).flatten()
        native_clusters = torch.argsort(native_centroids, descending=True, stable=True)[:32]
        reference_clusters = torch.topk(centroids.flatten(), k=32).indices
        ordering = head.masked_embedding.token_ordering.reshape(2048, 128)
        native_support = set(ordering[native_clusters].flatten().tolist())
        reference_support = set(ordering[reference_clusters].flatten().tolist())
        root_equal = bool(torch.equal(embedding, expected_embedding))
        greedy_equal = int(tensor(step["logits"]).argmax()) == int(output.logits.flatten().argmax())
        fixed = step["position"] == first["position"]
        raw_passed = root_equal and greedy_equal and fixed and all(value["passed"] for value in metrics.values())
        evidence = selection_evidence(
            native_centroids, tensor(step["logits"]), centroids,
            ordering, head.lm_head(output.hidden_states[-1]), output.logits,
        )
        contract_passed = qualify_step(evidence, metrics, root_equal, greedy_equal, fixed)
        results.append({
            "step": index + 1,
            "metrics": metrics,
            "root_embedding_exact": root_equal,
            "greedy_proposal_equal": greedy_equal,
            "fixed_position": fixed,
            "selected_token_support_difference": len(native_support ^ reference_support),
            "raw_original_passed": raw_passed,
            "selection_contract": evidence,
            "passed": contract_passed,
        })
        # Independent reference feedback chains subsequent proposals.
        hidden = output.last_hidden_state

print(json.dumps({
    "reference": "transformers 5.15.1 eager BF16 CPU; independent three-step feedback chain",
    "oracle_policy": POLICY,
    "selection_contract_module_sha256": helper_hash,
    "raw_original_passed": len(results) == 3 and all(value["raw_original_passed"] for value in results),
    "position": first["position"],
    "local_rows": first["local_rows"],
    "global_rows": first["global_rows"],
    "local_owner": first["local_owner"],
    "global_owner": first["global_owner"],
    "steps": results,
    "passed": len(results) == 3 and all(value["passed"] for value in results),
}))
