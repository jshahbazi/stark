"""Private aggregate-only selection evidence; no model, I/O or native changes."""
import torch

POLICY = "mathematical-topk-canonical-tie-v1"


def metrics(actual, expected):
    actual, expected = actual.float().flatten(), expected.float().flatten()
    delta = actual - expected
    maximum = float(delta.abs().max())
    relative = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected).clamp_min(1e-9))
    return {"max_abs": maximum, "relative_l2": relative,
            "passed": maximum <= 0.5 and relative <= 0.02}


def selection_evidence(native_centroids, native_masked, reference_scores, ordering,
                       reference_unmasked, reference_masked):
    native_centroids, native_masked = native_centroids.flatten(), native_masked.flatten()
    reference_scores, reference_unmasked = reference_scores.flatten(), reference_unmasked.flatten()
    reference_masked = reference_masked.flatten()
    ordering = ordering.reshape(-1)
    if (native_centroids.numel() != 2048 or reference_scores.numel() != 2048 or
            any(x.numel() != 262144 for x in (native_masked, reference_unmasked, reference_masked, ordering))):
        return {"shape_valid": False}
    permutation = (ordering.dtype in (torch.int32, torch.int64) and
                   bool(torch.equal(torch.sort(ordering).values, torch.arange(262144, dtype=ordering.dtype))))
    finite = all(bool(torch.isfinite(x).all()) for x in
                 (native_centroids, native_masked, reference_scores, reference_unmasked, reference_masked))
    if not permutation or not finite:
        return {"shape_valid": True, "native_ordering_permutation_valid": permutation,
                "finite_scores_and_logits": finite}
    ordering = ordering.reshape(2048, 128)
    sentinel = native_masked.min()
    observed = native_masked != sentinel
    observed_size = int(observed.sum())
    counts = observed[ordering].sum(dim=-1)
    whole = bool(torch.all((counts == 0) | (counts == 128)))
    clusters = torch.nonzero(counts == 128).flatten()
    cluster_ids = set(clusters.tolist())
    unique_in_range = len(cluster_ids) == 32 and all(0 <= x < 2048 for x in cluster_ids)
    minimum = native_masked[observed].min() if observed_size else sentinel
    gap = bool(sentinel < minimum)
    sentinel_correct = bool(sentinel == (minimum - 1).to(torch.bfloat16))
    native_top = torch.topk(native_centroids, k=32)
    source_top = torch.topk(reference_scores, k=32)
    native_cutoff, source_cutoff = native_top.values[-1], source_top.values[-1]
    native_multiset = bool(torch.equal(torch.sort(native_centroids[clusters]).values,
                                      torch.sort(native_top.values).values))
    source_multiset = bool(torch.equal(torch.sort(native_centroids[clusters]).values,
                                      torch.sort(source_top.values).values))
    native_strict = set(torch.nonzero(native_centroids > native_cutoff).flatten().tolist())
    source_strict = set(torch.nonzero(reference_scores > source_cutoff).flatten().tolist())
    reference_sentinel = reference_masked.min()
    reference_observed = reference_masked != reference_sentinel
    reference_size = int(reference_observed.sum())
    reference_counts = reference_observed[ordering].sum(dim=-1)
    reference_whole = bool(torch.all((reference_counts == 0) | (reference_counts == 128)))
    reference_clusters = torch.nonzero(reference_counts == 128).flatten()
    source_ids = set(reference_clusters.tolist())
    reference_minimum = reference_masked[reference_observed].min() if reference_size else reference_sentinel
    reference_gap = bool(reference_sentinel < reference_minimum)
    reference_sentinel_correct = bool(reference_sentinel == (reference_minimum - 1).to(torch.bfloat16))
    reference_multiset = bool(torch.equal(torch.sort(reference_scores[reference_clusters]).values,
                                         torch.sort(source_top.values).values))
    reference_actual_valid = (reference_whole and len(source_ids) == 32 and reference_multiset
                              and source_strict <= source_ids
                              and bool(torch.all(reference_scores[reference_clusters] >= source_cutoff))
                              and reference_size == 4096 and reference_gap and reference_sentinel_correct)
    changed = cluster_ids ^ source_ids
    source_cutoff_only = all(bool(reference_scores[x] == source_cutoff) for x in changed)
    native_topk_valid = (whole and unique_in_range and native_multiset and native_strict <= cluster_ids
                         and bool(torch.all(native_centroids[clusters] >= native_cutoff)))
    source_topk_valid = (source_multiset and source_strict <= cluster_ids
                         and bool(torch.all(reference_scores[clusters] >= source_cutoff)))
    canonical = torch.argsort(reference_scores, descending=True, stable=True)[:32]
    canonical_tokens = ordering[canonical].flatten()
    canonical_mask = torch.zeros(262144, dtype=torch.bool)
    canonical_mask[canonical_tokens] = True
    canonical_minimum = reference_unmasked[canonical_tokens].min()
    canonical_readout = torch.full_like(reference_unmasked, canonical_minimum - 1)
    canonical_readout[canonical_tokens] = reference_unmasked[canonical_tokens]
    source_mask = reference_observed

    def ambiguous_zero(scores, cutoff):
        zero = scores[scores == 0]
        return (bool(cutoff == 0) and zero.numel() > 0 and bool(torch.signbit(zero).any())
                and bool((~torch.signbit(zero)).any()))

    return {
        "shape_valid": True, "native_ordering_permutation_valid": permutation,
        "reference_support_observed_from_actual_mask": True,
        "reference_actual_mask_topk_valid": reference_actual_valid,
        "reference_observed_support_size": reference_size,
        "reference_observed_whole_cluster_blocks": reference_whole,
        "reference_strict_bf16_sentinel_gap": reference_gap,
        "reference_sentinel_equals_bf16_minimum_minus_one": reference_sentinel_correct,
        "finite_scores_and_logits": finite, "native_support_observed_from_actual_mask": True,
        "observed_support_size": observed_size, "observed_whole_cluster_blocks": whole,
        "native_unique_in_range_32": unique_in_range,
        "strict_bf16_sentinel_gap": gap, "native_sentinel_equals_bf16_minimum_minus_one": sentinel_correct,
        "native_valid_topk": native_topk_valid, "reference_topk_value_multiset_valid": source_topk_valid,
        "centroids_exact": bool(torch.equal(native_centroids, reference_scores)),
        "support_difference_only_equal_cutoff_scores": source_cutoff_only,
        "actual_selected_token_support_difference": int((observed ^ source_mask).sum()),
        "changed_selected_clusters": len(changed),
        "cutoff_tied_clusters": int((reference_scores == source_cutoff).sum()),
        "strictly_greater_clusters": len(source_strict),
        "signed_zero_cutoff_ambiguity": ambiguous_zero(native_centroids, native_cutoff)
            or ambiguous_zero(reference_scores, source_cutoff),
        "canonical_support_equals_native": bool(torch.equal(observed, canonical_mask)),
        "canonical_reference_metrics": metrics(native_masked, canonical_readout),
        "canonical_greedy_equal": int(native_masked.argmax()) == int(canonical_readout.argmax()),
        "raw_original_logits_metrics": metrics(native_masked, reference_masked),
    }


def qualify_step(evidence, original_metrics, root_equal, raw_greedy_equal, fixed):
    required = ("shape_valid", "native_ordering_permutation_valid", "finite_scores_and_logits",
                "native_support_observed_from_actual_mask", "observed_whole_cluster_blocks",
                "native_unique_in_range_32", "strict_bf16_sentinel_gap",
                "native_sentinel_equals_bf16_minimum_minus_one", "native_valid_topk",
                "canonical_support_equals_native", "canonical_greedy_equal",
                "reference_support_observed_from_actual_mask", "reference_actual_mask_topk_valid",
                "reference_observed_whole_cluster_blocks", "reference_strict_bf16_sentinel_gap",
                "reference_sentinel_equals_bf16_minimum_minus_one")
    common = (all(evidence.get(key) is True for key in required)
              and evidence.get("observed_support_size") == 4096
              and evidence.get("reference_observed_support_size") == 4096
              and evidence.get("signed_zero_cutoff_ambiguity") is False
              and evidence.get("canonical_reference_metrics", {}).get("passed") is True
              and original_metrics["feedback"]["passed"] and original_metrics["centroids"]["passed"]
              and root_equal and raw_greedy_equal and fixed)
    if not common:
        return False
    # Non-tied observed support differences are fatal even when raw logits
    # happen to fit the numerical tolerance. No epsilon defines a tie.
    if evidence["actual_selected_token_support_difference"]:
        return (evidence["centroids_exact"] and evidence["reference_topk_value_multiset_valid"]
                and evidence["support_difference_only_equal_cutoff_scores"])
    return original_metrics["logits"]["passed"]
