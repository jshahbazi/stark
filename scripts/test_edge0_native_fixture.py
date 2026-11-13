"""Model-free counterexamples for independent Edge0 qualification helpers."""

import numpy as np
import pytest
from edge0_native_fixture import matvec, source_route
from edge0_real_layer_activation_control import bf16, sigmoid, silu
from edge0_source_activation_control import q81_values
from validate_edge0_rust_tokenizer import batches, require
from validate_edge0_tokenizer import QualificationError


def test_router_ties_preserve_lowest_expert_index():
    ids, weights = source_route([0.5, 0.5, 0.5, -1.0, -2.0], 3)
    assert ids == [0, 1, 2]
    assert weights == [np.float32(1 / 3)] * 3


def test_router_keeps_probability_order_instead_of_sorting_expert_ids():
    ids, weights = source_route([-0.5, 0.75, -1.0, 1.5, 0.25], 3)
    assert ids == [3, 1, 4]
    assert weights[0] > weights[1] > weights[2]


def test_q81_ties_round_away_from_zero_and_zero_blocks_stay_zero():
    values = q81_values([127.0, 0.5, -0.5] + [0.0] * 29)
    assert values.tolist() == [127.0, 1.0, -1.0] + [0.0] * 29
    assert q81_values([0.0] * 32).tolist() == [0.0] * 32


def test_q81_clamps_half_scale_and_int8_values_to_finite_range():
    values = q81_values([20_000_000.0, -20_000_000.0] + [0.0] * 30)
    assert np.isfinite(values).all()
    assert values[0] == 65504 * 127
    assert values[1] == -65504 * 127


def test_f64_matvec_preserves_cancellation_and_product_bound():
    output, sumabs = matvec([[1e16, 1.0, -1e16]], [1.0, 1.0, 1.0])
    assert output.tolist() == [1.0]
    assert sumabs[0] >= 2e16


def test_bf16_uses_nearest_even_ties():
    source = np.asarray([0x3F808000, 0x3F818000], dtype="<u4").view("<f4")
    assert bf16(source).astype("<f4").view("<u4").tolist() == [0x3F800000, 0x3F820000]


def test_stable_silu_and_sigmoid_remain_finite_for_extreme_inputs():
    values = silu(np.asarray([-1000.0, 0.0, 1000.0]))
    assert np.isfinite(values).all()
    assert values[0] == 0 and values[1] == 0 and values[2] == 1000
    assert sigmoid(-1000.0) == 0 and sigmoid(1000.0) == 1


def test_private_probe_batches_never_exceed_rust_limit():
    result = list(batches(range(2050)))
    assert [len(batch) for batch in result] == [1024, 1024, 2]
    with pytest.raises(QualificationError, match="fixed-test-code"):
        require(False, "fixed-test-code")
