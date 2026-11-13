"""Model-free controls for the independent Edge0 MTP binary64 oracle."""

import math
import struct
from pathlib import Path

import edge0_mtp_fixture as oracle
import pytest


class PrivateFixture(tuple):
    def __repr__(self):
        return "<private synthetic fixture>"


def require(condition, message):
    if not condition:
        pytest.fail(message, pytrace=False)


def test_dense_decoders_hand_control():
    raw = struct.pack("<e32b", 0.125, -128, 127, *range(-15, 15))
    decoded = oracle.decode_dense(raw, "q8_0", 1, 32)[0]
    assert decoded == [-16.0, 15.875] + [x / 8 for x in range(-15, 15)]
    assert oracle.decode_dense(
        struct.pack("<3H", 0x3F80, 0xBF00, 0x4000), "bf16", 1, 3
    ) == [[1, -0.5, 2]]
    assert oracle.decode_dense(struct.pack("<2f", 0.75, 1.25), "f32", 1, 2) == [
        [0.75, 1.25]
    ]
    for raw, kind, nr, nc in (
        (b"", "q8_0", 1, 32),
        (b"\0\0", "bf16", 2, 2),
        (struct.pack("<f", math.nan), "f32", 1, 1),
    ):
        with pytest.raises(ValueError):
            oracle.decode_dense(raw, kind, nr, nc)


def hand_expert():
    # Every adjacent logical code participates; scales vary by role and row.
    gate = [[c % 4 for c in range(64)] for _ in range(64)]
    up = [[(c + 2) % 4 for c in range(64)] for _ in range(64)]
    interleaved = [row for pair in zip(gate, up, strict=True) for row in pair]
    gu_scales = [[(r + 1) / 128] for r in range(128)]
    down = [[(c + r) % 4 for c in range(64)] for r in range(64)]
    down_scales = [[(r + 1) / 256] for r in range(64)]
    gc, gs = oracle.encode_q2_matrix(interleaved, gu_scales)
    dc, ds = oracle.encode_q2_matrix(down, down_scales)
    return gc + dc + gs + ds


def test_four_plane_q2_hand_control_and_independent_dot():
    blob = hand_expert()
    assert len(blob) == 3 * 64 * 64 * 18 // 64
    gate, up, down = oracle.decode_expert(blob, 64, 64)
    assert gate[0][:8] == [-1 / 128, 0, 1 / 128, 2 / 128] * 2
    assert up[0][:8] == [2 / 128, 4 / 128, -2 / 128, 0] * 2
    assert gate[-1][0] == -127 / 128
    assert up[-1][0] == 1.0
    assert down[3][:4] == [2 / 64, -1 / 64, 0, 1 / 64]
    x = [(c - 31) / 32 for c in range(64)]
    for r in (0, 3, 63):
        # Expected products use the independently declared logical codes.
        products = [((c % 4) - 1) * ((2 * r + 1) / 128) * x[c] for c in range(64)]
        expected = math.fsum(products)
        assert abs(oracle.dot(gate[r], x) - expected) <= 2e-5 * max(
            math.fsum(abs(p) for p in products), 1
        )
    wrong = oracle.decode_expert(blob, 64, 64, "row_interleave")
    assert wrong[0][1] != gate[1]
    for malformed in (blob[:-1], blob + b"\0"):
        with pytest.raises(ValueError):
            oracle.decode_expert(malformed, 64, 64)


def test_hyperconnection_grouped_norm_lowrank_and_write_hand_controls():
    residual = [1.0, -1.0, 3.0, -3.0]
    gamma = [1.0] * 4
    zero_down, zero_up = [[0.0] * 4], [[0.0] for _ in range(4)]
    zero_inject = [[0.0] * 4 for _ in range(2)]
    normalized, low, inject, mixed = oracle.hc_read(
        residual, gamma, zero_down, zero_up, zero_inject, 2, 2
    )
    a, b = 1 / math.sqrt(1 + 1e-6), 3 / math.sqrt(9 + 1e-6)
    assert normalized == [a, -a, b, -b]
    assert low == [0.0] and inject == [0.0, 0.0]
    assert mixed == [(a + b) / 4, -(a + b) / 4]
    assert oracle.hc_write(residual, [0.5, -0.5], inject, 2, 2) == [
        1.5,
        -1.5,
        3.5,
        -3.5,
    ]
    down, up = [[1.0, 0, 0, 0]], [[1.0], [0], [0], [0]]
    _, low, _, mixed = oracle.hc_read(residual, gamma, down, up, zero_inject, 2, 2)
    assert low == [oracle.silu(a / 2)]
    assert mixed[0] == (a * oracle.sigmoid(oracle.silu(a / 2)) + b / 2) / 2
    assert low[0] != oracle.silu(a) / 2  # /streams is inside SiLU.


def test_global_hidden_norm_and_final_gamma_hand_controls():
    vector = [1, 1, 3, 3]
    normalized = oracle.rms(vector, [0.5, 1, 1.5, 2])
    scale = 1 / math.sqrt(5 + 1e-6)
    assert normalized == [0.5 * scale, scale, 4.5 * scale, 6 * scale]
    grouped = oracle.flatten([oracle.rms(r, [1, 1]) for r in oracle.rows(vector, 2)])
    assert max(abs(a - b) for a, b in zip(normalized, grouped, strict=True)) > 0.1


def test_rope_halfsplit_position_and_unrotated_tail():
    vector = [1.0, 2.0, 3.0, 4.0, 11.0, 12.0]
    assert oracle.rope(vector, 0, 4) == vector
    result = oracle.rope(vector, 1, 4)
    assert result[0] == math.cos(1) - 3 * math.sin(1)
    assert result[2] == math.sin(1) + 3 * math.cos(1)
    assert result[1] == 2 * math.cos(0.01) - 4 * math.sin(0.01)
    assert result[4:] == [11, 12]
    assert math.isclose(
        oracle.dot(vector, vector), oracle.dot(result, result), rel_tol=1e-15
    )


def test_router_source_order_and_gather_renormalization():
    ids, weights = oracle.route([0, 2, 2, -3], 2)
    assert ids == [1, 2] and weights == [0.5, 0.5]
    ids, weights = oracle.route([1.5, 0.75, -0.5, 0.25], 2)
    ex = [math.exp(x - 1.5) for x in (1.5, 0.75, -0.5, 0.25)]
    p = [x / math.fsum(ex) for x in ex]
    assert ids == [0, 1]
    assert weights == [p[0] / (p[0] + p[1]), p[1] / (p[0] + p[1])]
    assert oracle.route([0, 0, 0, 0], 3) == ([0, 1, 2], [1 / 3] * 3)


@pytest.fixture(scope="module")
def synthetic():
    payload, records = oracle.make_payload()
    expected, ids = oracle.evaluate(payload, records)
    return PrivateFixture((payload, records, expected, ids))


def exceeds_stage_gate(reference, mutated):
    for name in reference:
        a, b = oracle.flatten(reference[name]), oracle.flatten(mutated[name])
        bound = 3e-5 + 3e-5 * max(abs(x) for x in a)
        if any(abs(x - y) > bound for x, y in zip(a, b, strict=True)):
            return True
    return False


@pytest.mark.parametrize(
    "mutation",
    [
        "stream_hidden_norm",
        "gamma_plus_one",
        "row_interleave",
        "scale_plane",
        "rope_position",
    ],
)
def test_wrong_policy_mutations_exceed_frozen_stage_limits(synthetic, mutation):
    payload, records, reference, ids = synthetic
    try:
        mutated, mutated_ids = oracle.evaluate(payload, records, mutation=mutation)
    except ValueError:
        # Invalid/nonfinite scale-plane interpretation must also fail the gate.
        assert mutation == "scale_plane"
        return
    require(
        exceeds_stage_gate(reference, mutated) or ids != mutated_ids,
        "wrong-policy mutation failed to exceed frozen aggregate gates",
    )


def test_causal_first_cell_catchup_and_chain_controls(synthetic):
    payload, records, serial, serial_ids = synthetic
    # At the first causal cell, each grouped query head reads exactly its own V.
    value_heads = oracle.rows(serial["value"][0], oracle.GEOMETRY["head_dim"])
    require(
        serial["attention"][0]
        == oracle.flatten(
            [value_heads[0], value_heads[0], value_heads[1], value_heads[1]]
        ),
        "first-cell attention failed exact causal self-value gate",
    )
    require(
        serial["attention_probabilities"][0] == [1.0, 0, 0, 0] * 4,
        "first-cell attention failed causal probability gate",
    )
    catchup, ids = oracle.evaluate(payload, records, full_from=2)
    for name, actual in catchup.items():
        require(
            actual == (serial[name] if len(actual) == 4 else serial[name][2:]),
            "catch-up stage failed exact serial suffix gate",
        )
    require(ids == serial_ids[2:], "catch-up routing failed exact serial suffix gate")
    chain, _ = oracle.evaluate(payload, records, chain=True)
    require(
        chain["final_residual"][:3] == serial["final_residual"][:3],
        "chain changed the explicit-input prefix",
    )
    require(
        chain["hidden_norm"][3]
        == oracle.rms(
            chain["final_residual"][2],
            oracle.decode_dense(
                payload["dense.bin"][
                    next(
                        r["offset"]
                        for r in records
                        if r["name"] == "pre_fc_norm_hidden.weight"
                    ) :
                ][: 4 * 64 * 4],
                "f32",
                1,
                256,
            )[0],
        ),
        "chain failed preceding-head-wide input gate",
    )
    require(
        exceeds_stage_gate(serial, chain),
        "chain failed to produce a distinct synthetic suffix",
    )


def test_raw_schema_coverage_and_registration_output_guards(synthetic, tmp_path):
    payload, records, _, _ = synthetic
    dense = [r for r in records if r["file"] == "dense.bin"]
    assert len(dense) == 29
    assert len({r["name"] for r in dense}) == 29
    assert all(r["offset"] % 256 == 0 for r in dense)
    for rec in records:
        raw = payload[rec["file"]][rec["offset"] : rec["offset"] + rec["bytes"]]
        assert len(raw) == rec["bytes"] and oracle.sha256(raw) == rec["sha256"]
    external = {r["name"]: r for r in records if r["file"] == "inputs.bin"}
    assert (
        external["main_head"]["dtype"] == external["token_embedding"]["dtype"] == "q8_0"
    )
    assert external["main_head"]["shape"] == [128, 64]
    layout = oracle.expected_layout()
    assert len(layout) == 132 and len({r["name"] for r in layout}) == 132
    by_name = {r["name"]: r for r in layout}
    assert by_name["catchup/key"]["cell_indices"] == [0, 1, 2, 3]
    assert by_name["catchup/readout"]["cell_indices"] == [2, 3]
    assert by_name["chain/readout"]["shape"] == [4, 128]
    assert by_name["serial/routed_down_sumabs"]["shape"] == [4, 128]
    with pytest.raises(ValueError, match="private runtime"):
        oracle.generate(tmp_path / "outside-runtime")
    assert not (tmp_path / "outside-runtime").exists()
    with pytest.raises(ValueError, match="private runtime"):
        oracle.generate(Path(oracle.REPO) / "runtime")


def test_pinned_sources_match_immutable_git_blobs():
    if not Path("/tmp/edge0-intake/github-tree.json").is_file() or not all(
        (oracle.SOURCE_ROOT / name).is_file() for name in oracle.SOURCE_FILES
    ):
        pytest.skip("optional pinned reference source receipt is absent")
    records = oracle.source_bindings()
    assert len(records) == len(oracle.SOURCE_FILES)
    assert all(
        len(record["sha256"]) == 64 and len(record["git_blob"]) == 40
        for record in records
    )
