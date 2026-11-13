from benchmark_qwen38_cadence import summarize_positions


def test_route_summaries_do_not_combine_different_mtp_workloads():
    rows = [{"route": route, "position_after_rest": 0, "decode_seconds": seconds,
             "prefill_seconds": seconds, "wall_seconds": seconds + 1,
             "host_screen": {"eligible": eligible}}
            for route, seconds, eligible in [("off", 3, True), ("on", 2, False)]]
    off, on = summarize_positions(rows, 1)
    assert off["route"] == "off" and off["pooled_decode_tok_s"] == 85
    assert on["route"] == "on" and on["pooled_decode_tok_s"] == 127.5
    assert off["eligible_arms"] == 1 and on["eligible_arms"] == 0
