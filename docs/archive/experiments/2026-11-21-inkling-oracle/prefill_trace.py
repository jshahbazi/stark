"""Add full-width prompt taps after the archived full_target.py extension."""

import sys
from pathlib import Path


def edit(path, old, new):
    source = path.read_text()
    if source.count(old) != 1:
        raise ValueError(f"unexpected source contents in {path}: {old}")
    path.write_text(source.replace(old, new))


root = Path(sys.argv[1]) / "mlxrunner/model/inkling2"
capture = root / "oracle_capture.go"
edit(capture, "\tFullTarget bool", "\tFullTarget bool\n\t\tPrefillTrace *struct { Chunk int; Visit int }")
edit(
    capture,
    "var oracleFullTarget bool",
    """var oracleFullTarget bool
var oraclePrefillCapture bool
var oraclePrefillVisit int""",
)
edit(capture, "if oracleVisit != 0 {", """if oracleVisit != 0 {
        if oraclePrefillCapture {
            switch name {
            case "trace.embedding_rms", "trace.attn_norm_0", "trace.attn_norm_1", "trace.attn_norm_2",
                "trace.q_raw", "trace.key_raw", "trace.value_raw", "trace.r_raw", "trace.gate_raw",
                "trace.sink_raw", "trace.key_conv", "trace.value_conv", "trace.key_dynamic", "trace.value_dynamic",
                "trace.query_clipped", "trace.relative_clipped", "trace.relative_table", "key_rows", "value_rows",
                "trace.attended", "trace.attention_raw", "attention", "mlp_input", "trace.router_logits",
                "routes", "route_weights", "experts", "trace.mlp_conv":
                oracleCapture["stage."+name] = a
            }
            return
        }""")
edit(
    capture,
    "caches := target.NewCaches()",
    """if input.PrefillTrace != nil {
        oraclePrefillVisit = input.PrefillTrace.Visit
    }
    caches := target.NewCaches()""",
)
edit(capture, "oracleCapture = nil", "oracleCapture = nil\n\t\t\t\toraclePrefillCapture = false")
edit(
    capture,
    "oracleCapture = make(map[string]*mlx.Array)",
    """oracleCapture = make(map[string]*mlx.Array)
                oraclePrefillCapture = input.PrefillTrace != nil && offset < input.Start
                if oraclePrefillCapture {
                    oracleCapture["chunk"] = mlx.FromValues([]uint32{uint32(offset), uint32(n), uint32(oraclePrefillVisit)}, 3)
                }""",
)
edit(capture, 'if oracleFullTarget {\n\t\t\t\t\toracleCapture["full.hidden"]', 'if oracleFullTarget && !oraclePrefillCapture {\n\t\t\t\t\toracleCapture["full.hidden"]')
edit(capture, "for _, n := range input.Chunks {", "for chunk, n := range input.Chunks {")
edit(
    capture,
    'path := ""\n\t\tif oracleFullTarget',
    """path := ""
        if input.PrefillTrace != nil && input.PrefillTrace.Chunk == chunk {
            path = filepath.Join(outputDir, "prefill.safetensors")
        }
        if oracleFullTarget""",
)
model = root / "model.go"
edit(model, "if oracleCapture != nil && visit.pass == 1", "if !oraclePrefillCapture && oracleCapture != nil && visit.pass == 1")
edit(
    model,
    """if oracleFullTarget && oracleCapture != nil {
			if visit.layer >= 20""",
    """if oraclePrefillCapture {
            if i == oraclePrefillVisit {
                oracleVisit = 200 + i
                oraclePut("trace.embedding_rms", state.partial)
            }
        } else if oracleFullTarget && oracleCapture != nil {
			if visit.layer >= 20""",
)
attention = root / "attention.go"
source = attention.read_text()
old = "if 104 <= oracleVisit && oracleVisit <= 111 {"
if source.count(old) != 3:
    raise ValueError("unexpected source attention capture")
attention.write_text(source.replace(old, "if oracleVisit >= 200 || (104 <= oracleVisit && oracleVisit <= 111) {"))
edit(root / "conv.go", "(oracleVisit == 105 || oracleVisit == 111)", "(oracleVisit >= 200 || oracleVisit == 105 || oracleVisit == 111)")
