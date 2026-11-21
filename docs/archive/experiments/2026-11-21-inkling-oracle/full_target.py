"""Extend the pinned, instrumented Ollama source for the full-target oracle.

Apply after instrument.py and after copying capture.go into the isolated
Ollama tree. The initial two-layer captures keep their original tool hashes.
"""

import sys
from pathlib import Path


def edit(path, old, new):
    source = path.read_text()
    if source.count(old) != 1:
        raise ValueError(f"unexpected source contents in {path}")
    path.write_text(source.replace(old, new))


root = Path(sys.argv[1]) / "mlxrunner/model/inkling2"
capture = root / "oracle_capture.go"
edit(capture, "var oracleEmbedding bool", "var oracleEmbedding bool\nvar oracleFullTarget bool")
edit(capture, "\t\tEmbedding bool", "\t\tEmbedding bool\n\t\tFullTarget bool")
edit(
    capture,
    "defer func() { oracleEmbedding = false }()",
    """oracleFullTarget = input.FullTarget
	defer func() { oracleEmbedding, oracleFullTarget = false, false }()""",
)
edit(
    capture,
    "caches := target.NewCaches()",
    """if oracleFullTarget {
        tokens := make([]uint32, input.Start+4)
        for i, id := range ids[:input.Start+4] {
            tokens[i] = uint32(id)
        }
        if err := mlx.SaveSafetensors(filepath.Join(outputDir, "tokens.safetensors"),
            map[string]*mlx.Array{"tokens": mlx.FromValues(tokens, len(tokens))}); err != nil {
            return err
        }
    }
	caches := target.NewCaches()""",
)
edit(
    capture,
    "state = append(state, hidden)",
    """state = append(state, hidden)
                if oracleFullTarget {
                    oracleCapture["full.hidden"] = hidden
                    oracleCapture["full.logits"] = target.Unembed(hidden)
                }""",
)
edit(
    capture,
    'if err := step(offset, n, ""); err != nil {',
    """path := ""
    if oracleFullTarget && input.Start == 4 && n == 1 {
        path = filepath.Join(outputDir, fmt.Sprintf("prefix-%d.safetensors", offset))
    }
    if err := step(offset, n, path); err != nil {""",
)
model = root / "model.go"
edit(
    model,
    "state = layer.forward(state, messages, b, layerCaches, visit.layer%m.config.ResidualBlockSize == 0, m.config)",
    """if oracleFullTarget && oracleCapture != nil {
            if visit.layer >= 20 {
                oracleVisit = visit.layer
            } else if visit.pass == 0 && (visit.layer == 4 || visit.layer == 5 || visit.layer == 10 || visit.layer == 11) {
                oracleVisit = 100 + visit.layer
            }
        }
        state = layer.forward(state, messages, b, layerCaches, visit.layer%m.config.ResidualBlockSize == 0, m.config)""",
)
edit(
    model,
    'oraclePut("partial", state.partial)',
    """oraclePut("partial", state.partial)
        if oracleFullTarget && oracleCapture != nil {
            oracleCapture[fmt.Sprintf("full.visit.%02d.partial", i)] = state.partial
        }""",
)
edit(
    root / "experts.go",
    "values := mxfp8ExpertInput(x)",
    """values := mxfp8ExpertInput(x)
        if oracleVisit >= 20 && len(trace) == 1 {
            oraclePut(trace[0]+"_inner", values)
        }""",
)
edit(
    root / "attention.go",
    "var tau *mlx.Array",
    """if 104 <= oracleVisit && oracleVisit <= 111 {
        oraclePut("trace.key_conv", k)
        oraclePut("trace.value_conv", v)
    }
    var tau *mlx.Array""",
)
edit(
    root / "attention.go",
    "var history *nn.KVHistory",
    """if 104 <= oracleVisit && oracleVisit <= 111 {
        oraclePut("trace.query_clipped", q)
        oraclePut("trace.relative_clipped", r)
        oraclePut("trace.relative_table", table)
    }
    var history *nn.KVHistory""",
)
edit(
    root / "attention.go",
    "return projectTarget(a.Output, attended)",
    """output := projectTarget(a.Output, attended)
    if 104 <= oracleVisit && oracleVisit <= 111 {
        oraclePut("trace.attention_raw", output)
    }
    return output""",
)
edit(
    root / "conv.go",
    "inputs = append(inputs, perToken)",
    """if (oracleVisit == 105 || oracleVisit == 111) && len(trace) == 1 {
        oraclePut("trace."+trace[0]+"_dynamic", perToken.AsType(mlx.DTypeBFloat16))
    }
    inputs = append(inputs, perToken)""",
)
