"""Apply the capture taps to an isolated Ollama v0.40.17-rc3 source tree."""

import sys
from pathlib import Path

src = Path(sys.argv[1])
base = src / "mlxrunner/model/inkling2"


def edit(path, *changes):
    data = path.read_text()
    for old, new in changes:
        if data.count(old) != 1:
            raise ValueError((path, old, data.count(old)))
        data = data.replace(old, new)
    path.write_text(data)


edit(
    base / "model.go",
    (
        "state = layer.forward(state, messages, b, layerCaches, visit.layer%m.config.ResidualBlockSize == 0, m.config)",
        """if i == 0 && oracleEmbedding && oracleCapture != nil {
            oracleVisit = -1
            oraclePut("trace.embedding_rms", state.partial)
        }
        if oracleCapture != nil && visit.pass == 1 && (visit.layer == 10 || visit.layer == 11) {
            oracleSetVisit(visit.layer, state, previous, messages, layerCaches[0], b)
        }
        state = layer.forward(state, messages, b, layerCaches, visit.layer%m.config.ResidualBlockSize == 0, m.config)
        oraclePut("partial", state.partial)
        oracleVisit = 0""",
    ),
    (
        "mixed := state.mix(l.AttentionQuery, cfg.NormalizeResidualValues)",
        """mixed := state.mix(l.AttentionQuery, cfg.NormalizeResidualValues)
        oraclePut("trace.attn_mixed", mixed)""",
    ),
    (
        "l.MessageProjections[index].apply(input, selectMessage(messages, index))",
        "l.MessageProjections[index].apply(input, selectMessage(messages, index), index)",
    ),
    (
        "inputs[index] = modelRMS(input, l.AttentionNorm[index], cfg.NormEps)",
        """inputs[index] = modelRMS(input, l.AttentionNorm[index], cfg.NormEps)
        oraclePut("trace.attn_norm_"+fmt.Sprint(index), inputs[index])""",
    ),
    (
        "l.AttentionOutput.forward(attention, nil, b, caches[3])",
        """l.AttentionOutput.forward(attention, nil, b, caches[3], "attn")
    oraclePut("attention", attention)""",
    ),
    (
        "ffnInput := subtotal.mix(l.FeedForwardQuery, cfg.NormalizeResidualValues)",
        """ffnInput := subtotal.mix(l.FeedForwardQuery, cfg.NormalizeResidualValues)
    oraclePut("trace.mlp_mixed", ffnInput)""",
    ),
    (
        "l.MessageProjections[3].apply(ffnInput, selectMessage(messages, 3))",
        "l.MessageProjections[3].apply(ffnInput, selectMessage(messages, 3), 3)",
    ),
    (
        "ffn := l.FeedForward.forward(modelRMS(ffnInput, l.FeedForwardNorm, cfg.NormEps))\n\tffn = l.FeedForwardOutput.forward(ffn, nil, b, caches[4])",
        """ingress := modelRMS(ffnInput, l.FeedForwardNorm, cfg.NormEps)
    oraclePut("mlp_input", ingress)
    ffn := l.FeedForward.forward(ingress)
    oraclePut("experts", ffn)
    ffn = l.FeedForwardOutput.forward(ffn, nil, b, caches[4], "mlp")
    oraclePut("trace.mlp_conv", ffn)""",
    ),
)
edit(
    base / "attention.go",
    ("b, keyCache).Reshape", 'b, keyCache, "key").Reshape'),
    ("b, valueCache).Reshape", 'b, valueCache, "value").Reshape'),
    (
        "var tau *mlx.Array",
        """oraclePut("trace.q_raw", q)
    oraclePut("trace.r_raw", r)
    var tau *mlx.Array""",
    ),
    (
        "var history *nn.KVHistory",
        """if oracleVisit != 0 {
        oraclePut("key_rows", oracleKV(k))
        oraclePut("value_rows", oracleKV(v))
    }
    oraclePut("trace.sink_raw", sink)
    var history *nn.KVHistory""",
    ),
    (
        "if mlx.MetalIsAvailable() && gate.DType() == qInput.DType() {\n\t\treturn projectTarget(a.Output, compiledAttentionGate(attended, gate))\n\t}\n\treturn projectTarget(a.Output, gateAttention(attended, gate, qInput.DType()))",
        """oraclePut("trace.gate_raw", gate)
    oraclePut("trace.sdpa", attended)
    if mlx.MetalIsAvailable() && gate.DType() == qInput.DType() {
        attended = compiledAttentionGate(attended, gate)
    } else {
        attended = gateAttention(attended, gate, qInput.DType())
    }
    oraclePut("trace.attended", attended)
    return projectTarget(a.Output, attended)""",
    ),
)
edit(
    base / "conv.go",
    (
        "layerCache cache.Layer) *mlx.Array",
        "layerCache cache.Layer, trace ...string) *mlx.Array",
    ),
    (
        "window, states := nn.CausalConvWindow(b, x, tail, options...)",
        """if oracleVisit != 0 && len(trace) == 1 {
        oraclePut("trace."+trace[0]+"_raw", x)
        if history != nil {
            oraclePut("history."+trace[0], history.ConvState().Clone())
        }
    }
    window, states := nn.CausalConvWindow(b, x, tail, options...)""",
    ),
)
edit(
    base / "residual.go",
    ('"math"', '"math"\n"fmt"'),
    (
        "apply(input, message *mlx.Array)",
        "apply(input, message *mlx.Array, trace ...int)",
    ),
    (
        "compressed := projectTarget(g.Down, paired).AsType(mlx.DTypeFloat32)",
        """rank := projectTarget(g.Down, paired)
        if oracleVisit != 0 && len(trace) == 1 {
            suffix := fmt.Sprint(trace[0])
            oraclePut("trace.controller_pair_"+suffix, paired)
            oraclePut("trace.controller_rank_"+suffix, rank)
        }
        compressed := rank.AsType(mlx.DTypeFloat32)""",
    ),
    (
        "return compiledControllerMix(input, message, gain, forget)[0]",
        """out := compiledControllerMix(input, message, gain, forget)[0]
        if oracleVisit != 0 && len(trace) == 1 {
            oraclePut("trace.controller_out_"+fmt.Sprint(trace[0]), out)
        }
        return out""",
    ),
)
edit(
    base / "experts.go",
    (
        "project(x, indices, inputRows *mlx.Array, grouped bool)",
        "project(x, indices, inputRows *mlx.Array, grouped bool, trace ...string)",
    ),
    (
        "e.nvfp4Inner(x, indices, inputRows, grouped)",
        "e.nvfp4Inner(x, indices, inputRows, grouped, trace...)",
    ),
    (
        "nvfp4Inner(x, indices, inputRows *mlx.Array, grouped bool)",
        "nvfp4Inner(x, indices, inputRows *mlx.Array, grouped bool, trace ...string)",
    ),
    (
        "values, activationMax := mlx.QuantizeNVFP4Activation4Over6(x, nvfp4ActivationDenominator)",
        """values, activationMax := mlx.QuantizeNVFP4Activation4Over6(x, nvfp4ActivationDenominator)
    if oracleVisit != 0 && len(trace) == 1 {
        oraclePut(trace[0]+"_inner", values)
        oraclePut(trace[0]+"_max", activationMax)
    }""",
    ),
    (
        "indices, weights := routeExperts(logits, e.Bias, e.topK, e.routeScale)",
        """indices, weights := routeExperts(logits, e.Bias, e.topK, e.routeScale)
    oraclePut("routes", indices)
    oraclePut("route_weights", weights)
    oraclePut("trace.router_logits", logits)""",
    ),
    (
        "down := e.Down.project(weighted, indices, nil, grouped)",
        """oraclePut("mid", weighted)
    down := e.Down.project(weighted, indices, nil, grouped, "down")""",
    ),
    (
        "e.GateUp.nvfp4Inner(x, indices, inputRows, grouped)",
        'e.GateUp.nvfp4Inner(x, indices, inputRows, grouped, "up")',
    ),
    (
        "e.GateUp.project(x, indices, inputRows, grouped)",
        'e.GateUp.project(x, indices, inputRows, grouped, "up")',
    ),
)
edit(
    src / "mlxrunner/server.go",
    (
        '"github.com/ollama/ollama/mlxrunner/sample"',
        '"github.com/ollama/ollama/mlxrunner/sample"\n"github.com/ollama/ollama/mlxrunner/model/inkling2"',
    ),
    (
        "defer runner.Close()",
        """defer runner.Close()
    if input := os.Getenv("INKLING_ORACLE_INPUT"); input != "" {
        return worker.Do(runnerCtx, func() error {
            return inkling2.CaptureOracle(runner.Model, input, os.Getenv("INKLING_ORACLE_OUTPUT"))
        })
    }""",
    ),
)
