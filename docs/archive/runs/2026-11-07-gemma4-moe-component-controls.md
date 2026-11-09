# Gemma 4 MoE component controls

The bounded no-PLE Gemma converter and runtime now implement the combined dense
and sparse expert MLP. Routing receives the original pre-FFN residual through
the optional MLP hook. It applies weightless RMS, learned channel scaling and
the root-width coefficient with BF16 boundaries, then BF16 router projection,
F32 softmax/top-k renormalization and learned expert gains. Dense and expert
branches have separate source norms before their BF16 sum and final norm.

The shared routed primitive adds optional learned gains, ascending expert-ID
execution and BF16 rounding after each accumulator update. These controls
default to false; the ordinary Qwen selection retains its prior choices. Source
expert gate/up rows are streamed into native up/gate order per expert, with
flattened canonical matrix banks and selected-row gathers.

The private six-layer canonical fixture uses four experts/top two, distinct
branch norms, nonuniform router channel gains, negative/nonunit expert gains,
and negative/nonunit final layer scalars. Q/K norm weights are 0.125. Its
Transformers 5.15.1 manifest SHA-256 is
`c9015c21fd875636e30cd1bb552a7e7a8bedf259df7630a22a6611e5a2d84b49`.
All four lengths (1, 9, 31, 63) passed CPU and Metal gates with greedy agreement.
CPU centered logit RMS error was at most 0.010053; Metal at most 0.009049.
Metal centered maximum absolute error was at most 0.025209. Two CPU controls
also verify gain ordering/ID pairing and BF16 partial accumulation independently.
The converter topology/byte-order and support tests passed (124 tests).

Separately, `materialize_rope_bf16` is a default-false attention control for BF16
rotary coefficients/products without changing the attention-score algorithm.
The canonical no-PLE converter selects it. Unit-Q/K one- and two-layer canonical
CPU controls passed all four lengths, with maximum centered RMS 0.004549.
The six-layer unit-Q/K stress fixture still fails; its original results remain
preserved. The controlled fixture isolates residual/MoE mechanics and does not
replace that failed precision test. Full released 26B/31B/12B language and image
generation remain unqualified until the precision gap and exact-artifact runs
are completed. No throughput claim follows from these component checks.
