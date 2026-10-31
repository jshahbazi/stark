# Model context capacity validation

Date: 2026-10-31. Source: base revision `a395f68b22ed2a74cb8b0b6ceacbd3160a880705`
plus the context-capacity patch committed with this record. Existing dated
validation and benchmark records remain unchanged.

Generation, scoring, Decision mode, desktop advanced options and API metadata
now use the loaded model's configured context maximum. The fixed 16,384-token
worker/desktop ceilings and Lily's inherited 262,144-token host ceiling were
removed. Lily reserves at most 16,384 tokens initially and grows working buffers
as needed. Shader representation checks and request byte limits remain active.

Validation:

- Converter suite: 5,092 passed, 88 skipped. This ran in the shared worktree,
  including preexisting request-parser changes that remain outside this commit.
- Generation protocol tests: 44 passed; scoring tests: 21 passed.
- Both release workers built with Rust 1.98.1; scorer used `optimized`.
  Both workers passed release Clippy with warnings denied.
- Both final release workers processed a synthetic 16,385-token input with
  a 32,768-token context. The loaded dense model reported 40,960 tokens.
  Both rejected an oversized context and successfully processed the next request.
- Native engine CPU/Metal test targets compiled on the pinned nightly;
  the context-capacity reservation test passed.
- Vendored Lily capacity-rounding and grid-arithmetic tests: two passed.
  Physical Metal controls passed at 262,145 and 1,048,576 tokens for BF16
  dense attention and Q8/affine64 sparse attention: six route/length cases.
  These use one query/head and exact-zero K/V readouts, with output initialized
  to one to detect an omitted dispatch.
- Generation process boundary checks, repository policy, Rust formatting,
  Python lint/format and diff whitespace checks passed.

Final native smoke executable SHA-256:

| Worker | SHA-256 |
| --- | --- |
| `stark-generate` | `0c1f97b674ee7ff5e1d85bfc0c01347b49999322fe278f1a3d100c00e5eacef0` |
| `stark-score` | `039874b586527a43df8cb2184384970a4d825f4cecc6946696909ead3bf93667` |

This establishes admission, buffer-boundary controls and recoverable rejection.
It does not qualify full-length model quality, memory requirements or throughput.
The playground's initial 8,192-token context default and generation's 4,096-token
output cap remain unchanged. No installed app or release was replaced.
