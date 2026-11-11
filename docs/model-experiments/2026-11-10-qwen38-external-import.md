# Qwen3.8 external-table affine import

Attempt `20261110-Q38IMPORT-01`, executed 2026-11-10. The catalog and
generic import now recognize the external-table affine layout already handled
by the native Qwen3.8 loader. Catalog discovery checks pinned config and file
inventory. Preparation verifies selected source hashes, checks shard and n-gram
headers, and preserves the source weight bytes and their storage precision.

| Role | Repository | Revision |
| --- | --- | --- |
| Candidate | `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw` | `a0e89db090a3e4b2e8ea2118fe9d0324d3854aca` |
| Storage-layout control | `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit` | `788c07a8725642bccf13a0a3b8a21954ab84b4f2` |

The live catalog returned the candidate at its exact revision and reported
107,303,406,192 weight bytes, including the n-gram table. Text preflight
returned conditional support. Bounded range reads covered the headers of all
101 indexed shards and the n-gram table for both revisions. Their parsed
indexes and headers matched, covering 3,167 indexed tensors and three table
planes. The config difference was the iQ calibration annotation in the two
quantization fields. The layout comparison covers tensor names, shapes, dtypes,
and offsets. It leaves the quantized payload values and output behavior
unmeasured. Synthetic fixtures exercised publication, offline manifest replay,
index mutation, unsupported widths, and incomplete table rejection.

| Candidate evidence | SHA-256 |
| --- | --- |
| Source config | `9a2ab86be9f4860a1ac78c4d908aa78a4ad1522d0358e68bbc837af20161fe7a` |
| Source tokenizer config | `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27` |
| Extracted chat template | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| Parsed header inventory | `7c248e2355ffe5bdb1495aee425876287caa8f156051a1370954e9b1b8fc0eb8` |

The exact template resolves to the existing `qwen38-xml-v1` parser. Tool-call
generation, MTP execution, and exact prefix restoration for this checkpoint
remain unverified. The import selects the existing external-affine backend,
whose 4,096-token snapshot route restores the GDN recurrence, convolution
windows, and token history alongside retained attention and indexer rows.
The worker remains responsible for capability and invalidation checks.
The default import is text generation. The source retains its vision tensors
and MTP head, and image admission still requires an exact image artifact check.

The full model was too large for the approximately 62 GiB of available local
storage. Exact payload verification, native load, generation, image parity,
memory, and throughput were deferred. The [earlier iQ-MLX investigation](../benchmarks/QWEN38_IQ47_INVESTIGATION_2026_10_29.md)
records its own source-bound results before the external-affine loader existed.

## Catalog header admission follow-up

The metadata-only filter also offered the iQ-MLX 3.3 bpw revision
`4fabeb5ee1ef4577cb24a58b495341b5b85c866c`. Its config uses the same default
affine fields as 4.7 bpw. A pinned expert header showed 3-bit packing with
groups of 128, outside the native requirement of 4-bit packing with groups
of 64. The revised catalog checks every selected shard header and the
n-gram table header against the native text and declared draft layout.
It leaves 4.7 bpw downloadable and lists 3.3 bpw as unavailable with the
packing mismatch. Remote preparation repeats the check before bulk downloads,
and local preparation rechecks the staged headers. These checks cover
structure and leave native execution and vision qualification open.
