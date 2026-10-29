# stark_lily_core

> **Modified vendored source:** this is a selected and modified extraction of upstream Lily, not an unmodified upstream release. Upstream copyright and applicable attribution are retained in [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). See [`UPSTREAM.md`](UPSTREAM.md) for provenance and the extraction boundary.

stark_lily_core is Stark’s in-process native Metal 4 implementation of the Qwen4Exp text inference core. Its Rust library crate is named `stark_lily_core` and is packaged as stark_lily_core. It runs inside the Stark host process and does not provide an executable or service.

The extraction excludes upstream binaries, server, dashboard, HTTP APIs, tools, session persistence, probes, and ablation controls. Only resident experts are supported; the disk-backed expert cache is excluded. Read-only paging of the n-gram table from the checkpoint is allowed. MTP and vision are not supported by this load path.

The inherited precision policy is BF16 model input and activations with F32 GDN recurrent state. This statement records the selected implementation policy; it does not claim parity with upstream or establish numerical qualification in Stark.

Environment-variable selection and diagnostic controls from the upstream implementation are disabled in this integration. No performance claim is made. Source parity, numerical equivalence, and performance have not been established by this vendoring record.
