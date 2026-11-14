# Production Flash-Next idle queue maintenance

[production.patch](production.patch) applies to
`5f974a939f9fa97d7d5935ea690b82c334e8397e`. It contains the four Rust source
changes used to build the measured fixed worker, including lifecycle/expiry
tests. Existing upstream copyright and licenses remain unchanged. Vendored
Metal kernels and precision policies are unchanged.

The generation worker invokes the existing Lily empty queue signal on its
owner thread while waiting for the next request, every 500 ms. It preserves
model state and prefix-cache expiry; faults clear retained state and end the
worker. No diagnostic protocol operation or background thread is added.

The [benchmark report](../../../benchmarks/2026-11-07-flashnext-idle-wake-fix.md)
and [aggregate receipt](../../../benchmarks/data/flashnext-idle-wake-fix-2026-11-07.json)
bind the patch, Rust source, generator executable, harness, fixed workload and
private native validation logs by SHA-256. Cadence v5 is in the repository;
earlier source snapshots and measurements remain bound to their own versions.

A preliminary attempt was stopped after review found a prefix-expiry edge;
its completed receipts are retained separately in the aggregate data. The
final measurements use the corrected binary and restart the entire ABBA plan.
Raw clock timelines, model fixtures, vectors and executables stay private under
ignored runtime directories. The prior empty-queue diagnostic prototype is
archived in [the earlier study](../2026-11-07-flashnext-moe/README.md).
