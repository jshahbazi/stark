# Locked N-API native identity patch

These sources are vendored from the exact crate versions already in Stark's
root lock. The root manifest patches only these three packages; their versions
and transitive dependency versions are unchanged. `LICENSE` preserves upstream's
MIT license. Original normalized manifests, original manifests, README files,
source files and crate VCS metadata are retained; generated Cargo cache state
and upstream package-local lockfiles are omitted.

| Crate | Upstream Git revision | Original registry archive SHA-256 |
| --- | --- | --- |
| napi 3.10.3 | 1ac467e06e71f78b983630926c7908894d08e496 | 0c71997d6f7ad4a756966e452426848ac27d3b37a295302d63afbbcce0270f93 |
| napi-derive-backend 5.1.1 | 2785de583a97e49adea8194090fca2ee12f067c8 | ddd961eb2aa8965e3f29722d754f3a86907eb1984e2fbcbe3fe87b9a02d6bfba |
| napi-sys 3.2.2 | dea608eae7481a47d64aab563a2ab5cdd8eda03c | 1f5bcdf71abd3a50d00b49c1c2c75251cb3c913777d6139cd37dabc093a5e400 |

Upstream repository: [napi-rs/napi-rs](https://github.com/napi-rs/napi-rs).
Each original archive checksum was verified against the cached `.crate` bytes
and the root lock before changing its source to a local path.

## Local patch provenance

Stark's inherited macros clone owned native classes and select enum variants
from JavaScript values. The locked N-API dependencies previously cast any
successfully unwrapped pointer to the requested class, including direct
`ClassInstance` arguments and method receivers. JavaScript `instanceof` cannot
establish that native allocation's Rust type.

The local patch uses [Node-API 8 object type tags](https://nodejs.org/api/n-api.html#napi_type_tag_object).
The runtime registry gives each native Rust `TypeId` a unique process-lifetime
tag; JavaScript cannot copy or inherit it. Every native class creation path tags
the object immediately after successful `napi_wrap`. Tagging must follow a
successful wrap so a failed attempt cannot label an unrelated allocation.
If tagging fails, the runtime detaches that newly installed wrap and runs its
matching finalizer before returning the error. Native reference bookkeeping is
published only after tagging succeeds. A caller-retained object therefore keeps
its prior identity and remains usable after a failed factory or wrapper call.
Generated class references, mutable references, validation, method receivers,
property accessors, direct `ClassInstance`, and `Reference` conversions check
the tag before `napi_unwrap` or a typed pointer cast. The runtime's generic
`TaggedObject` envelope and abort-signal storage have distinct tags so those
helpers cannot reinterpret class allocations either. Generic wrapper checks
retain the existing inner `TypeId` test and support removing and rewrapping
the same generic envelope.

`napi-sys` adds the standard `napi_type_tag` ABI and both Node-API 8 tag functions
to the existing generated symbol loader. Missing dynamic symbols return errors
through its existing failure stubs. Builds without the `napi8` feature fail
closed at native class operations. The Stark workspace already requests
`napi10`; no supported host-version requirement is raised here. Typed native
class conversion requires `'static` storage, as the runtime's class creation
paths already do. No new external dependencies are added.

The source changes relative to the original packages are:

- `napi`: `src/bindgen_runtime/native_type.rs` (new),
  `src/bindgen_runtime/mod.rs`, `callback_info.rs`, `class_accessor.rs`,
  `js_values/class.rs`, `js_values/value_ref.rs`, `js_values/object.rs`,
  `js_values/task.rs`, and `src/env.rs`.
- `napi-derive-backend`: `src/codegen/struct.rs`.
- `napi-sys`: `src/types.rs` and `src/functions.rs`.

The model-free regression fixture and its run instructions are in
[`crates/legacy/bindings/tests/napi_identity`](../../crates/legacy/bindings/tests/napi_identity/README.md).
No upstream release, distribution qualification, full-addon build, or model
execution is implied by this dependency patch.
