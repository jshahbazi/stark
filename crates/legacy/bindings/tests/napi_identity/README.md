# Native class identity regression fixture

This standalone, model-free addon exercises the project's actual `bindings`
owned-class and enum macros and the locked N-API runtime. Its dependency lock is
the corresponding subset of the root lock; it adds no dependency versions.
The fixture requires Rust 1.98.1 and a Node.js host supporting Node-API 10,
matching the workspace's existing `napi10` feature.

Run with a disposable build directory:

```sh
uv run --no-project python crates/legacy/bindings/tests/napi_identity/run.py --target-dir /tmp/stark-napi-identity-target
```

The runner builds offline with `--locked`, keeps compiled artifacts in the
requested directory, and runs both the narrow wrong-class probe and the broader
regressions. `probe.cjs` covers five typed argument conversions and a
prototype-forged enum using a synthetic class with the same field layout.
`test.cjs` covers native constructors, factories, direct `new_instance` classes,
owned/shared/mutable arguments, `ClassInstance`, `Reference`, method receivers,
getters/setters, prototype mutations, `Symbol.hasInstance`, generic object
wrappers and reused abort signals. Ordinary inputs must continue to work and
foreign native classes must raise JavaScript errors before field access.
Synchronous and asynchronous factory rejection must also detach newly installed
storage, so caller-retained objects remain reusable under their original tag.

`external.cjs` covers modern and deprecated external creation and extraction,
shared/mutable arguments, strong references, generic transport, zero/positive/
negative memory hints, and single finalization after garbage collection. Null,
one-byte and forged-header foreign allocations and wrong Rust types must fail
before typed payload access. The runner exposes Node's garbage collector for
this synthetic lifetime check. A focused pre-fix null-external probe aborted
its disposable Node subprocess while its ordinary external control passed;
this is evidence about the shared runtime API, not deployed Stark exposure.

For pre-patch comparison, build the same fixture with Cargo configuration
overriding its three `patch.crates-io` paths to pristine sources for exactly
`napi` 3.10.3, `napi-derive-backend` 5.1.1 and `napi-sys` 3.2.2 in a separate
target directory. Run `node probe.cjs /absolute/path/to/addon.node` without
`fixed`: all six cases demonstrate baseline acceptance of the wrong class.
Never run the broader hostile-input suite against the unpatched runtime:
its deliberately incompatible layouts may crash that runtime.

This fixture verifies the shared conversion boundary. It does not build the
complete inherited Stark addon, load models, qualify a binding distribution,
or demonstrate exploitation against a deployed application.
