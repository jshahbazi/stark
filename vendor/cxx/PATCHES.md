# CXX security backport

This directory contains the CXX crate family required by Stark's pinned XGrammar dependency. The source comes from [`trymirai/cxx`](https://github.com/trymirai/cxx/tree/aefe85bdd845e391c68e5400e19bdb437502859d) revision `aefe85bdd845e391c68e5400e19bdb437502859d`, whose package version was `1.0.194`. Stark's XGrammar revision remains `04995a307d58f3b2cd015cd5bfd4d7bbeab98f3e`.

The fork's `compile-builtins-at-bridge-stage` feature and `safe_shared_extern` support are preserved. These are required by the existing XGrammar bridge. The imported source includes the runtime, macros, shared syntax, generators, flags, C++ build helper, their Cargo build script, and the focused CXX string tests. Upstream CI, release infrastructure, demos, and unrelated test suites are omitted.

## Security change

[`RUSTSEC-2026-0202`](https://rustsec.org/advisories/RUSTSEC-2026-0202.html) reports that `let_cxx_string!` can destroy uninitialized storage when its value expression or `AsRef<[u8]>` conversion panics. The backport creates the destruction guard only after the C++ string has been initialized.

The patch is the four-commit fix from [upstream pull request 1731](https://github.com/dtolnay/cxx/pull/1731), merged as `5f2e82ea3098e369c090113395ec476b4c80fcf8` and released in upstream `1.0.195` at `c39c127851ca0e45a05e02a6d98d306e6a0c1398`:

- `19fd30d7ea5d19564d2033f48e51588354ccac21`: regression for a panicking value expression.
- `91649e6a6779ca730281e90a0ea883209e105567`: retain `Send + Sync` for an async future holding a CXX string.
- `884ff7ea42612068fed5ce5ae6528d3f2c635581`: move stack storage into `UnsafeCell`.
- `c38adfef01e0fc7ceeec4f557758fb3acd63428c`: initialize storage before creating its destruction guard.

Stark adds a regression for a panicking `AsRef<[u8]>` conversion. The test imports are formatted to Stark's existing Rustfmt configuration; this changes whitespace and ordering only. The existing display, debug, string mutation, I/O, and async trait tests provide ordinary-use controls.

The local crate family is versioned `1.0.195` (`cxx-gen` is `0.7.195`) to identify the patched security boundary. This is a backport onto the recorded fork revision; it does not incorporate every change in the upstream `1.0.195` release. Cargo dependency paths and the root Git-source patch select this local family. The vendored workspace omits the unused demo and FFI test-suite members and their unused root dev-dependency.

## Source verification

Only the seven imported Cargo manifests, `src/cxx_string.rs`, and `tests/cxx_string.rs` differ from the recorded fork. All other imported files retain their original bytes. `LICENSE-MIT` and `LICENSE-APACHE` are preserved.

| File | SHA-256 |
| --- | --- |
| Fork `src/cxx_string.rs` | `a9b63de38db35e9d14edbb84cd2b8ee347ca7063723e57ca9a74341bf3939176` |
| Patched `src/cxx_string.rs` | `395396d42315295f06024584b0e9e04e3caa727e34e22da1c72575a07272cdc1` |
| Patched and formatted `tests/cxx_string.rs` | `5a438ea3272c5a5eb3ad2b4b4d6a8d398f38aad3b2e00100147c01ed91bcd975` |
| `LICENSE-MIT` | `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3` |
| `LICENSE-APACHE` | `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a` |

## Validation

Run the focused tests through Stark's locked workspace so they exercise the same patched family and XGrammar bridge as the engine:

```shell
cargo +1.98.1 check -p stark-engine --no-default-features --features cpu,grammar --locked
cargo +1.98.1 test -p stark-engine --test cxx_string --no-default-features --features cpu,grammar --locked
```

These checks require no models or private evaluation data. The original migration record is [the dependency security migration](../../docs/archive/runs/DEPENDENCY_SECURITY_MIGRATION.md); new validation results go in the pull request that changes this patch.
