# Temporary rustls exception for non-production tests

On 2026-10-15, the maintainer explicitly accepted using rustls 0.23.45 for non-production checkpoint testing. Aikido Device Protection currently blocks installation of the newly published 0.23.46 release. Both existing Cargo lockfiles retain 0.23.45; this change accepts the known findings without changing dependency resolution.

The exception expires at **2026-11-14 00:00 UTC**. It covers only crates.io `rustls@0.23.45`, checksum `0d41d731c7d2f962d1ccc364cec258de3c0e93b38c2fb3ba97ac74513048d634`, in `Cargo.lock` and `crates/legacy/download-manager/examples/wasm/Cargo.lock`.

| Advisory | Accepted risk |
| --- | --- |
| [RUSTSEC-2026-0346](https://rustsec.org/advisories/RUSTSEC-2026-0346.html) | Custom session-store debug formatting can expose TLS session secrets. |
| [RUSTSEC-2026-0347](https://rustsec.org/advisories/RUSTSEC-2026-0347.html) | Rejected ECH can send buffered application data to the authenticated outer server. |
| [RUSTSEC-2026-0348](https://rustsec.org/advisories/RUSTSEC-2026-0348.html) | Brotli certificate decompression can cause excessive allocation before authentication when the optional feature is enabled. |
| [RUSTSEC-2026-0349](https://rustsec.org/advisories/RUSTSEC-2026-0349.html) | Unauthenticated key-share hints can downgrade later connections to classical key exchange. |

The upstream patched release is 0.23.46. This exception does not establish that every affected path is unreachable. It is limited to the maintainer's non-production testing decision and must be removed before production use.

`scripts/ci/dependency-exceptions.toml` records the scope, rationale, and expiry. Both dependency jobs run `check_dependency_exceptions.py` before scanning. It rejects expiry, lockfile identity changes, and drift between the manifest and cargo-deny's ignored advisory IDs. The OSV report gate applies the same policy only to matching package, version, lockfile, and advisory IDs, and prints accepted findings as warnings. All other advisories remain blocking.

When a patched release is permitted, update both lockfiles and remove these exceptions, cargo-deny ignores, and their dedicated preflight wiring together. Record a new dependency scan. Existing source-bound native validation evidence remains unchanged.
