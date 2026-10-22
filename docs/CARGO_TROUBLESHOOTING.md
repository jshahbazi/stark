# Locked Cargo dependency resolution

Use Rust 1.98.1 and keep the checked-in lockfiles. Check the normal registry
resolution before changing a package version:

```sh
cargo +1.98.1 metadata --locked --format-version=1 > /dev/null
```

## Stale sparse index entry

A local Cargo index entry can omit a published release even after Cargo prints
`Updating crates.io index`. This occurred with `rustls-webpki` 0.103.16: the local
entry stopped at 0.103.15, while both the crates.io index and package API listed
0.103.16 with the checksum already recorded in `Cargo.lock`.

Version 0.103.16 contains the certificate distrust fix for
[RUSTSEC-2026-0382](https://rustsec.org/advisories/RUSTSEC-2026-0382). Preserve that
version when repairing the cache.

Compare the published record with the locked version and checksum:

```sh
curl --fail --silent --show-error -H 'Cache-Control: no-cache' \
  https://index.crates.io/ru/st/rustls-webpki \
  | uv run --no-project --python 3.13.13 python -c 'import json,sys; print(*[json.dumps({k:r.get(k) for k in ("vers","cksum","yanked")}) for r in map(json.loads,sys.stdin) if r["vers"] == "0.103.16"],sep="\n")'
```

If the official record matches the lock, back up the one stale local entry and
let Cargo fetch it again. The following path is Cargo's default crates.io sparse
cache; inspect the actual index directory first if this installation uses a
different registry or `CARGO_HOME`.

```sh
stark_cargo_root="${CARGO_HOME:-$HOME/.cargo}"
stark_index_entry="$stark_cargo_root/registry/index/index.crates.io-1949cf8c6b5b557f/.cache/ru/st/rustls-webpki"
if [ -f "$stark_index_entry" ] && [ ! -L "$stark_index_entry" ]; then
  stark_index_backup="$(mktemp -d "${TMPDIR:-/tmp}/stark-cargo-index.XXXXXX")" &&
    mv "$stark_index_entry" "$stark_index_backup/rustls-webpki" &&
    cargo +1.98.1 metadata --locked --format-version=1 > /dev/null &&
    git diff --exit-code -- Cargo.lock
else
  printf '%s\n' 'Inspect the registry cache path before moving an entry.' >&2
fi
```

The repair should leave `Cargo.lock`, TLS features, registry checksums and
certificate verification unchanged. If resolution still fails, retain the
backed-up entry and inspect the new error before changing other cache records.
