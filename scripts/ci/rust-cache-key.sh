#!/usr/bin/env bash
# Cache only compatible native outputs; let Cargo fingerprint Rust source changes.
set -euo pipefail

cache_scope=${1:?missing cache scope}
cache_metal=${2:-false}
case "$cache_scope" in
  ''|*[!a-z0-9-]*) echo 'Invalid Rust cache scope.' >&2; exit 1 ;;
esac
case "$cache_metal" in
  true|false) ;;
  *) echo 'Invalid Metal cache setting.' >&2; exit 1 ;;
esac

hash_stream() { shasum -a 256 | cut -d ' ' -f 1; }

hash_files() {
  local cache_regular_files=()
  while IFS= read -r -d '' cache_file; do
    printf '%s\0' "$cache_file"
    if test -L "$cache_file"; then
      readlink "$cache_file" || return 1
    else
      cache_regular_files+=("$cache_file")
    fi
  done
  if test "${#cache_regular_files[@]}" -gt 0; then
    printf '%s\0' "${cache_regular_files[@]}" | xargs -0 git hash-object -- || return 1
  fi
}

cache_identity=$({
  set -e
  printf '%s\0' "${RUNNER_OS:?}" "${RUNNER_ARCH:?}" "${ImageOS:-}" "${ImageVersion:-}"
  rustc -vV
  cargo --version
  command -v cc
  command -v c++
  cc --version
  c++ --version
  # Explicit compiler settings also invalidate compiled dependencies.
  printf '%s\0' "${RUSTFLAGS:-}" "${CARGO_ENCODED_RUSTFLAGS:-}" "${CARGO_INCREMENTAL:-}"
  printf '%s\0' "${CC:-}" "${CXX:-}" "${CFLAGS:-}" "${CXXFLAGS:-}"
  printf '%s\0' "${RUSTDOCFLAGS:-}" "${AR:-}" "${RANLIB:-}" "${LD:-}" "${LDFLAGS:-}"
  printf '%s\0' "${BUILD_ALWAYS:-}" "${BUILD_CLEAN:-}"
  printf '%s\0' "${SDKROOT:-}" "${MACOSX_DEPLOYMENT_TARGET:-}" "${DEVELOPER_DIR:-}"
  while IFS= read -r cache_variable; do
    case "$cache_variable" in
      (CARGO_PROFILE_*|CARGO_BUILD_*|CARGO_TARGET_*)
        printf '%s\0' "$cache_variable" "${!cache_variable}" ;;
    esac
  done < <(compgen -e | LC_ALL=C sort)
  if test "$RUNNER_OS" = macOS; then
    xcodebuild -version
    xcrun --sdk macosx --show-sdk-version
    xcrun --sdk macosx --show-sdk-build-version
    xcrun --find clang
    xcrun clang --version
    if test "$cache_metal" = true; then
      # Metal's cryptex mount has a random suffix on every hosted runner.
      # Keep compiler build/target identity, excluding its temporary location.
      xcrun -sdk macosx metal --version | sed '/^[[:space:]]*InstalledDir:/d'
    fi
  fi
} | hash_stream)

# Cover locks, profiles, build scripts, code generators, native inputs and cache
# logic. No fallback crosses this boundary, even if Cargo misses a native input.
cache_inputs=$({
  set -e
  git ls-files -z -- Cargo.toml Cargo.lock '**/Cargo.toml' '**/Cargo.lock' \
    rust-toolchain.toml .cargo .github/workflows/ci.yml \
    .github/actions/rust-cache scripts/ci/rust-cache-key.sh \
    '**/build.rs' '**/build/**' crates/legacy/hanashi/configs | hash_files
  while IFS= read -r -d '' cache_file; do
    case "$cache_file" in
      (*.rs) ;;
      (*) printf '%s\0' "$cache_file" ;;
    esac
  done < <(git ls-files -z -- crates/stark-engine vendor/cxx) | hash_files
} | hash_stream)

cache_source=$(git ls-files -z -- '*.rs' | hash_files | hash_stream)
cache_prefix="stark-rust-v1-$cache_scope-$cache_identity-$cache_inputs-"
printf 'restore-prefix=%s\nkey=%s%s\n' "$cache_prefix" "$cache_prefix" "$cache_source"
