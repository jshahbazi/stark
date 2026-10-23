#!/bin/sh
# A source checkout uses the same relocatable bundle as a binary installation.
set -eu
root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)
fail() { printf 'stark source install: %s\n' "$*" >&2; exit 1; }
prefix=${HOME:?HOME must be set}/.local
app_dir=$HOME/Applications
while [ "$#" -gt 0 ]; do
    [ "$#" -ge 2 ] || fail 'source installation requires --prefix DIR or --app-dir DIR'
    case "$1" in
        --prefix) prefix=$2 ;;
        --app-dir) app_dir=$2 ;;
        *) fail 'source installation accepts only --prefix DIR and --app-dir DIR' ;;
    esac
    shift 2
done
case "$prefix" in /*) ;; *) fail '--prefix must be an absolute path' ;; esac
case "$app_dir" in /*) ;; *) fail '--app-dir must be an absolute path' ;; esac
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || fail 'use an Apple silicon Mac in a native arm64 terminal'
sw_vers -productVersion | awk -F. '$1 > 26 || ($1 == 26 && $2 >= 4) { ok=1 } END { exit !ok }' \
    || fail 'macOS 26.4 or newer is required'
# Discover the usual tool locations without changing the caller\047s shell profile.
PATH=${PATH:-/usr/bin:/bin}:/opt/homebrew/bin:${HOME}/.cargo/bin
export PATH
command -v uv >/dev/null 2>&1 || fail 'install uv first: brew install uv (required version: 0.12.24)'
command -v rustup >/dev/null 2>&1 || fail 'install Rust with rustup: https://rustup.rs'
[ "$(uv --version | awk '{print $2}')" = 0.12.24 ] || fail 'this source build requires uv 0.12.24 to preserve build provenance'
xcrun -sdk macosx metal --version >/dev/null 2>&1 \
    || fail 'install Xcode and its Metal Toolchain, then select Xcode in Settings > Locations > Command Line Tools'
rustup toolchain install 1.98.1 --profile minimal --no-self-update
revision=$(git -C "$root" rev-parse --short=12 HEAD)
# Include all owned build inputs, including new source files, in development identity.
suffix=$(uv run --no-project --python 3.13.13 python -c '
import hashlib, json, sys
sys.path.insert(0, sys.argv[1] + "/scripts")
from build_distribution import clean_environment, source_provenance
identity = source_provenance(clean_environment(), allow_dirty=True)
print(hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12])
' "$root")
# Upstream Python installation metadata can vary across builds. Each source
# build gets its own immutable version; reinstalling an existing archive keeps
# its exact version and checksum instead.
build_time=$(date -u +%Y%m%d%H%M%S)
version=0.1.0-dev.$revision.$suffix.$build_time
output=$root/artifacts/distributions
printf 'Building Stark %s. This can take several minutes.\n' "$version"
uv run --no-project --python 3.13.13 python "$root/scripts/build_distribution.py" \
    --version "$version" --output-dir "$output" --allow-dirty
exec /bin/sh "$root/install.sh" --archive "$output/stark-$version-aarch64-apple-darwin.tar.gz" --prefix "$prefix" --app-dir "$app_dir"
