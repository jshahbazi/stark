#!/bin/sh
# Install the standalone Stark bundle. No model files or system tools are changed.
set -eu

fail() { printf 'stark install: %s\n' "$*" >&2; exit 1; }
valid_version() {
    case "$1" in ''|*[!0-9A-Za-z.-]*) return 1 ;; esac
    printf '%s\n' "$1" | LC_ALL=C grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9]+)*$'
}
valid_hash() {
    case "$1" in *[!0-9a-f]*) return 1 ;; esac
    [ "${#1}" -eq 64 ]
}
usage() {
    printf '%s\n' \
        'Usage: ./install.sh [--from-source | --archive FILE | --version VERSION] [--prefix DIR] [--app-dir DIR]' \
        '       ./install.sh --uninstall [--prefix DIR]' \
        '' \
        'Default prefix: ~/.local. Requires Apple silicon and macOS 26.4 or newer.' \
        'Downloads and installs the latest stable GitHub release by default.' \
        '--archive FILE    Install a local bundle; requires FILE.sha256 or --sha256 HASH.' \
        '--version VERSION Download the exact vVERSION GitHub release and its checksum.' \
        '--from-source     Build and install from this source checkout.' \
        '--sha256 HASH     Verify against this SHA-256 instead of an adjacent checksum file.' \
        '--prefix DIR      Install into DIR/bin and DIR/lib/stark, without sudo.' \
        '--app-dir DIR     Install Stark.app into DIR (default: ~/Applications).' \
        '--uninstall       Remove this prefix\047s managed CLI and app; keep all models and app data.'
}

prefix=${HOME:?HOME must be set}/.local
app_dir=$HOME/Applications
archive=
version=
expected_hash=
mode=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --archive|--version|--prefix|--sha256|--app-dir)
            [ "$#" -ge 2 ] && [ -n "$2" ] || fail "$1 requires a value"
            case "$1" in
                --archive) [ -z "$mode" ] || fail 'choose one install mode'; mode=archive; archive=$2 ;;
                --version) [ -z "$mode" ] || fail 'choose one install mode'; mode=download; version=$2 ;;
                --prefix) prefix=$2 ;;
                --app-dir) app_dir=$2 ;;
                --sha256) expected_hash=$2 ;;
            esac
            shift 2 ;;
        --from-source|--uninstall)
            [ -z "$mode" ] || fail 'choose one install mode'
            mode=${1#--}; shift ;;
        *) fail "unknown option: $1 (see --help)" ;;
    esac
done
case "$prefix" in /*) ;; *) fail '--prefix must be an absolute path' ;; esac
case "$app_dir" in /*) ;; *) fail '--app-dir must be an absolute path' ;; esac

source_root=$(CDPATH='' cd -- "$(dirname -- "$0")" 2>/dev/null && pwd -P) || source_root=
[ -n "$mode" ] || mode=download
if [ "$mode" = from-source ]; then
    [ -z "$expected_hash" ] || fail '--sha256 applies only to binary bundles'
    [ -f "$source_root/scripts/install-from-source.sh" ] || fail '--from-source requires a source checkout'
    exec /bin/sh "$source_root/scripts/install-from-source.sh" --prefix "$prefix" --app-dir "$app_dir"
fi
[ "$mode" != uninstall ] || [ -z "$expected_hash" ] || fail '--sha256 cannot be used with --uninstall'

base=$prefix/lib/stark
names='stark stark-score stark-generate stark-convert'
target_for() {
    case "$1" in
        stark|stark-convert) printf '%s\n' '../lib/stark/current/bin/stark' ;;
        *) printf '../lib/stark/current/libexec/%s\n' "$1" ;;
    esac
}
check_links() {
    for name in $names; do
        link=$prefix/bin/$name
        if [ -e "$link" ] || [ -L "$link" ]; then
            [ -L "$link" ] && [ "$(readlink "$link")" = "$(target_for "$name")" ] \
                || fail "refusing to replace another installation: $link"
        fi
    done
}

if [ "$mode" = uninstall ]; then
    [ ! -L "$base" ] || fail 'installation directory must not be a symlink'
    [ -f "$base/INSTALL_FORMAT" ] && [ "$(cat "$base/INSTALL_FORMAT")" = 1 ] \
        || fail "no managed Stark installation at $base"
    check_links
    mkdir "$base/.install-lock" 2>/dev/null || fail 'another installation is in progress'
    trap 'rmdir "$base/.install-lock" 2>/dev/null || :' EXIT
    trap 'exit 1' HUP INT TERM
    if [ -e "$base/managed-app.json" ] || [ -L "$base/managed-app.json" ]; then
        python=$base/current/libexec/python/bin/python3.13
        helper=$base/current/share/stark/install_app.py
        [ -x "$python" ] && [ -f "$helper" ] || fail 'combined installation is incomplete; reinstall before uninstalling'
        "$python" -I -B "$helper" uninstall --prefix "$prefix" \
            || fail 'app removal failed; CLI and models were kept'
    fi
    for name in $names; do
        [ ! -L "$prefix/bin/$name" ] || rm "$prefix/bin/$name"
    done
    rm -rf -- "$base"
    rmdir "$prefix/lib" "$prefix/bin" 2>/dev/null || :
    printf 'Removed Stark from %s. Model files were kept.\n' "$prefix"
    exit 0
fi

[ "$(uname -s)" = Darwin ] || fail 'Stark requires an Apple silicon Mac (macOS 26.4 or newer)'
[ "$(uname -m)" = arm64 ] || fail 'Stark requires arm64; on Apple silicon use a native terminal without Rosetta'
os_version=$(sw_vers -productVersion)
printf '%s\n' "$os_version" | awk -F. '$1 > 26 || ($1 == 26 && $2 >= 4) { ok=1 } END { exit !ok }' \
    || fail "macOS $os_version is unsupported; Stark requires macOS 26.4 or newer"
if [ -n "$version" ]; then
    valid_version "$version" \
        || fail 'invalid release version; use a version such as 0.1.0 or 0.1.0-preview.1, without v'
fi
if [ -n "$expected_hash" ]; then
    valid_hash "$expected_hash" || fail '--sha256 must be 64 lowercase hexadecimal characters'
fi
if [ "$mode" = download ] && [ -z "$version" ]; then
    latest_url=$(curl --proto '=https' --proto-redir '=https' --tlsv1.2 \
        --fail --silent --show-error --location --retry 3 \
        --connect-timeout 15 --max-time 60 --head --output /dev/null \
        --write-out '%{url_effective}' https://github.com/jshahbazi/stark/releases/latest) \
        || fail 'latest stable release unavailable; installation stopped'
    case "$latest_url" in
        https://github.com/jshahbazi/stark/releases/tag/v*)
            version=${latest_url#https://github.com/jshahbazi/stark/releases/tag/v} ;;
        *) fail 'invalid latest release redirect; installation stopped' ;;
    esac
    case "$version" in
        ''|*[!0-9.]*) fail 'invalid latest stable release version; installation stopped' ;;
    esac
    printf '%s\n' "$version" | LC_ALL=C grep -Eq '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$' \
        || fail 'invalid latest stable release version; installation stopped'
fi

[ ! -L "$base" ] || fail 'installation directory must not be a symlink'
if [ -e "$base" ]; then
    [ -f "$base/INSTALL_FORMAT" ] && [ "$(cat "$base/INSTALL_FORMAT")" = 1 ] \
        || fail "refusing to replace an unmanaged directory: $base"
fi
[ ! -L "$base/releases" ] || fail 'release directory must not be a symlink'
if [ -e "$base/current" ] || [ -L "$base/current" ]; then
    [ -L "$base/current" ] || fail 'current must be a managed version symlink'
    current_target=$(readlink "$base/current")
    case "$current_target" in
        releases/*) valid_version "${current_target#releases/}" || fail 'invalid current version link' ;;
        *) fail 'invalid current version link' ;;
    esac
    [ ! -L "$base/$current_target" ] || fail 'version directory must not be a symlink'
fi
check_links
mkdir -p "$prefix/bin" "$base/releases"
mkdir "$base/.install-lock" 2>/dev/null || fail "another install is running; if it was interrupted, remove $base/.install-lock"
printf '1\n' > "$base/INSTALL_FORMAT"
stage=
cleanup() {
    [ -z "$stage" ] || rm -rf -- "$stage"
    rmdir "$base/.install-lock" 2>/dev/null || :
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM
stage=$(mktemp -d "$base/.stage.XXXXXXXX") || fail 'cannot create staging directory'
if [ "$mode" = download ]; then
    asset=stark-$version-aarch64-apple-darwin.tar.gz
    archive=$stage/$asset
    url=https://github.com/jshahbazi/stark/releases/download/v$version/$asset
    printf 'Downloading Stark %s...\n' "$version"
    curl --proto '=https' --proto-redir '=https' --tlsv1.2 --fail --silent --show-error --location --retry 3 \
        --connect-timeout 15 --max-time 600 "$url" -o "$archive" \
        || fail 'release bundle unavailable; check the release, or use ./install.sh --from-source in a source checkout'
    if [ -z "$expected_hash" ]; then
        curl --proto '=https' --proto-redir '=https' --tlsv1.2 --fail --silent --show-error --location --retry 3 \
            --connect-timeout 15 --max-time 60 "$url.sha256" -o "$archive.sha256" \
            || fail 'release checksum unavailable; installation stopped'
    fi
fi
[ -f "$archive" ] || fail 'archive does not exist'
if [ -z "$expected_hash" ]; then
    [ -f "$archive.sha256" ] || fail 'checksum required: provide the adjacent .sha256 file or --sha256 HASH'
    expected_hash=$(awk 'NR == 1 { print $1 } END { if (NR != 1) exit 1 }' "$archive.sha256") \
        || fail 'checksum file must contain exactly one checksum'
    valid_hash "$expected_hash" || fail 'invalid SHA-256 checksum file'
fi
if [ "$mode" = archive ]; then
    # Hash, inspect and execute only the private snapshot of caller-owned bytes.
    # Replacing the external pathname after this copy cannot change the bundle.
    cp -- "$archive" "$stage/archive.tar.gz" || fail 'cannot stage local archive'
    archive=$stage/archive.tar.gz
fi
actual_hash=$(shasum -a 256 -- "$archive" | awk '{print $1}')
[ "$actual_hash" = "$expected_hash" ] || fail 'SHA-256 mismatch; installation stopped'

# Our bundles contain only regular files and directories, under one fixed root.
tar -tzf "$archive" > "$stage/paths" || fail 'invalid archive'
awk '
    $0 != "stark" && $0 !~ /^stark\// { bad=1 }
    /(^|\/)\.\.?(\/|$)/ || /\/\// || /[[:cntrl:]]/ { bad=1 }
    END { exit bad || NR == 0 }
' "$stage/paths" || fail 'unsafe archive path'
tar -tvzf "$archive" > "$stage/types" || fail 'invalid archive listing'
LC_ALL=C awk 'substr($0,1,1) != "-" && substr($0,1,1) != "d" { bad=1 } END { exit bad }' "$stage/types" \
    || fail 'archive links and special files are not supported'
tar -xzf "$archive" -C "$stage" --no-same-owner --no-same-permissions || fail 'cannot unpack archive'
bundle=$stage/stark
[ -f "$bundle/share/stark/INSTALL_FORMAT" ] && [ "$(cat "$bundle/share/stark/INSTALL_FORMAT")" = 1 ] \
    || fail 'unsupported bundle format'
bundle_version=$(cat "$bundle/share/stark/VERSION")
valid_version "$bundle_version" \
    || fail 'invalid bundle version'
[ -z "$version" ] || [ "$bundle_version" = "$version" ] || fail 'bundle version does not match requested release'
for file in bin/stark libexec/stark-score libexec/stark-generate libexec/python/bin/python3.13; do
    [ -x "$bundle/$file" ] || fail "bundle component missing: $file"
done
# Doctor exercises the relocated Python environment and the workers without a model.
"$bundle/bin/stark" doctor || fail 'bundle self-check failed; previous installation was kept'
has_app=false
if [ -e "$bundle/app/Stark.app" ]; then
    [ -f "$bundle/share/stark/install_app.py" ] || fail 'combined bundle is missing its app installer'
    has_app=true
elif [ -e "$base/managed-app.json" ] || [ -L "$base/managed-app.json" ]; then
    fail 'CLI-only bundle cannot replace a combined installation; uninstall first'
fi
release=$base/releases/$bundle_version
[ ! -L "$release" ] || fail 'version directory must not be a symlink'
if [ -e "$release" ]; then
    [ -f "$release/.archive-sha256" ] && [ "$(cat "$release/.archive-sha256")" = "$actual_hash" ] \
        || fail "version $bundle_version is already installed with different contents; use a new version"
else
    printf '%s\n' "$actual_hash" > "$bundle/.archive-sha256"
    mv "$bundle" "$release"
fi
if [ "$has_app" = true ]; then
    "$release/libexec/python/bin/python3.13" -I -B "$release/share/stark/install_app.py" \
        install --bundle "$release" --prefix "$prefix" --app-dir "$app_dir" \
        || fail 'combined activation failed; previous installation was kept'
else
    ln -s "releases/$bundle_version" "$stage/current"
    # BSD mv -h replaces a symlink to a directory instead of moving into its target.
    /bin/mv -fh "$stage/current" "$base/current"
    for name in $names; do
        if [ ! -L "$prefix/bin/$name" ]; then
            ln -s "$(target_for "$name")" "$prefix/bin/$name"
        fi
    done
fi
printf 'Installed Stark %s in %s\n' "$bundle_version" "$prefix"
case ":${PATH:-}:" in
    *":$prefix/bin:"*) ;;
    *) printf 'Add %s/bin to PATH in your shell profile.\n' "$prefix" ;;
esac
printf 'Run: %s/bin/stark doctor\n' "$prefix"
if [ "$has_app" = true ]; then
    printf 'App: %s/Stark.app (open it from Finder).\n' "$app_dir"
fi
