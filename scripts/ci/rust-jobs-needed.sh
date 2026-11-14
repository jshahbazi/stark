#!/usr/bin/env bash
# Decide whether a change can affect the macOS Rust jobs in CI.
#
# Prints `rust=false` only when every changed path is known not to feed those
# jobs. Unknown paths, a missing base and any Git failure print `rust=true`.
#
#   rust-jobs-needed.sh              compare HEAD with the event's base
#   rust-jobs-needed.sh --classify   classify NUL-separated paths on stdin
set -euo pipefail

# Paths the scoring, cxx and engine-unit jobs never read. Anything else,
# including a new top-level directory, keeps the jobs running. Rust sources
# include data files from tools/ (fixtures, memory profiles), so only Python
# and packaging files are listed there. Remove a path from this list before a
# Rust job starts reading it; scripts/ci/test_rust_jobs_needed.py checks the
# files Rust sources include today.
skippable() {
  case "$1" in
    docs/*|website/*|apps/*|provenance/*) return 0 ;;
    install.sh|LICENSE|LICENSE.*) return 0 ;;
    scripts/ci/*|scripts/test_generation_process.py) return 1 ;;
    scripts/*) return 0 ;;
    tools/*.py|tools/*.md|tools/*uv.lock|tools/*pyproject.toml) return 0 ;;
    */*) return 1 ;;
    *.md) return 0 ;;
  esac
  return 1
}

classify() {
  local path
  while IFS= read -r -d '' path; do
    if ! skippable "$path"; then
      printf 'Rust jobs needed: %s\n' "$path" >&2
      echo 'rust=true'
      return 0
    fi
  done
  echo 'No changed path feeds the Rust jobs.' >&2
  echo 'rust=false'
}

if [ "${1:-}" = --classify ]; then
  classify
  exit 0
fi

run_everything() {
  printf '%s; running the Rust jobs.\n' "$1" >&2
  echo 'rust=true'
  exit 0
}

base=''
case "${EVENT_NAME:-}" in
  pull_request)
    # The checked-out merge commit's first parent is the base branch tip, so
    # this is exactly what the pull request changes on the current base.
    if git rev-parse --verify --quiet 'HEAD^2' >/dev/null; then
      base=$(git rev-parse 'HEAD^1')
    fi
    ;;
  push)
    # BASE_SHA is the last commit whose CI run on this branch succeeded, not
    # the previous push: a cancelled run must not hide its Rust changes from
    # the run that replaces it.
    case "${BASE_SHA:-}" in
      *[!0-9a-f]*|'') ;;
      *)
        if [ "${#BASE_SHA}" -eq 40 ] \
          && git fetch --quiet --no-tags --depth=1 origin "$BASE_SHA"; then
          base=$BASE_SHA
        fi
        ;;
    esac
    ;;
esac
test -n "$base" || run_everything 'No comparable base'

changed=$(mktemp)
trap 'rm -f "$changed"' EXIT
# --no-renames lists both sides of a move, so moving a file out of a Rust
# input directory still counts.
git diff --name-only -z --no-renames "$base" HEAD > "$changed" \
  || run_everything 'Could not list changed paths'
classify < "$changed"
