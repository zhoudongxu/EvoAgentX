#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
lean --version | grep -F 'version 4.19.0,'
lake build CompactFlow Examples
log=$(mktemp)
trap 'rm -f "$log"' EXIT
lake env lean Check.lean | tee "$log"
if grep -Eq 'sorryAx|declaration uses .sorry.' "$log"; then
  echo 'Unfinished proof found' >&2
  exit 1
fi
# The source contains no project-specific axioms, opaque assumptions or unsafe proof shortcuts.
if grep -En '^[[:space:]]*(axiom |sorry|admit|unsafe |opaque )|native_decide' -- *.lean; then
  echo 'Forbidden proof escape hatch' >&2
  exit 1
fi
