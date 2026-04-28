#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [DIR] [--delete]

Lists files named '._*' under DIR (default: current directory).
Pass --delete as the second argument to actually remove the files.
EOF
  exit 1
}

if [[ "${1:-}" == "" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
fi

# Accept either: ./script.sh DIR --delete  OR  ./script.sh --delete (uses .)
DIR="."
DELETE=false
if [[ "${1:-}" == "--delete" ]]; then
  DIR="."
  DELETE=true
else
  DIR="$1"
  if [[ "${2:-}" == "--delete" ]]; then
    DELETE=true
  fi
fi

echo "Searching for files named '._*' under: $DIR"

if ! command -v find >/dev/null 2>&1; then
  echo "Error: find command not found." >&2
  exit 2
fi

if [[ "$DELETE" == true ]]; then
  echo "Deleting matching files (this cannot be undone)."
  find "$DIR" -type f -name '._*' -print -exec rm -v {} + || true
  echo "Done."
else
  echo "Dry-run: the following files would be removed. Re-run with --delete to remove them."
  find "$DIR" -type f -name '._*' -print || true
fi
