#!/usr/bin/env bash
set -euo pipefail

# Copy subject folders listed in a split file from source root to destination root.
# Default behavior matches the common workflow in this repo.

SPLIT_FILE="data/split_val.txt"
SRC_ROOT="processed"
DST_ROOT="output_gt"
DRY_RUN=0
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/copy_subjects_from_split.sh [options]

Options:
  --split PATH       Split file containing subject names (default: data/split_val.txt)
  --src PATH         Source root containing subject folders (default: processed)
  --dst PATH         Destination root (default: output_gt)
  --overwrite        Replace destination subject folder if it already exists
  --dry-run          Print planned operations without copying
  -h, --help         Show this help

Examples:
  ./scripts/copy_subjects_from_split.sh
  ./scripts/copy_subjects_from_split.sh --split data/split_train.txt --src processed --dst output_train_gt
  ./scripts/copy_subjects_from_split.sh --overwrite
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split)
      SPLIT_FILE="$2"
      shift 2
      ;;
    --src)
      SRC_ROOT="$2"
      shift 2
      ;;
    --dst)
      DST_ROOT="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -f "$SPLIT_FILE" ]]; then
  echo "Split file not found: $SPLIT_FILE" >&2
  exit 1
fi

if [[ ! -d "$SRC_ROOT" ]]; then
  echo "Source root not found: $SRC_ROOT" >&2
  exit 1
fi

mkdir -p "$DST_ROOT"

expected=0
copied=0
skipped_existing=0
missing_source=0

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  # Normalize CRLF and trim surrounding whitespace.
  subject="$(printf '%s' "$raw_line" | tr -d '\r' | xargs)"
  [[ -z "$subject" ]] && continue
  expected=$((expected + 1))

  src_dir="$SRC_ROOT/$subject"
  dst_dir="$DST_ROOT/$subject"

  if [[ ! -d "$src_dir" ]]; then
    echo "MISSING_SOURCE: $subject"
    missing_source=$((missing_source + 1))
    continue
  fi

  if [[ -d "$dst_dir" && "$OVERWRITE" -eq 0 ]]; then
    echo "SKIP_EXISTS: $subject"
    skipped_existing=$((skipped_existing + 1))
    continue
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "COPY: $src_dir -> $DST_ROOT/"
    copied=$((copied + 1))
    continue
  fi

  if [[ -d "$dst_dir" && "$OVERWRITE" -eq 1 ]]; then
    rm -rf "$dst_dir"
  fi
  cp -a "$src_dir" "$DST_ROOT/"
  copied=$((copied + 1))
done < "$SPLIT_FILE"

echo "--- Summary ---"
echo "split_file: $SPLIT_FILE"
echo "source_root: $SRC_ROOT"
echo "dest_root:   $DST_ROOT"
echo "subjects_in_split: $expected"
echo "copied: $copied"
echo "skipped_existing: $skipped_existing"
echo "missing_source: $missing_source"
