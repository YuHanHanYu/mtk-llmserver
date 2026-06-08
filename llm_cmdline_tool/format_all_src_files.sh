#!/usr/bin/env bash
# Converted from format_all_src_files.bat for macOS/Linux.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TEMP_FILELIST="$(mktemp)"
trap 'rm -f "${TEMP_FILELIST}"' EXIT

find . -type f \( -name '*.cpp' -o -name '*.h' \) \
  ! -path '*/third_party/*' \
  ! -path '*/backend/api/*' \
  -print > "${TEMP_FILELIST}"

clang-format -i --verbose --files="${TEMP_FILELIST}"
