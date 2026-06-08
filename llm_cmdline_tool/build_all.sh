#!/usr/bin/env bash
# Converted from build_all.bat for macOS/Linux.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

./build_clean.sh
./build_libcommon.sh
./build_libtokenizer.sh
./build_libllm.sh
./build_runner.sh
