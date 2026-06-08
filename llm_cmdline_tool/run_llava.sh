#!/usr/bin/env bash
# Converted from run_llava.bat for macOS/Linux.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PHONE_PATH="/data/local/tmp/llm_sdk"
CONFIG_FILE="config_llava.yaml"

adb push "${CONFIG_FILE}" "${PHONE_PATH}"

adb shell "chmod +x ${PHONE_PATH}/main_llava"

adb shell "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:${PHONE_PATH} ${PHONE_PATH}/main_llava ${PHONE_PATH}/${CONFIG_FILE}"
