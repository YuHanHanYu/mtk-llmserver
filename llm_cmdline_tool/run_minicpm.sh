#!/usr/bin/env bash
# Converted from run_minicpm.bat for macOS/Linux.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PHONE_PATH="/data/local/tmp/llm_sdk"
CONFIG_FILE="config_minicpm_2b.yaml"
PROMPT_FILE="sample_prompt.txt"
PREFORMATTER="MinicpmNoInput"
MAX_RESPONSE="100"

adb push "${CONFIG_FILE}" "${PHONE_PATH}"
adb push "${PROMPT_FILE}" "${PHONE_PATH}"

adb shell "chmod +x ${PHONE_PATH}/main"

adb shell "cd ${PHONE_PATH}; LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$PWD ./main ${CONFIG_FILE} -i ${PROMPT_FILE} -m ${MAX_RESPONSE} --preformatter ${PREFORMATTER}"
