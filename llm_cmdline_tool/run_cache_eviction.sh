#!/usr/bin/env bash
# Converted from run_cache_eviction.bat for macOS/Linux.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PHONE_PATH="/data/local/tmp/llm_sdk"
CONFIG_FILE="config_qwen2.5_3b_cache_eviction.yaml"
INPUT_PROMPT="longbench_vcsum.txt"
MAX_RESPONSE="512"
PREFORMATTER="QwenNoInput"

adb push "${CONFIG_FILE}" "${PHONE_PATH}"
adb push "${INPUT_PROMPT}" "${PHONE_PATH}"

adb shell "chmod +x ${PHONE_PATH}/main"

adb shell "cd ${PHONE_PATH}; LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$PWD ./main ${CONFIG_FILE} -i ${INPUT_PROMPT} --preformatter ${PREFORMATTER} -m ${MAX_RESPONSE} --one-prompt-per-line"
