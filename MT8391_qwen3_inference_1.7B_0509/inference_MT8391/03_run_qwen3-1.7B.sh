#!/usr/bin/env bash
# Converted from 03_run_qwen3-4B.bat for macOS/Linux.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PHONE_PATH="/data/local/tmp/llm_sdk"
CONFIG_FILE="config_np8-qwen3-1.7B.yaml"
INPUT_PROMPT="sample_prompt-q1.txt"
MAX_RESPONSE="4096"
PREFORMATTER="Qwen3NoInputNoThink"

adb push "${CONFIG_FILE}" "${PHONE_PATH}"
adb push "${INPUT_PROMPT}" "${PHONE_PATH}"

adb shell "chmod +x ${PHONE_PATH}/main"

adb shell "cd ${PHONE_PATH}; LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$PWD ./main ${CONFIG_FILE} -i ${INPUT_PROMPT} --preformatter ${PREFORMATTER} -m ${MAX_RESPONSE} --one-prompt-per-line"
