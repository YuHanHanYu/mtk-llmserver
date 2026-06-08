#!/usr/bin/env bash
# Converted from run_llama3.2_1b.bat for macOS/Linux.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PHONE_PATH="/data/local/tmp/llm_sdk"
CONFIG_FILE="config_llama3.2_1b_instruct.yaml"
PROMPT_FILE="sample_prompt_introduce.txt"
MAX_RESPONSE="128"
PREFORMATTER="Llama3NoInput"

adb push "${CONFIG_FILE}" "${PHONE_PATH}"
adb push "${PROMPT_FILE}" "${PHONE_PATH}"

adb shell "chmod +x ${PHONE_PATH}/main"

adb shell "cd ${PHONE_PATH}; LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$PWD ./main ${CONFIG_FILE} -i ${PROMPT_FILE} -m ${MAX_RESPONSE} --preformatter ${PREFORMATTER} --one-prompt-per-line"
