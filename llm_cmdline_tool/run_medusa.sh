#!/usr/bin/env bash
# Converted from run_medusa.bat for macOS/Linux.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PHONE_PATH="/data/local/tmp/llm_sdk"
CONFIG_FILE="config_medusa_vicuna_7b_3heads_8t.yaml"
INPUT_PROMPT="mt_bench_1st_turn.txt"
MAX_RESPONSE="512"
PREFORMATTER="VicunaNoInput"
TEMPERATURE="0.7"

adb push "${CONFIG_FILE}" "${PHONE_PATH}"
adb push "${INPUT_PROMPT}" "${PHONE_PATH}"

adb shell "chmod +x ${PHONE_PATH}/main_medusa"

adb shell "cd ${PHONE_PATH}; LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$PWD ./main_medusa ${CONFIG_FILE} -i ${INPUT_PROMPT} -m ${MAX_RESPONSE} --preformatter ${PREFORMATTER} --one-prompt-per-line --temperature ${TEMPERATURE}"
