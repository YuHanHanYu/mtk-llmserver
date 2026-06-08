#!/usr/bin/env bash
# Converted from run_llava_spec_dec.bat for macOS/Linux.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PHONE_PATH="/data/local/tmp/llm_sdk"
TARGET_CONFIG_FILE="config_llava_v1p5_7b_spec_dec_target.yaml"
DRAFT_CONFIG_FILE="config_vicuna_160m_spec_dec_draft.yaml"
INPUT_PROMPT="llava_in_the_wild_questions.txt"
INPUT_IMAGES="llava_in_the_wild_images.txt"
MAX_RESPONSE="512"
PREFORMATTER="VicunaNoInput"
TARGET_TEMPERATURE="0.0"
DRAFT_TEMPERATURE="0.0"

adb push "${INPUT_PROMPT}" "${PHONE_PATH}"
adb push "${INPUT_IMAGES}" "${PHONE_PATH}"
adb push "${TARGET_CONFIG_FILE}" "${PHONE_PATH}"
adb push "${DRAFT_CONFIG_FILE}" "${PHONE_PATH}"

adb shell "chmod +x ${PHONE_PATH}/main_llava_spec_dec"

INFER_TYPE="0"
DRAFT_LENGTH="5"
adb shell "cd ${PHONE_PATH}; LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$PWD ./main_llava_spec_dec ${TARGET_CONFIG_FILE} --infer-type ${INFER_TYPE} --draft ${DRAFT_CONFIG_FILE} --draft-len ${DRAFT_LENGTH} -i ${INPUT_PROMPT} -im ${INPUT_IMAGES} -m ${MAX_RESPONSE} --preformatter ${PREFORMATTER} --one-prompt-per-line --target-temperature ${TARGET_TEMPERATURE} --draft-temperature ${DRAFT_TEMPERATURE}"
