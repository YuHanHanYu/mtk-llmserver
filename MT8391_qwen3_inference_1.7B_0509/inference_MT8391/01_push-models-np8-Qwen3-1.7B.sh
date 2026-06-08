#!/usr/bin/env bash
# Converted from 01_push-models-np8-Qwen3-4B.bat for macOS/Linux.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PHONE_PATH="/data/local/tmp/llm_sdk"
adb shell "mkdir -p ${PHONE_PATH}"

MODEL_DIR="Qwen3-1.7B_0509"
adb shell "mkdir -p ${PHONE_PATH}/${MODEL_DIR}"

MODEL_NAME="2048c"

adb push "../dla/Qwen3-1.7B/${MODEL_NAME}" "${PHONE_PATH}/${MODEL_DIR}/${MODEL_NAME}/"
