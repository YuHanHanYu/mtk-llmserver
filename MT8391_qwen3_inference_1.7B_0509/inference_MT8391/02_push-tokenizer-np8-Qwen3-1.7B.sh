#!/usr/bin/env bash
# Converted from 02_push-tokenizer-np8-Qwen3-4B.bat for macOS/Linux.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PHONE_PATH="/data/local/tmp/llm_sdk"
adb shell "mkdir -p ${PHONE_PATH}"

MODEL_DIR="Qwen3-1.7B_0509"
adb shell "mkdir -p ${PHONE_PATH}/${MODEL_DIR}"

adb push "../dla/Qwen3-1.7B/tokenizer" "${PHONE_PATH}/${MODEL_DIR}/"
adb push "../dla/Qwen3-1.7B/embedding_int16.bin" "${PHONE_PATH}/${MODEL_DIR}/tokenizer/"
