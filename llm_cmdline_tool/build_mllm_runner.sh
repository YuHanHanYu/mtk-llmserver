#!/usr/bin/env bash
# Converted from build_mllm_runner.bat for macOS/Linux.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ndk-build -j 8 APP_BUILD_SCRIPT="jni/Android_mllm.mk"

PHONE_PATH="/data/local/tmp/llm_sdk"
adb shell mkdir -p "${PHONE_PATH}"

adb push "libs/arm64-v8a/main_llava" "${PHONE_PATH}"
adb push "libs/arm64-v8a/main_llava_spec_dec" "${PHONE_PATH}"
adb push "libs/arm64-v8a/libmtk_llm.so" "${PHONE_PATH}"
adb push "libs/arm64-v8a/libmtk_mllm.so" "${PHONE_PATH}"
adb push "libs/arm64-v8a/libc++_shared.so" "${PHONE_PATH}"
adb push "libs/arm64-v8a/libyaml-cpp.so" "${PHONE_PATH}"
adb push "jni/prebuilt/libopencv_java4.so" "${PHONE_PATH}"

adb shell "chmod +x ${PHONE_PATH}/main_llava"
