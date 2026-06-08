#!/usr/bin/env bash
# Converted from build_libcommon.bat for macOS/Linux.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LIB_LLM_SRC_PATH="jni/prebuilt/src"

ndk-build -j 8 NDK_PROJECT_PATH=. APP_BUILD_SCRIPT="${LIB_LLM_SRC_PATH}/common/Android.mk" NDK_APPLICATION_MK="${LIB_LLM_SRC_PATH}/Application.mk"

mkdir -p "jni/prebuilt"
cp -f "libs/arm64-v8a/libcommon.so" "jni/prebuilt/"

mkdir -p "jni/prebuilt/include/common"
cp -f "${LIB_LLM_SRC_PATH}/common/"*.h "jni/prebuilt/include/common/"
