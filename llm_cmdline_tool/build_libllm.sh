#!/usr/bin/env bash
# Converted from build_libllm.bat for macOS/Linux.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXTRA_FLAGS=""
if [[ "${1:-}" == "-usdk" ]]; then
  EXTRA_FLAGS="-DUSE_USDK_BACKEND"
  shift
fi

LIB_LLM_SRC_PATH="jni/prebuilt/src"

ndk-build -j 8 NDK_PROJECT_PATH=. APP_BUILD_SCRIPT="${LIB_LLM_SRC_PATH}/Android.mk" NDK_APPLICATION_MK="${LIB_LLM_SRC_PATH}/Application.mk" EXTRA_GLOBAL_CPPFLAGS="${EXTRA_FLAGS}"

mkdir -p "jni/prebuilt/include"
cp -f "libs/arm64-v8a/libmtk_llm.so" "jni/prebuilt/"
cp -f "${LIB_LLM_SRC_PATH}/mtk_llm_types.h" "jni/prebuilt/include/"
cp -f "${LIB_LLM_SRC_PATH}/mtk_llm_options.h" "jni/prebuilt/include/"
cp -f "${LIB_LLM_SRC_PATH}/mtk_llm.h" "jni/prebuilt/include/"
cp -f "${LIB_LLM_SRC_PATH}/medusa_config.h" "jni/prebuilt/include/"
