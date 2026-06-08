# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build commands

- Build all Android/arm64 targets from the NDK project:
  ```sh
  cd /Users/hanyu/Documents/sbc/mtk_v0606/llm_cmdline_tool
  ndk-build -j8
  ```
- Main llmserver outputs after build:
  ```text
  llm_cmdline_tool/libs/arm64-v8a/libllmserver.so
  llm_cmdline_tool/libs/arm64-v8a/main_server
  ```
- There is no detected unit-test runner or lint command in this repository. Validation is done by building, deploying to the board, starting Magnus or `main_server`, and exercising HTTP endpoints with `curl`.

## Deploy and run

- Deploy updated llmserver library into the Magnus release:
  ```sh
  cd /Users/hanyu/Documents/sbc/mtk_v0606/llm_cmdline_tool
  adb push libs/arm64-v8a/libllmserver.so \
    /data/local/tmp/magnus_llmserver_release_android_aarch64_mtk_npu_v0.0.6/libs/libllmserver.so
  ```
- Start Magnus llm task on device:
  ```sh
  adb shell 'cd /data/local/tmp/magnus_llmserver_release_android_aarch64_mtk_npu_v0.0.6; LD_LIBRARY_PATH=$PWD/libs ./tools/test_magnus_llm -p ./ -n llm'
  ```
- Stop existing Magnus or standalone server processes:
  ```sh
  adb shell 'pkill test_magnus_llm || true'
  adb shell 'pkill main_server || true'
  ```
- Forward the default Magnus HTTP port:
  ```sh
  adb forward tcp:12081 tcp:12081
  ```
- Basic checks:
  ```sh
  curl -sS http://127.0.0.1:12081/health
  curl -sS http://127.0.0.1:12081/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3-1.7b-mt8391","messages":[{"role":"user","content":"你好"}],"max_tokens":16}'
  ```

## Architecture overview

This is an Android NDK C++ codebase for MTK board-side LLM inference. It contains the original command-line MTK LLM runner plus an OpenAI-compatible HTTP server that can be loaded by Magnus through a C ABI.

- Public ABI: root `llmserver.h` declares `llmserver_start(const char*)` and `llmserver_stop()`.
- Magnus entrypoint: `llm_cmdline_tool/jni/llmserver/llmserver.cpp` parses JSON startup params, initializes one global `LlmEngine`, and starts one `OpenAIHttpServer`.
- Inference engine: `llm_cmdline_tool/jni/llmserver/llm_engine.cpp` wraps MTK runtime initialization, tokenizer setup, prompt formatting, prompt digestion, autoregressive generation, KV-cache reuse, reset, and cache-size protection.
- HTTP layer: `llm_cmdline_tool/jni/llmserver/openai_http_server.cpp` implements a small POSIX socket server with `/health`, `/v1/models`, `/v1/chat/completions`, and `/v1/completions`.
- Standalone runner: `llm_cmdline_tool/jni/llmserver/main_server.cpp` builds startup JSON and calls the same C ABI for local board testing outside Magnus.
- Minimal JSON: `llm_cmdline_tool/jni/llmserver/json_minimal.cpp` avoids adding extra runtime JSON dependencies.
- Original reference CLI: `llm_cmdline_tool/jni/main/main.cpp` is the baseline MTK inference flow that the llmserver implementation mirrors.

## Build structure

- Top-level NDK include chain starts at `llm_cmdline_tool/jni/Android.mk`.
- `jni/main/Android.mk` builds original CLI executables: `main`, `main_batch_gen`, `main_spec_dec`, `main_medusa`, and `main_tree_spec_dec_plus`.
- `jni/llmserver/Android.mk` builds:
  - `libllmserver.so` from `disable_heap_tagging.cpp`, `json_minimal.cpp`, `llm_engine.cpp`, `openai_http_server.cpp`, and `llmserver.cpp`.
  - `main_server` from `main_server.cpp`.
- Prebuilt/shared dependencies include `llm_prebuilt`, `common`, `tokenizer`, and `yaml_cpp`.

## Runtime behavior and API notes

- `llmserver_start()` accepts JSON fields including `config_file`, `yamlCfg`, `config`, `host`, `port`, `model_name`, `model_id`, `preformatter`, `max_tokens`, `keep_kv_cache`, and `preload_shared_weights`.
- Existing Magnus configs may pass `model` as a config path; do not treat that field as the OpenAI model id. Use `model_name` or `model_id` if a model id override is needed.
- `/v1/completions` applies the configured preformatter by default, usually `Qwen3NoInputNoThink`.
- `/v1/chat/completions` builds Qwen-style role-tagged prompts directly and does not apply the preformatter.
- KV-cache reuse depends on a stable `session_id` and a common prompt prefix. Cache misses reset runtime cache before digesting the new prompt.
- When cache eviction is disabled, generation is clamped so runtime `token_index` stays within `modelOpt_.cacheSize`, typically 4096.

## Development cautions

- Do not replace manual response JSON construction in `openai_http_server.cpp` with heavier `Json`/`ostringstream` construction without re-testing on Magnus; earlier versions were adjusted to avoid board-side request-time instability.
- Keep `disable_heap_tagging.cpp` and SIGPIPE handling in the HTTP server unless board-side crash behavior is revalidated.
- Prefer bounded device commands during testing; long-running Magnus server commands should be run in the background or stopped with `pkill test_magnus_llm`.
- Build outputs and binary/model artifacts are intentionally ignored by the root `.gitignore` (`libs/`, `obj/`, `*.so`, `*.dla`, `*.bin`, etc.).
