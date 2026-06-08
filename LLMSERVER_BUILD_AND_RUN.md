# MTK LLM Server / Magnus 集成流程说明

## 1. 项目目录说明

当前相关目录如下：

```text
/Users/hanyu/Documents/sbc/mtk_v0606/
├── llmserver.h
├── LLMSERVER_BUILD_AND_RUN.md
├── llm_cmdline_tool/
│   ├── jni/
│   │   ├── Android.mk
│   │   ├── main/
│   │   │   └── main.cpp
│   │   ├── llmserver/
│   │   │   ├── Android.mk
│   │   │   ├── llmserver.cpp
│   │   │   ├── llm_engine.h
│   │   │   ├── llm_engine.cpp
│   │   │   ├── openai_http_server.h
│   │   │   ├── openai_http_server.cpp
│   │   │   ├── json_minimal.h
│   │   │   ├── json_minimal.cpp
│   │   │   ├── main_server.cpp
│   │   │   └── disable_heap_tagging.cpp
│   │   ├── utils/
│   │   └── prebuilt/
│   └── libs/arm64-v8a/
│       ├── libllmserver.so
│       ├── main_server
│       ├── libmtk_llm.so
│       ├── libcommon.so
│       ├── libtokenizer.so
│       ├── libyaml-cpp.so
│       └── libc++_shared.so
└── MT8391_qwen3_inference_1.7B_0509/
    └── inference_MT8391/
        ├── 03_run_qwen3-1.7B.sh
        ├── config_np8-qwen3-1.7B.yaml
        └── sample_prompt-q1.txt
```

核心文件说明：

| 文件 | 作用 |
|---|---|
| `llmserver.h` | 对外 C ABI 头文件，供 Magnus 加载调用 |
| `llmserver.cpp` | 实现 `llmserver_start()` / `llmserver_stop()` |
| `llm_engine.cpp` | 封装 MTK LLM runtime、tokenizer、推理、KV-cache 逻辑 |
| `openai_http_server.cpp` | OpenAI 兼容 HTTP 服务实现 |
| `main_server.cpp` | 独立启动 HTTP 服务的测试程序 |
| `json_minimal.cpp` | 最小 JSON 解析/序列化实现，避免额外依赖 |
| `disable_heap_tagging.cpp` | Android heap tagging 兼容处理 |
| `Android.mk` | NDK 构建配置 |

## 2. 当前代码支持功能

### 2.1 Magnus 动态库接口

生成的动态库为：

```text
libllmserver.so
```

导出接口来自 `llmserver.h`：

```c
void llmserver_start(const char* json_params);
void llmserver_stop();
```

Magnus 加载 `libllmserver.so` 后，调用 `llmserver_start()` 启动推理服务，调用 `llmserver_stop()` 停止服务。

### 2.2 OpenAI 兼容接口

HTTP 服务支持：

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions
```

支持普通输出和流式输出：

```json
{
  "stream": true
}
```

### 2.3 Prompt 格式化

`/v1/completions` 默认使用：

```text
Qwen3NoInputNoThink
```

会把用户 prompt 包装成 Qwen3 对话格式。

`/v1/chat/completions` 会根据 `messages` 拼接：

```text
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
<think>

</think>

```

### 2.4 KV-cache 支持

当前已支持前缀 KV-cache 复用。

请求中可以传：

```json
{
  "keep_kv_cache": true,
  "session_id": "session-001"
}
```

同一个 `session_id` 下，如果新 prompt 和上一次 prompt 有相同长前缀，服务会：

1. 复用已有前缀 KV-cache；
2. rollback 到公共前缀位置；
3. 只 digest 新 prompt 的 suffix；
4. 避免重复处理完整长 prompt。

也兼容以下字段名：

```json
{
  "keepKvCache": true
}
```

或：

```json
{
  "cache": true
}
```

强制清 cache：

```json
{
  "reset": true
}
```

### 2.5 Cache 上限保护

模型配置中 `cacheSize` 为 4096。

服务会保证：

```text
prompt token 数 + completion token 数 <= 4096
```

如果请求的 `max_tokens` 过大，会自动 clamp，例如：

```text
WARN: llmserver clamp max_tokens from 4096 to 1674 to keep token_index <= 4096
```

含义是本次最多只能生成 1674 个 token，避免 KV-cache 超过 4096。

### 2.6 日志输出

当前日志只保留关键信息：

- 服务启动；
- 请求 endpoint / max_tokens / session_id；
- cache hit / cache miss；
- reused tokens / digest tokens；
- token_index / cache_size；
- 生成完成统计；
- cache clamp / send failure 等 warning。

## 3. 编译指令

在主机执行：

```sh
cd /Users/hanyu/Documents/sbc/mtk_v0606/llm_cmdline_tool
ndk-build -j8
```

编译成功后产物位于：

```text
libs/arm64-v8a/libllmserver.so
libs/arm64-v8a/main_server
```

## 4. 所需 so 清单

运行 `libllmserver.so` / `main_server` 需要以下 so 位于 `LD_LIBRARY_PATH` 可见目录：

```text
libllmserver.so
libmtk_llm.so
libcommon.so
libtokenizer.so
libyaml-cpp.so
libc++_shared.so
```

Magnus 运行目录中一般放在：

```text
/data/local/tmp/magnus_llmserver_release_android_aarch64_mtk_npu_v0.0.6/libs/
```

## 5. 部署到 Magnus 目录

当前 Magnus 目录：

```text
/data/local/tmp/magnus_llmserver_release_android_aarch64_mtk_npu_v0.0.6
```

推送新版 `libllmserver.so`：

```sh
cd /Users/hanyu/Documents/sbc/mtk_v0606/llm_cmdline_tool
adb push libs/arm64-v8a/libllmserver.so \
  /data/local/tmp/magnus_llmserver_release_android_aarch64_mtk_npu_v0.0.6/libs/libllmserver.so
```

如依赖 so 也需要一起替换，可推送：

```sh
adb push libs/arm64-v8a/libmtk_llm.so \
  /data/local/tmp/magnus_llmserver_release_android_aarch64_mtk_npu_v0.0.6/libs/

adb push libs/arm64-v8a/libcommon.so \
  /data/local/tmp/magnus_llmserver_release_android_aarch64_mtk_npu_v0.0.6/libs/

adb push libs/arm64-v8a/libtokenizer.so \
  /data/local/tmp/magnus_llmserver_release_android_aarch64_mtk_npu_v0.0.6/libs/

adb push libs/arm64-v8a/libyaml-cpp.so \
  /data/local/tmp/magnus_llmserver_release_android_aarch64_mtk_npu_v0.0.6/libs/

adb push libs/arm64-v8a/libc++_shared.so \
  /data/local/tmp/magnus_llmserver_release_android_aarch64_mtk_npu_v0.0.6/libs/
```

## 6. Magnus 配置说明

Magnus 使用任务配置：

```text
/data/local/tmp/magnus_llmserver_release_android_aarch64_mtk_npu_v0.0.6/tasks/llm/task.pbtxt
```

旧版配置中常见字段：

```json
{
  "llmserver": {
    "backend": "mtk_npu",
    "model": "res/llm_v1/config.yaml",
    "host": "0.0.0.0",
    "logDisable": 0,
    "port": 12081
  }
}
```

当前 `libllmserver.so` 已兼容旧版 Magnus 字段：

- `yamlCfg`：作为实际 YAML 配置文件路径；
- `model`：旧版中可能是配置路径，不再作为 OpenAI model id 使用；
- `model_name` / `model_id`：如果传入，则作为返回给 OpenAI API 的 model 名称。

Magnus 实际传入 `llmserver_start()` 的参数类似：

```json
{
  "backend": "mtk_npu",
  "host": "0.0.0.0",
  "logDisable": 0,
  "model": ".//res/llm_v1/config.yaml",
  "port": 12081,
  "yamlCfg": ".//runtime.yaml"
}
```

## 7. 启动 Magnus 服务

进入设备端 Magnus 目录启动：

```sh
adb shell '
cd /data/local/tmp/magnus_llmserver_release_android_aarch64_mtk_npu_v0.0.6
LD_LIBRARY_PATH=$PWD/libs ./tools/test_magnus_llm -p ./ -n llm
'
```

如果已有旧进程，先停止：

```sh
adb shell 'pkill test_magnus_llm || true'
```

然后重新启动：

```sh
adb shell '
cd /data/local/tmp/magnus_llmserver_release_android_aarch64_mtk_npu_v0.0.6
LD_LIBRARY_PATH=$PWD/libs ./tools/test_magnus_llm -p ./ -n llm
'
```

## 8. 端口转发

Magnus 默认端口为 `12081`。

主机访问前需要执行：

```sh
adb forward tcp:12081 tcp:12081
```

## 9. 基础接口测试

### 9.1 健康检查

```sh
curl -sS http://127.0.0.1:12081/health
```

预期：

```json
{"status":"ok"}
```

### 9.2 Chat Completions

```sh
curl -sS http://127.0.0.1:12081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-1.7b-mt8391",
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "max_tokens": 16
  }'
```

### 9.3 Completions

```sh
curl -sS http://127.0.0.1:12081/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-1.7b-mt8391",
    "prompt": "你好，介绍一下你自己",
    "max_tokens": 128
  }'
```

### 9.4 流式输出

```sh
curl -N http://127.0.0.1:12081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-1.7b-mt8391",
    "stream": true,
    "messages": [
      {"role": "user", "content": "写一首短诗"}
    ],
    "max_tokens": 128
  }'
```

## 10. KV-cache 测试

### 10.1 第一次请求：长前缀 + suffix 1

```sh
curl -sS http://127.0.0.1:12081/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-1.7b-mt8391",
    "prompt": "共同前缀内容：这是一段很长的共享上下文......第一个问题：请总结核心规则。",
    "max_tokens": 64,
    "keep_kv_cache": true,
    "session_id": "kvtest"
  }'
```

日志预期：

```text
llmserver cache miss: prompt_tokens=..., reused_tokens=0, digest_tokens=..., start_token_index=0, cache_size=4096
```

### 10.2 第二次请求：相同长前缀 + suffix 2

```sh
curl -sS http://127.0.0.1:12081/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-1.7b-mt8391",
    "prompt": "共同前缀内容：这是一段很长的共享上下文......第二个问题：请列出三个要点。",
    "max_tokens": 64,
    "keep_kv_cache": true,
    "session_id": "kvtest"
  }'
```

日志预期：

```text
llmserver cache hit: prompt_tokens=..., reused_tokens=..., digest_tokens=..., start_token_index=..., cache_size=4096
```

其中：

```text
reused_tokens > 0
digest_tokens << prompt_tokens
```

说明 KV-cache 前缀复用生效。

### 10.3 Cache miss/reset 测试

同一个 `session_id`，但换成完全不同前缀：

```sh
curl -sS http://127.0.0.1:12081/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-1.7b-mt8391",
    "prompt": "完全不同的新前缀：这段文本不应该复用前一次KV缓存。",
    "max_tokens": 64,
    "keep_kv_cache": true,
    "session_id": "kvtest"
  }'
```

日志预期：

```text
llmserver cache miss: prompt_tokens=..., reused_tokens=0, digest_tokens=..., start_token_index=0, cache_size=4096
```

说明没有命中 KV-cache 时会 reset cache 后重新 digest。

## 11. Cache 4096 上限测试

发送较长 prompt，并设置较大的 `max_tokens`：

```sh
curl -sS http://127.0.0.1:12081/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-1.7b-mt8391",
    "prompt": "超长缓存预算测试：这段文本用于测试cache不会超过4096......",
    "max_tokens": 4096,
    "keep_kv_cache": true,
    "session_id": "overflowtest"
  }'
```

如果 prompt 已占用较多 token，日志中可能出现：

```text
WARN: llmserver clamp max_tokens from 4096 to xxxx to keep token_index <= 4096
```

这是正常保护逻辑，表示服务自动降低生成长度，保证：

```text
token_index <= 4096
```

## 12. 独立 main_server 启动方式

如果不通过 Magnus，也可以直接启动 `main_server`。

推送到设备目录后：

```sh
adb shell '
cd /data/local/tmp/llm_sdk
LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH ./main_server \
  --config config_np8-qwen3-1.7B.yaml \
  --host 0.0.0.0 \
  --port 8000 \
  --preformatter Qwen3NoInputNoThink \
  --model qwen3-1.7b-mt8391 \
  --max-tokens 4096
'
```

主机转发：

```sh
adb forward tcp:8000 tcp:8000
```

测试：

```sh
curl -sS http://127.0.0.1:8000/health
```

## 13. 常用停止命令

停止 Magnus 测试进程：

```sh
adb shell 'pkill test_magnus_llm || true'
```

停止独立 `main_server`：

```sh
adb shell 'pkill main_server || true'
```
