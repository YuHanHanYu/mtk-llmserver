@echo on

:: Setup phone environment
set PHONE_PATH=/data/local/tmp/llm_sdk
adb shell "mkdir -p %PHONE_PATH%"

set MODEL_DIR=Qwen3-4B
adb shell "mkdir -p %PHONE_PATH%/%MODEL_DIR%"

adb push ../dla/Qwen3-4B/tokenizer %PHONE_PATH%/%MODEL_DIR%/
adb push ../dla/Qwen3-4B/embedding_int16.bin %PHONE_PATH%/%MODEL_DIR%/tokenizer/

:error
:: pause