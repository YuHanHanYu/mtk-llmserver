@echo on

:: Setup phone environment
set PHONE_PATH=/data/local/tmp/llm_sdk
adb shell "mkdir -p %PHONE_PATH%"

set MODEL_DIR=Qwen3-4B
adb shell "mkdir -p %PHONE_PATH%/%MODEL_DIR%"

set MODEL_NAME=2048c

adb push ../dla/Qwen3-4B/%MODEL_NAME% %PHONE_PATH%/%MODEL_DIR%/%MODEL_NAME%/

:error