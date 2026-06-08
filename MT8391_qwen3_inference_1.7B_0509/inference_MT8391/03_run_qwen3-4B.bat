@echo on

:: Setup phone environment
set PHONE_PATH=/data/local/tmp/llm_sdk

set CONFIG_FILE=config_np8-qwen3-1.7B.yaml
set INPUT_PROMPT=sample_prompt-q1.txt
set MAX_RESPONSE=384
set PREFORMATTER=Qwen3NoInput

:: Push yaml config file
adb push %CONFIG_FILE% %PHONE_PATH%
adb push %INPUT_PROMPT% %PHONE_PATH%

:: Set execute permission
adb shell "chmod +x %PHONE_PATH%/main"

:: Run using the below commands
adb shell "cd %PHONE_PATH%; LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD ./main %CONFIG_FILE% -i %INPUT_PROMPT% --preformatter %PREFORMATTER% -m %MAX_RESPONSE% --one-prompt-per-line"

:error
