LOCAL_PATH := $(call my-dir)
USER_LOCAL_C_INCLUDES := $(LOCAL_C_INCLUDES)

LLMSERVER_SRC := disable_heap_tagging.cpp json_minimal.cpp llm_engine.cpp openai_http_server.cpp llmserver.cpp

include $(CLEAR_VARS)
LOCAL_MODULE := llmserver
LOCAL_SRC_FILES := $(LLMSERVER_SRC)
LOCAL_STATIC_LIBRARIES += utils
LOCAL_SHARED_LIBRARIES += llm_prebuilt common tokenizer yaml_cpp
LOCAL_C_INCLUDES := $(USER_LOCAL_C_INCLUDES) \
                   $(LOCAL_PATH) \
                   $(LOCAL_PATH)/../../../
LOCAL_LDLIBS := -llog
include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := main_server
LOCAL_SRC_FILES := main_server.cpp
LOCAL_SHARED_LIBRARIES += llmserver
LOCAL_C_INCLUDES := $(USER_LOCAL_C_INCLUDES) \
                   $(LOCAL_PATH) \
                   $(LOCAL_PATH)/../../../
LOCAL_LDLIBS := -llog
include $(BUILD_EXECUTABLE)

LOCAL_C_INCLUDES := $(USER_LOCAL_C_INCLUDES)
