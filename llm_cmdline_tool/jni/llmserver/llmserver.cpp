#include "llmserver.h"

#include "json_minimal.h"
#include "llm_engine.h"
#include "openai_http_server.h"

#include <memory>
#include <mutex>

namespace {

std::mutex gMutex;
std::shared_ptr<llmserver::LlmEngine> gEngine;
std::unique_ptr<llmserver::OpenAIHttpServer> gServer;

} // namespace

extern "C" void llmserver_start(const char* json_params) {
    std::lock_guard<std::mutex> lock(gMutex);
    if (gServer) return;

    std::string error;
    llmserver::Json params = llmserver::parseJson(json_params ? json_params : "{}", &error);
    if (!error.empty()) return;

    llmserver::EngineConfig engineConfig;
    engineConfig.configFile = params.get("config_file").stringValue(
        params.get("yamlCfg").stringValue(params.get("config").stringValue("config_np8-qwen3-1.7B.yaml")));
    engineConfig.model = params.get("model_name").stringValue(
        params.get("model_id").stringValue("qwen3-1.7b-mt8391"));
    engineConfig.preformatter = params.get("preformatter").stringValue("Qwen3NoInputNoThink");
    engineConfig.maxTokens = params.get("max_tokens").intValue(4096);
    engineConfig.keepKvCache = params.get("keep_kv_cache").boolValue(false);
    engineConfig.preloadSharedWeights = params.get("preload_shared_weights").boolValue(false);

    llmserver::ServerConfig serverConfig;
    serverConfig.host = params.get("host").stringValue("0.0.0.0");
    serverConfig.port = params.get("port").intValue(8000);

    gEngine = std::make_shared<llmserver::LlmEngine>();
    if (!gEngine->init(engineConfig, &error)) {
        gEngine.reset();
        return;
    }

    gServer.reset(new llmserver::OpenAIHttpServer(serverConfig, gEngine));
    if (!gServer->start(&error)) {
        gServer.reset();
        gEngine->release();
        gEngine.reset();
    }
}

extern "C" void llmserver_stop() {
    std::lock_guard<std::mutex> lock(gMutex);
    if (gServer) {
        gServer->stop();
        gServer.reset();
    }
    if (gEngine) {
        gEngine->release();
        gEngine.reset();
    }
}
