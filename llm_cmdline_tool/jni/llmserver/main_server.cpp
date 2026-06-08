#include "llmserver.h"

#include <csignal>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>

namespace {

volatile std::sig_atomic_t gStop = 0;

void onSignal(int) {
    gStop = 1;
}

std::string jsonEscapeArg(const std::string& value) {
    std::string out;
    for (char c : value) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    std::string config = "config_np8-qwen3-1.7B.yaml";
    std::string host = "0.0.0.0";
    std::string model = "qwen3-1.7b-mt8391";
    std::string preformatter = "Qwen3NoInputNoThink";
    int port = 8000;
    int maxTokens = 4096;
    bool keepKvCache = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (arg == "--config" || arg == "--config-file") config = next();
        else if (arg == "--host") host = next();
        else if (arg == "--port") port = std::atoi(next());
        else if (arg == "--model") model = next();
        else if (arg == "--preformatter") preformatter = next();
        else if (arg == "--max-tokens") maxTokens = std::atoi(next());
        else if (arg == "--keep-kv-cache") keepKvCache = true;
    }

    std::ostringstream params;
    params << "{"
           << "\"config_file\":\"" << jsonEscapeArg(config) << "\","
           << "\"host\":\"" << jsonEscapeArg(host) << "\","
           << "\"port\":" << port << ","
           << "\"model\":\"" << jsonEscapeArg(model) << "\","
           << "\"preformatter\":\"" << jsonEscapeArg(preformatter) << "\","
           << "\"max_tokens\":" << maxTokens << ","
           << "\"keep_kv_cache\":" << (keepKvCache ? "true" : "false")
           << "}";

    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);
    llmserver_start(params.str().c_str());
    while (!gStop) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    llmserver_stop();
    return 0;
}
