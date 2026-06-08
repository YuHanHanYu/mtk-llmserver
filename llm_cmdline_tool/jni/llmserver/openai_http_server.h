#pragma once

#include "json_minimal.h"
#include "llm_engine.h"

#include <atomic>
#include <memory>
#include <string>
#include <thread>

namespace llmserver {

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8000;
};

class OpenAIHttpServer {
public:
    OpenAIHttpServer(const ServerConfig& config, std::shared_ptr<LlmEngine> engine);
    ~OpenAIHttpServer();

    bool start(std::string* error);
    void stop();

private:
    void serveLoop();
    void handleClient(int clientFd);
    std::string handleRequest(const std::string& method, const std::string& path,
                              const std::string& body, bool& stream, int clientFd);
    std::string modelsResponse() const;
    std::string completionResponse(const Json& request, bool chat, bool stream, int clientFd);
    std::string errorResponse(const std::string& message, int status = 400) const;
    std::string httpResponse(const std::string& body, const std::string& contentType = "application/json",
                             int codeValue = 200, const std::string& messageText = "OK") const;
    std::string chatPrompt(const Json& messages) const;
    bool sendAll(int fd, const std::string& data) const;

    ServerConfig config_;
    std::shared_ptr<LlmEngine> engine_;
    std::atomic<bool> running_{false};
    int listenFd_ = -1;
    std::thread thread_;
};

} // namespace llmserver
