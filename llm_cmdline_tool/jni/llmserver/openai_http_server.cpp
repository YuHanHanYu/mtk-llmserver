#include "openai_http_server.h"

#include "common/logging.h"

#include <arpa/inet.h>
#include <ctime>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <sstream>
#include <strings.h>
#include <sys/socket.h>
#include <unistd.h>

namespace llmserver {
namespace {

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

std::string nowId() {
    return "chatcmpl-mt8391-" + std::to_string(static_cast<long long>(std::time(nullptr)));
}

std::string statusText(int status) {
    if (status == 200) return "OK";
    if (status == 404) return "Not Found";
    if (status == 500) return "Internal Server Error";
    return "Bad Request";
}

} // namespace

OpenAIHttpServer::OpenAIHttpServer(const ServerConfig& config, std::shared_ptr<LlmEngine> engine)
    : config_(config), engine_(std::move(engine)) {}

OpenAIHttpServer::~OpenAIHttpServer() {
    stop();
}

bool OpenAIHttpServer::start(std::string* error) {
    struct sigaction sa{};
    sa.sa_handler = SIG_IGN;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGPIPE, &sa, nullptr);
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGPIPE);
    pthread_sigmask(SIG_BLOCK, &set, nullptr);
    listenFd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd_ < 0) {
        if (error) *error = "socket failed";
        return false;
    }

    int opt = 1;
    setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(config_.port);
    if (inet_pton(AF_INET, config_.host.c_str(), &addr.sin_addr) != 1) {
        addr.sin_addr.s_addr = INADDR_ANY;
    }

    if (bind(listenFd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        if (error) *error = "bind failed";
        close(listenFd_);
        listenFd_ = -1;
        return false;
    }
    if (listen(listenFd_, 8) < 0) {
        if (error) *error = "listen failed";
        close(listenFd_);
        listenFd_ = -1;
        return false;
    }

    running_ = true;
    thread_ = std::thread(&OpenAIHttpServer::serveLoop, this);
    LOG(INFO) << "llmserver HTTP listening on " << config_.host << ":" << config_.port;
    return true;
}

void OpenAIHttpServer::stop() {
    running_ = false;
    if (listenFd_ >= 0) {
        shutdown(listenFd_, SHUT_RDWR);
        close(listenFd_);
        listenFd_ = -1;
    }
    if (thread_.joinable()) thread_.join();
}

void OpenAIHttpServer::serveLoop() {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGPIPE);
    pthread_sigmask(SIG_BLOCK, &set, nullptr);
    while (running_) {
        int fd = accept(listenFd_, nullptr, nullptr);
        if (fd < 0) continue;
        handleClient(fd);
        close(fd);
    }
}

void OpenAIHttpServer::handleClient(int clientFd) {
    std::string request;
    char buf[4096];
    while (request.find("\r\n\r\n") == std::string::npos) {
        ssize_t n = recv(clientFd, buf, sizeof(buf), 0);
        if (n <= 0) return;
        request.append(buf, n);
        if (request.size() > 1024 * 1024) return;
    }

    const auto headerEnd = request.find("\r\n\r\n");
    std::string headers = request.substr(0, headerEnd);
    size_t contentLength = 0;
    std::istringstream hs(headers);
    std::string requestLine;
    std::getline(hs, requestLine);
    std::string line;
    while (std::getline(hs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = line.substr(0, colon);
        if (strcasecmp(key.c_str(), "Content-Length") == 0) {
            contentLength = static_cast<size_t>(std::atoi(line.substr(colon + 1).c_str()));
        }
    }

    std::string body = request.substr(headerEnd + 4);
    while (body.size() < contentLength) {
        ssize_t n = recv(clientFd, buf, sizeof(buf), 0);
        if (n <= 0) return;
        body.append(buf, n);
    }

    std::istringstream rs(requestLine);
    std::string method, path, version;
    rs >> method >> path >> version;
    const auto query = path.find('?');
    if (query != std::string::npos) path = path.substr(0, query);
    bool stream = false;
    std::string response = handleRequest(method, path, body, stream, clientFd);
    if (!stream) {
        sendAll(clientFd, response);
    }
}

std::string OpenAIHttpServer::handleRequest(const std::string& method, const std::string& path,
                                            const std::string& body, bool& stream, int clientFd) {
    if (method == "GET" && path == "/health") return httpResponse("{\"status\":\"ok\"}");
    if (method == "GET" && path == "/v1/models") return httpResponse(modelsResponse());
    if (method != "POST") return errorResponse("not found", 404);

    std::string error;
    Json request = parseJson(body, &error);
    if (!error.empty()) return errorResponse("invalid json: " + error);
    stream = request.get("stream").boolValue(false);

    if (path == "/v1/chat/completions") return completionResponse(request, true, stream, clientFd);
    if (path == "/v1/completions") return completionResponse(request, false, stream, clientFd);
    return errorResponse("not found", 404);
}

std::string OpenAIHttpServer::modelsResponse() const {
    Json model = jsonObject({
        {"id", jsonString(engine_->model())},
        {"object", jsonString("model")},
        {"created", jsonNumber(std::time(nullptr))},
        {"owned_by", jsonString("mtk")},
    });
    return dumpJson(jsonObject({{"object", jsonString("list")}, {"data", Json{Json::Array, false, 0, "", {model}, {}}}}));
}

std::string OpenAIHttpServer::completionResponse(const Json& request, bool chat, bool stream, int clientFd) {
    GenerateRequest gen;
    gen.maxTokens = request.get("max_tokens").intValue(static_cast<int>(engine_->defaultMaxTokens()));
    gen.keepKvCache = request.get("keep_kv_cache").boolValue(
        request.get("keepKvCache").boolValue(request.get("cache").boolValue(true)));
    gen.reset = request.get("reset").boolValue(false);
    gen.sessionId = request.get("session_id").stringValue("");
    LOG(INFO) << "llmserver request endpoint=" << (chat ? "/v1/chat/completions" : "/v1/completions")
              << ", stream=" << stream << ", max_tokens=" << gen.maxTokens
              << ", keep_kv_cache=" << gen.keepKvCache << ", reset=" << gen.reset
              << ", session_id=" << gen.sessionId;

    if (chat) {
        gen.prompt = chatPrompt(request.get("messages"));
        gen.applyPreformatter = false;
    } else {
        gen.prompt = request.get("prompt").stringValue("");
        gen.applyPreformatter = true;
    }

    if (gen.prompt.empty()) return errorResponse("prompt/messages is required");

    std::string id = nowId();
    std::string error;
    if (stream) {
        std::ostringstream hdr;
        hdr << "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
        sendAll(clientFd, hdr.str());
        auto cb = [&](const std::string& token) {
            Json delta = chat ? jsonObject({{"content", jsonString(token)}})
                              : jsonObject({{"text", jsonString(token)}});
            Json chunk = jsonObject({
                {"id", jsonString(id)},
                {"object", jsonString(chat ? "chat.completion.chunk" : "text_completion.chunk")},
                {"created", jsonNumber(std::time(nullptr))},
                {"model", jsonString(engine_->model())},
                {"choices", Json{Json::Array, false, 0, "", {jsonObject({{"index", jsonNumber(0)}, {chat ? "delta" : "", Json{}}, {"finish_reason", Json{}}})}, {}}},
            });
            if (chat) chunk.o["choices"].a[0].o["delta"] = delta;
            else chunk.o["choices"].a[0].o["text"] = jsonString(token);
            sendAll(clientFd, "data: " + dumpJson(chunk) + "\n\n");
            return true;
        };
        engine_->generate(gen, cb, &error);
        if (!error.empty()) LOG(WARN) << "llmserver stream generate error=" << error;
        sendAll(clientFd, "data: [DONE]\n\n");
        return "";
    }

    GenerateResult result = engine_->generate(gen, nullptr, &error);
    if (!error.empty()) return errorResponse(error, 500);

    std::string json = "{\"id\":\"" + jsonEscape(id) + "\",";
    json += "\"object\":\"";
    json += chat ? "chat.completion" : "text_completion";
    json += "\",\"created\":" + std::to_string(static_cast<long long>(std::time(nullptr)));
    json += ",\"model\":\"" + jsonEscape(engine_->model()) + "\",\"choices\":[{";
    if (chat) {
        json += "\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"";
        json += jsonEscape(result.text);
        json += "\"},\"finish_reason\":\"stop\"";
    } else {
        json += "\"index\":0,\"text\":\"";
        json += jsonEscape(result.text);
        json += "\",\"finish_reason\":\"stop\"";
    }
    json += "}],\"usage\":{\"prompt_tokens\":" + std::to_string(result.promptTokens);
    json += ",\"completion_tokens\":" + std::to_string(result.completionTokens);
    json += ",\"total_tokens\":" + std::to_string(result.promptTokens + result.completionTokens);
    json += "},\"mtk_token_index\":" + std::to_string(result.tokenIndex) + "}";
    return httpResponse(json);
}

std::string OpenAIHttpServer::chatPrompt(const Json& messages) const {
    std::ostringstream prompt;
    if (messages.isArray()) {
        for (const auto& msg : messages.a) {
            std::string role = msg.get("role").stringValue("user");
            std::string content = msg.get("content").stringValue("");
            prompt << "<|im_start|>" << role << "\n" << content << "<|im_end|>\n";
        }
    }
    prompt << "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return prompt.str();
}

std::string OpenAIHttpServer::errorResponse(const std::string& message, int status) const {
    Json body = jsonObject({{"error", jsonObject({{"message", jsonString(message)}, {"type", jsonString("llmserver_error")}})}});
    return httpResponse(dumpJson(body), "application/json", status, statusText(status));
}

std::string OpenAIHttpServer::httpResponse(const std::string& body, const std::string& contentType,
                                           int codeValue, const std::string& messageText) const {
    std::ostringstream out;
    out << "HTTP/1.1 " << std::to_string(codeValue) << ' ' << messageText << "\r\n"
        << "Content-Type: " << contentType << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    return out.str();
}

bool OpenAIHttpServer::sendAll(int fd, const std::string& data) const {
    const char* p = data.data();
    size_t left = data.size();
    while (left > 0) {
        ssize_t n = send(fd, p, left, MSG_NOSIGNAL);
        if (n <= 0) {
            LOG(WARN) << "llmserver send failed n=" << n << ", left=" << left;
            return false;
        }
        p += n;
        left -= n;
    }
    return true;
}

} // namespace llmserver
