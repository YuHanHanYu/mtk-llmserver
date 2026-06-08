#pragma once

#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_factory.h"
#include "mtk_llm_options.h"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace llmserver {

struct EngineConfig {
    std::string configFile;
    std::string model = "qwen3-1.7b-mt8391";
    std::string preformatter = "Qwen3NoInputNoThink";
    size_t maxTokens = 4096;
    bool keepKvCache = false;
    bool preloadSharedWeights = false;
};

struct GenerateRequest {
    std::string prompt;
    size_t maxTokens = 0;
    bool applyPreformatter = true;
    bool keepKvCache = false;
    bool reset = false;
    std::string sessionId;
};

struct GenerateResult {
    std::string text;
    size_t promptTokens = 0;
    size_t completionTokens = 0;
    size_t tokenIndex = 0;
};

class LlmEngine {
public:
    using TokenCallback = std::function<bool(const std::string&)>;

    LlmEngine();
    ~LlmEngine();

    bool init(const EngineConfig& config, std::string* error);
    GenerateResult generate(const GenerateRequest& request, const TokenCallback& onToken,
                            std::string* error);
    void reset();
    void release();

    const std::string& model() const { return config_.model; }
    size_t defaultMaxTokens() const { return config_.maxTokens; }

private:
    using TokenType = mtk::Tokenizer::TokenType;
    using TokenizerUPtr = std::unique_ptr<mtk::Tokenizer>;

    TokenizerUPtr prepareTokenizer();
    TokenType digestPrompt(const std::vector<TokenType>& inputTokens, size_t modelTokenSize,
                           size_t& promptTokens);
    TokenType autoregressiveStep(TokenType inputToken);
    bool isStopToken(TokenType token) const;
    size_t commonPrefixLength(const std::vector<TokenType>& a, const std::vector<TokenType>& b) const;
    bool canFitCache(size_t startTokenIndex, size_t promptTokensToDigest, size_t maxTokens) const;
    void swapModel(size_t tokenSize);
    void resetLocked();

    EngineConfig config_;
    LlmModelOptions modelOpt_;
    LlmRuntimeOptions runtimeOpt_;
    void* runtime_ = nullptr;
    TokenizerUPtr tokenizer_;
    std::string currentSessionId_;
    std::vector<TokenType> cachedPromptTokens_;
    size_t cachedPromptTokenIndex_ = 0;
    TokenType cachedFirstOutputToken_ = 0;
    bool hasCachedFirstOutputToken_ = false;
    bool inGenMode_ = false;
    bool hasActiveCache_ = false;
    size_t inferenceStep_ = 0;
    std::mutex mutex_;
};

} // namespace llmserver
