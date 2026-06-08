#include "llm_engine.h"

#include "common/dump.h"
#include "common/logging.h"
#include "mtk_llm.h"
#include "utils/NPUWareUtilsLib.h"
#include "utils/config_parser.h"
#include "utils/utils.h"

#include <algorithm>
#include <cmath>

namespace llmserver {
namespace {

SharedWeightsHandle* gSharedWeightsHandle = nullptr;

} // namespace

LlmEngine::LlmEngine() = default;

LlmEngine::~LlmEngine() {
    release();
}

bool LlmEngine::init(const EngineConfig& config, std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    release();
    config_ = config;
    modelOpt_ = {};
    runtimeOpt_ = {};

    if (config_.configFile.empty()) {
        if (error) *error = "config_file is required";
        return false;
    }

    utils::YamlConfigParser configParser(config_.configFile);
    configParser.parse(modelOpt_, runtimeOpt_);

    if (config_.preloadSharedWeights && gSharedWeightsHandle == nullptr) {
        mtk_llm_shared_weights_preload(&gSharedWeightsHandle, runtimeOpt_);
    }

    if (!mtk_llm_init(&runtime_, modelOpt_, runtimeOpt_, gSharedWeightsHandle)) {
        runtime_ = nullptr;
        if (error) *error = "mtk_llm_init failed";
        return false;
    }

    tokenizer_ = prepareTokenizer();
    inGenMode_ = false;
    hasActiveCache_ = false;
    currentSessionId_.clear();
    LOG(INFO) << "llmserver initialized model " << config_.model
              << ", vocab size: " << tokenizer_->vocabSize();
    return true;
}

LlmEngine::TokenizerUPtr LlmEngine::prepareTokenizer() {
    auto tokenizer = mtk::TokenizerFactory().create(runtimeOpt_.tokenizerPath, runtimeOpt_.tokenizerRegex);
    const auto& specialTokens = runtimeOpt_.specialTokens;
    if (specialTokens.addBos) tokenizer->enableBosToken(specialTokens.bosId);
    return tokenizer;
}

LlmEngine::TokenType LlmEngine::digestPrompt(const std::vector<TokenType>& inputTokens,
                                             size_t modelTokenSize, size_t& promptTokens) {
    void* lastLogits = nullptr;
    const auto logitsType = modelOpt_.modelOutputType;
    const auto inputTokenCount = inputTokens.size();
    size_t inputTokenIndex = 0;
    const auto inpBeginIt = inputTokens.cbegin();

    const auto startTokenIndex = mtk_llm_get_token_index(runtime_);
    const bool useCacheComp = mtk_llm_is_enabled_cache_eviction(runtime_);
    if (!useCacheComp && startTokenIndex + inputTokenCount > modelOpt_.cacheSize) {
        LOG(WARN) << "Prompt length may overflow cache: start=" << startTokenIndex
                  << ", prompt=" << inputTokenCount << ", cache=" << modelOpt_.cacheSize;
    }

    while (inputTokenIndex < inputTokenCount) {
        SET_DUMP_INDEX(inferenceStep_++);
        const size_t numInputTokenLeft = inputTokenCount - inputTokenIndex;
        const size_t remainder = numInputTokenLeft % modelTokenSize;
        const size_t numNewTok = remainder ? remainder : modelTokenSize;
        const auto tokIdxStart = inputTokenIndex;
        const auto tokIdxEnd = tokIdxStart + numNewTok;
        const auto curInputTokens = std::vector<TokenType>(inpBeginIt + tokIdxStart,
                                                           inpBeginIt + tokIdxEnd);
        const bool isLastPromptStep = inputTokenIndex + numNewTok >= inputTokenCount;
        const auto logitsKind = isLastPromptStep ? mtk::LogitsKind::LAST : mtk::LogitsKind::NONE;
        lastLogits = mtk_llm_inference_once(runtime_, curInputTokens, logitsKind);
        inputTokenIndex += numNewTok;
    }

    promptTokens = inputTokenCount;
    return utils::argmaxFrom16bitLogits(logitsType, lastLogits, tokenizer_->vocabSize());
}

LlmEngine::TokenType LlmEngine::autoregressiveStep(TokenType inputToken) {
    void* lastLogits = mtk_llm_inference_once(runtime_, {inputToken});
    return utils::argmaxFrom16bitLogits(modelOpt_.modelOutputType, lastLogits, tokenizer_->vocabSize());
}

bool LlmEngine::isStopToken(TokenType token) const {
    const auto& stopTokenSet = runtimeOpt_.specialTokens.stopToken;
    return stopTokenSet.find(token) != stopTokenSet.end();
}

size_t LlmEngine::commonPrefixLength(const std::vector<TokenType>& a, const std::vector<TokenType>& b) const {
    const size_t limit = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < limit && a[i] == b[i]) ++i;
    return i;
}

bool LlmEngine::canFitCache(size_t startTokenIndex, size_t promptTokensToDigest, size_t maxTokens) const {
    if (mtk_llm_is_enabled_cache_eviction(runtime_)) return true;
    return startTokenIndex + promptTokensToDigest + maxTokens <= modelOpt_.cacheSize;
}

void LlmEngine::swapModel(size_t tokenSize) {
    mtk_llm_swap_model(runtime_, tokenSize);
    inGenMode_ = tokenSize == modelOpt_.genTokenBatchSize;
}

GenerateResult LlmEngine::generate(const GenerateRequest& request, const TokenCallback& onToken,
                                   std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    GenerateResult result;
    if (!runtime_ || !tokenizer_) {
        if (error) *error = "engine is not initialized";
        return result;
    }

    std::string prompt = request.prompt;
    if (request.applyPreformatter && !config_.preformatter.empty()) {
        if (!utils::addPreformatter(config_.preformatter, prompt)) {
            LOG(WARN) << "Invalid preformatter: " << config_.preformatter;
        }
    }

    auto inputTokens = tokenizer_->tokenize(prompt);
    if (inputTokens.empty()) {
        if (error) *error = "empty prompt after tokenization";
        return result;
    }

    const bool keepCache = request.keepKvCache || config_.keepKvCache;
    const bool sessionChanged = !request.sessionId.empty() && request.sessionId != currentSessionId_;
    size_t maxTokens = request.maxTokens ? request.maxTokens : config_.maxTokens;
    bool cacheHit = false;
    size_t reusedTokens = 0;
    size_t promptOffset = 0;
    const size_t prefix = commonPrefixLength(cachedPromptTokens_, inputTokens);
    constexpr size_t kMinReusablePrefixTokens = 32;
    if (keepCache && !request.reset && hasActiveCache_ && !sessionChanged && prefix >= kMinReusablePrefixTokens) {
        const size_t tokenIndex = mtk_llm_get_token_index(runtime_);
        if (tokenIndex > prefix) {
            mtk_llm_rollback(runtime_, tokenIndex - prefix);
        }
        cacheHit = true;
        reusedTokens = prefix;
        promptOffset = reusedTokens;
    }

    std::vector<TokenType> promptTokens(inputTokens.begin() + promptOffset, inputTokens.end());
    if (!cacheHit || !canFitCache(mtk_llm_get_token_index(runtime_), promptTokens.size(), maxTokens)) {
        if (cacheHit) {
            LOG(WARN) << "llmserver cache hit exceeds budget, reset cache: start="
                      << mtk_llm_get_token_index(runtime_) << ", suffix_tokens=" << promptTokens.size()
                      << ", max_tokens=" << maxTokens << ", cache_size=" << modelOpt_.cacheSize;
        }
        resetLocked();
        cacheHit = false;
        reusedTokens = 0;
        promptOffset = 0;
        promptTokens = inputTokens;
    }
    if (!request.sessionId.empty()) currentSessionId_ = request.sessionId;

    const size_t startTokenIndex = mtk_llm_get_token_index(runtime_);
    if (!mtk_llm_is_enabled_cache_eviction(runtime_)) {
        if (startTokenIndex + promptTokens.size() >= modelOpt_.cacheSize) {
            if (error) *error = "prompt exceeds cache budget";
            return result;
        }
        const size_t available = modelOpt_.cacheSize - startTokenIndex - promptTokens.size();
        if (maxTokens > available) {
            LOG(WARN) << "llmserver clamp max_tokens from " << maxTokens << " to " << available
                      << " to keep token_index <= " << modelOpt_.cacheSize;
            maxTokens = available;
        }
    }

    LOG(INFO) << "llmserver cache " << (cacheHit ? "hit" : "miss")
              << ": prompt_tokens=" << inputTokens.size() << ", reused_tokens=" << reusedTokens
              << ", digest_tokens=" << promptTokens.size() << ", start_token_index=" << startTokenIndex
              << ", cache_size=" << modelOpt_.cacheSize;

    TokenType outputToken = 0;
    if (promptTokens.empty() && hasCachedFirstOutputToken_) {
        outputToken = cachedFirstOutputToken_;
        result.promptTokens = inputTokens.size();
    } else {
        if (inGenMode_) swapModel(modelOpt_.promptTokenBatchSize);
        size_t digestedPromptTokens = 0;
        outputToken = digestPrompt(promptTokens, modelOpt_.promptTokenBatchSize, digestedPromptTokens);
        result.promptTokens = inputTokens.size();
        cachedFirstOutputToken_ = outputToken;
        hasCachedFirstOutputToken_ = true;
    }
    cachedPromptTokenIndex_ = mtk_llm_get_token_index(runtime_);

    if (modelOpt_.promptTokenBatchSize != modelOpt_.genTokenBatchSize) {
        swapModel(modelOpt_.genTokenBatchSize);
    }

    const size_t maxTokenLength = modelOpt_.maxTokenLength;
    utils::UTF8CharResolver utf8Resolver;
    std::string first = tokenizer_->detokenize(outputToken);
    result.text += first;
    if (onToken && !first.empty() && !onToken(first)) return result;

    size_t genTokCount = 0;
    while (genTokCount < maxTokens && mtk_llm_get_token_index(runtime_) < maxTokenLength) {
        if (!mtk_llm_is_enabled_cache_eviction(runtime_) && mtk_llm_get_token_index(runtime_) >= modelOpt_.cacheSize) {
            break;
        }
        SET_DUMP_INDEX(inferenceStep_++);
        outputToken = autoregressiveStep(outputToken);
        result.completionTokens++;
        genTokCount++;
        if (isStopToken(outputToken)) break;

        const std::string tokStr = tokenizer_->detokenize(outputToken);
        if (utf8Resolver.addBytes(tokStr)) {
            const auto resolved = utf8Resolver.getResolvedStr();
            result.text += resolved;
            if (onToken && !resolved.empty() && !onToken(resolved)) break;
        }
    }

    result.tokenIndex = mtk_llm_get_token_index(runtime_);
    LOG(INFO) << "llmserver generation done: prompt_tokens=" << result.promptTokens
              << ", completion_tokens=" << result.completionTokens << ", token_index=" << result.tokenIndex
              << ", response_len=" << result.text.size();
    hasActiveCache_ = keepCache;
    if (keepCache) {
        cachedPromptTokens_ = inputTokens;
    }
    return result;
}

void LlmEngine::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    resetLocked();
}

void LlmEngine::resetLocked() {
    if (!runtime_) return;
    mtk_llm_reset(runtime_);
    if (modelOpt_.promptTokenBatchSize != modelOpt_.genTokenBatchSize) {
        mtk_llm_swap_model(runtime_, modelOpt_.promptTokenBatchSize);
    }
    inGenMode_ = false;
    hasActiveCache_ = false;
    currentSessionId_.clear();
    cachedPromptTokens_.clear();
    cachedPromptTokenIndex_ = 0;
    cachedFirstOutputToken_ = 0;
    hasCachedFirstOutputToken_ = false;
}

void LlmEngine::release() {
    if (runtime_) {
        mtk_llm_release(runtime_);
        runtime_ = nullptr;
    }
    tokenizer_.reset();
    if (gSharedWeightsHandle) {
        mtk_llm_shared_weights_release_handle(gSharedWeightsHandle);
        gSharedWeightsHandle = nullptr;
    }
}

} // namespace llmserver
