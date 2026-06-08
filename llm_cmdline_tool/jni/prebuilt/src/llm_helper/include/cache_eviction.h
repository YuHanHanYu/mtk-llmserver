#pragma once

#include <algorithm>
#include <array>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace mtk::llm_helper {

struct KVCacheInfo {
    char* buffer = nullptr;
    std::array<uint32_t, 4> shape;
    size_t typeSize = 0;

    size_t strideSize() const { return shape[3] * typeSize; }
    size_t numRows() const { return shape[0] * shape[1]; }
};

class CacheEvictionAgent {
public:
    explicit CacheEvictionAgent(const std::vector<KVCacheInfo>& kvCaches);

    virtual ~CacheEvictionAgent();

    // Update LLM Context
    void updateContext(const size_t tokenSize, const size_t cacheSize, const size_t numCachedTokens,
                       const size_t leftPadSize = 0, const size_t rightPadSize = 0);

    // Update KV Cache buffers
    void updateCacheBuffers(const std::vector<char*>& cacheBuffers);

    // Update Attention Weights
    void updateAttnWeights(const std::vector<std::string_view>& attnWeights);

    // Run the cache eviction algorithm
    virtual void run() = 0;

    // Reset the cache eviction algorithm state
    virtual void reset() = 0;

    // Evict KV Cache from relative positions of the cached tokens.
    // For example, given cache size of 5 and 3 cached tokens, evict relative token position 1:
    //   Cached token pos  :          0   1   2
    //   Cache token slots : [ ] [ ] [C] [C] [C]
    //   Relative evict pos:              1
    void evictCacheRel(const std::vector<size_t>& relativePositions);

    // Evict KV Cache from absolute positions of the entire cache.
    // For example, given cache size of 5 and 3 cached tokens, evict absolute token position 3:
    //   Full cache pos    :  0   1   2   3   4
    //   Cache token slots : [ ] [ ] [C] [C] [C]
    //   Absolute evict pos:              3
    void evictCacheAbs(const std::vector<size_t>& absolutePositions);

protected:
    // KV Cache
    std::vector<KVCacheInfo> mKvCaches;

    // Attention Weights
    std::vector<std::string_view> mAttnWeightsBuffers;

    // General Context
    size_t mTokenSize = 0;
    size_t mCacheSize = 0;
    size_t mNumCachedTokens = 0;
    size_t mLeftPadSize = 0;
    size_t mRightPadSize = 0;
};

namespace {
class LocalSnapKVContext;
}

class LocalSnapKVAgent : public CacheEvictionAgent {
public:
    explicit LocalSnapKVAgent(const std::vector<KVCacheInfo>& kvCaches, const size_t attnSinkSize,
                              const size_t numAttnHeads);

    ~LocalSnapKVAgent() override;

    // Run LocalSnapKV main algorithm
    void run() override;

    void reset() override;

private:
    std::unique_ptr<LocalSnapKVContext> mCtx;
};

} // namespace mtk::llm_helper
