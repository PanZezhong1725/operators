#include "topK_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <algorithm>
#include <queue>
#include <vector>
#include <iostream>

struct Compare {
    bool operator()(const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.first > b.first;
    }
};

void topK_cpu(Tensor indices, Tensor probs, Tensor logits, int64_t k) {
    ASSERT(k > 0);

    int *indexData = static_cast<int *>(indices.data);
    int *probData = static_cast<int *>(probs.data);
    int *logitData = static_cast<int *>(logits.data);

    uint64_t ndim = indices.layout->ndim;
    uint64_t *shape = indices.layout->shape;
    int64_t *strides = indices.layout->strides;

    uint64_t totalElements = 1;
    for (uint64_t i = 0; i < ndim; ++i) {
        totalElements *= shape[i];
    }
    ASSERT(k < totalElements);
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, Compare> maxHeap;

    for (uint64_t i = 0; i < k && i < totalElements; ++i) {
        maxHeap.push(std::make_pair(indexData[i], i));
    }

    for (uint64_t i = k; i < totalElements; ++i) {
        if (indexData[i] > maxHeap.top().first) {
            maxHeap.pop();
            maxHeap.push(std::make_pair(indexData[i], i));
        }
    }

    while (!maxHeap.empty()) {
        probData[--k] = maxHeap.top().first;
        logitData[--k] = maxHeap.top().second;
        maxHeap.pop();
    }
}
