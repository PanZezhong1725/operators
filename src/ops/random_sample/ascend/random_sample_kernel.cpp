#include "../../../../include/status.h"
#include "kernel_operator.h"
#include <cstdio>

using namespace AscendC;

template<typename T>
class KernelSoftmaxWithT {
private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> pQue;
    TQue<TPosition::VECIN, 1> topkQue;
    TQue<TPosition::VECOUT, 1> resQue;

    TBuf<TPosition::VECCALC> softMaxBuf1;
    TBuf<TPosition::VECCALC> softMaxBuf2;
    TBuf<TPosition::VECCALC> softMaxBuf3;

    GlobalTensor<T> pGm;
    GlobalTensor<T> topkGm;
    GlobalTensor<T> resGm;
    GlobalTensor<float> workspaceGm;

    int32_t topk;
    int32_t voc;
    float temper;
    float max_p;

    int64_t _block_idx;
    int64_t _num_blocks;
    int32_t _base;
    int32_t _remainder;
    int32_t _tile_len;
    int32_t _copy_len;

    int32_t _topk_base;
    int32_t _topk_remainder;
    int32_t _topk_tile_len;
    int32_t _topk_copy_len;

private:
    __aicore__ inline GlobalTensor<T> GetTilingGm(GlobalTensor<T> &gm,
                                                  int32_t remainder,
                                                  int32_t tile_len) {
        return _block_idx < remainder
                   ? gm[tile_len * _block_idx]
                   : gm[tile_len * _block_idx + remainder];
    }

public:
    __aicore__ inline KernelSoftmaxWithT() {}
    __aicore__ inline void Init(GM_ADDR p, GM_ADDR topkp, GM_ADDR res, GM_ADDR workspace,
                                int32_t topk_, int32_t voc_, float temper_) {
        topk = topk_;
        voc = voc_;
        temper = temper_;

        // Tiling input p
        _block_idx = GetBlockIdx();
        _num_blocks = GetBlockNum();
        _base = voc_ / _num_blocks;
        _remainder = voc_ % _num_blocks;
        _tile_len = _block_idx < _remainder ? _base + 1 : _base;
        // Length in DataCopy should be 32B aligned
        _copy_len = _tile_len * sizeof(T) % 32 == 0
                        ? _tile_len
                        : (_tile_len * sizeof(T) + 31) / 32 * 32 / sizeof(T);

        // Tiling topkp and res
        _topk_base = topk_ / _num_blocks;
        _topk_remainder = topk_ % _num_blocks;
        _topk_tile_len = _block_idx < _topk_remainder ? _topk_base + 1 : _topk_base;
        _topk_copy_len = _topk_tile_len * sizeof(T) % 32 == 0
                             ? _topk_tile_len
                             : (_topk_tile_len * sizeof(T) + 31) / 32 * 32 / sizeof(T);

        pGm.SetGlobalBuffer((__gm__ T *) p);
        topkGm.SetGlobalBuffer((__gm__ T *) topkp);
        resGm.SetGlobalBuffer((__gm__ T *) res);

        // Get max value in p
        max_p = static_cast<float>(topkGm(0));

        // Set workspace for core sync
        workspaceGm.SetGlobalBuffer((__gm__ float *) (workspace), _num_blocks);

        // Init Buffer for I/O
        pipe.InitBuffer(pQue, 1, _copy_len * sizeof(T));
        pipe.InitBuffer(topkQue, 1, _topk_copy_len * sizeof(T));
        pipe.InitBuffer(resQue, 1, _topk_copy_len * sizeof(T));

        // Init Buffer for softmax compute
        pipe.InitBuffer(softMaxBuf1, _tile_len * sizeof(T));
        pipe.InitBuffer(softMaxBuf2, _tile_len * sizeof(T));
        pipe.InitBuffer(softMaxBuf3, _tile_len * sizeof(T));

        printf("Init kernel softmax\n");
    }
    __aicore__ inline void CopyIn() {
        // Alloc Buffer for input
        LocalTensor<T> pLocal = pQue.AllocTensor<T>();
        LocalTensor<T> topkLocal = topkQue.AllocTensor<T>();
        // Get current core's Gm
        GlobalTensor<T> tilepGm = GetTilingGm(pGm, _remainder, _tile_len);
        GlobalTensor<T> tileTopkGm = GetTilingGm(topkGm, _topk_remainder, _topk_tile_len);
        // Copy from global mem
        // _copy_len and _topk_copy_len are 32B aligned
        DataCopy(pLocal, tilepGm, _copy_len);
        DataCopy(topkLocal, tileTopkGm, _topk_copy_len);
        // Enque
        pQue.EnQue(pLocal);
        topkQue.EnQue(topkLocal);
        pQue.FreeTensor(pLocal);
        topkQue.FreeTensor(topkLocal);
    }
    __aicore__ inline void Compute() {
        LocalTensor<T> pLocal = pQue.DeQue<T>();
        float negMax = -max_p;
        float invTemperature = 1.0f / temper;
        float sum_s = 0.f;
        float sum = 0.f;
        // Compute sum((p - max) / t)
        LocalTensor<T> tmpBuffer = softMaxBuf1.Get<T>();
        LocalTensor<T> tmpBuffer2 = softMaxBuf2.Get<T>();
        LocalTensor<T> tmpBuffer3 = softMaxBuf3.Get<T>();
        Adds(tmpBuffer, pLocal, static_cast<T>(negMax), _tile_len);
        Muls(tmpBuffer2, tmpBuffer, static_cast<T>(invTemperature), _tile_len);
        Exp(tmpBuffer3, tmpBuffer2, _tile_len);
        for (int j = 0; j < _tile_len; ++j) {
            sum_s += static_cast<float>(tmpBuffer3(j));
        }
        // Set value to workspace
        workspaceGm.SetValue(GetBlockIdx(), sum_s);
        SyncAll();
        // Compute total sum
        for (int32_t j = 0; j < _num_blocks; j++) {
            sum += static_cast<float>(workspaceGm(j));
        }
        // Compute softmax
        LocalTensor<T> topkLocal = topkQue.DeQue<T>();
        LocalTensor<T> softMaxOutLocal = resQue.AllocTensor<T>();
        float invSum = 1.0f / sum;
        Adds(tmpBuffer, topkLocal, static_cast<T>(negMax), _topk_tile_len);
        Muls(tmpBuffer2, tmpBuffer, static_cast<T>(invTemperature), _topk_tile_len);
        Exp(tmpBuffer3, tmpBuffer2, _topk_tile_len);
        Muls(softMaxOutLocal, tmpBuffer3, static_cast<T>(invSum), _topk_tile_len);
        // Enque
        resQue.EnQue(softMaxOutLocal);
        pQue.FreeTensor(pLocal);
        topkQue.FreeTensor(topkLocal);
    }
    __aicore__ inline void CopyOut() {
        LocalTensor<T> softMaxLocal = resQue.DeQue<T>();
        GlobalTensor<T> outGm = GetTilingGm(resGm, _topk_remainder, _topk_tile_len);
        if (_topk_tile_len * sizeof(T) % 32 != 0) {
            DataCopyExtParams dcep = {1, static_cast<uint32_t>(_topk_tile_len * sizeof(T)), 0, 0, 0};
            DataCopyPad(outGm, softMaxLocal, dcep);
        } else {
            DataCopy(outGm, softMaxLocal, _topk_tile_len);
        }
        resQue.FreeTensor(softMaxLocal);
    }
    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }
};

template<typename T>
class KernelRandomSample {
public:
    __aicore__ inline KernelRandomSample() {}
    __aicore__ inline void Init(GM_ADDR topkAddr, GM_ADDR topkIdxAddr, GM_ADDR res,
                                int32_t topk_, float topp_, float random_) {

        topk = topk_;
        topp = topp_;
        random = random_;

        // Align topk & topkIdx to 32B
        topkAligned = topk * sizeof(T) % 32 == 0
                          ? topk
                          : (topk * sizeof(T) + 31) / 32 * 32 / sizeof(T);
        topkIdxAligned = (topk + 3) / 4 * 4;

        // Set Gm
        topkGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(topkAddr), topk);
        topkIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(topkIdxAddr), topk);
        resGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(res), 1);

        // Global input and output
        pipe.InitBuffer(topkQue, 1, topkAligned * sizeof(T));
        pipe.InitBuffer(topkIdxQue, 1, topkIdxAligned * sizeof(int64_t));
        pipe.InitBuffer(resQue, 1, 32);// 32 bytes for aligned

        pipe.InitBuffer(inclusiveSumOutBuf, topkAligned * sizeof(T));
    }
    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    // Cumsum
    __aicore__ inline void InclusiveSum(LocalTensor<T> &topkValIn,
                                        LocalTensor<T> &topkValOut) {
        static constexpr CumSumConfig cumSumConfig{true, false, false};
        LocalTensor<T> lastRowLocal;
        CumSum<T, cumSumConfig>(topkValOut, lastRowLocal, topkValIn,
                                {1, static_cast<uint32_t>(topkAligned)});
    }

    // Random sample
    __aicore__ inline void RandomSample(LocalTensor<T> &valIn,
                                        LocalTensor<int64_t> &Index,
                                        LocalTensor<int64_t> &result) {
        int end = 0;
        for (end = 0; end < topk; end++) {
            if (static_cast<float>(valIn(end)) >= topp) {
                break;
            }
        }
        if (end < topk - 1) {
            end += 1;
        } else {
            end = topk;
        }

        auto randomVal = random * static_cast<float>(valIn(end - 1));
        for (int i = 0; i < end; i++) {
            if (randomVal < static_cast<float>(valIn(i))) {
                result(0) = Index(i);
                return;
            }
        }
        result(0) = Index(end - 1);
    }

    __aicore__ inline void CopyIn() {
        LocalTensor<T> topkValLocal = topkQue.AllocTensor<T>();
        LocalTensor<int64_t> topkIdxLocal = topkIdxQue.AllocTensor<int64_t>();

        DataCopy(topkValLocal, topkGm, topkAligned);
        DataCopy(topkIdxLocal, topkIdxGm, topkIdxAligned);

        topkQue.EnQue(topkValLocal);
        topkIdxQue.EnQue(topkIdxLocal);
    }

    __aicore__ inline void Compute() {
        // Deque softmax res
        LocalTensor<T> topkValLocal = topkQue.DeQue<T>();

        // InclusiveSum
        LocalTensor<T> inclusiveOutLocal = inclusiveSumOutBuf.Get<T>();
        InclusiveSum(topkValLocal, inclusiveOutLocal);

        // randomSample
        LocalTensor<int64_t> topkIdxLocal = topkIdxQue.DeQue<int64_t>();
        LocalTensor<int64_t> resultLocal = resQue.AllocTensor<int64_t>();
        RandomSample(inclusiveOutLocal, topkIdxLocal, resultLocal);

        topkQue.FreeTensor(topkValLocal);
        topkIdxQue.FreeTensor(topkIdxLocal);
        resQue.EnQue(resultLocal);
    }
    __aicore__ inline void CopyOut() {
        LocalTensor<int64_t> resLocal = resQue.DeQue<int64_t>();
        DataCopy(resGm, resLocal, 32 / sizeof(int64_t));
        resQue.FreeTensor(resLocal);
    }

private:
    GlobalTensor<T> topkGm;
    GlobalTensor<int64_t> topkIdxGm;
    GlobalTensor<int64_t> resGm;

    TPipe pipe;

    TQue<QuePosition::VECIN, 1> topkQue;
    TQue<QuePosition::VECIN, 1> topkIdxQue;
    TQue<QuePosition::VECOUT, 1> resQue;

    TBuf<TPosition::VECCALC> inclusiveSumOutBuf;

    // Kernel params
    int32_t topk;
    int32_t topkAligned;
    int32_t topkIdxAligned;
    float topp;
    float random;
};

extern "C" __global__ __aicore__ void
random_sample_kernel_f16(GM_ADDR p, GM_ADDR res, GM_ADDR topkAddr,
                         GM_ADDR topkIdxAddr, GM_ADDR workspace, int32_t topk_,
                         int32_t voc_, float topp_, float temper_, float random_) {
    KernelSoftmaxWithT<half> kernel_softmax;
    GM_ADDR softmaxOutAddr = workspace;
    GM_ADDR softmaxWorkspace = workspace + topk_ * sizeof(half);
    kernel_softmax.Init(p, topkAddr, softmaxOutAddr, softmaxWorkspace, topk_, voc_, temper_);
    kernel_softmax.Process();
    KernelRandomSample<half> kernel_random_sample;
    kernel_random_sample.Init(softmaxOutAddr, topkIdxAddr, res, topk_, topp_, random_);
    kernel_random_sample.Process();
}

extern "C" infiniopStatus_t
random_sample_do(void *p, void *res, void *topkAddr, void *topkIdxAddr,
                 void *workspace, int32_t topk, int32_t voc,
                 float topp, float temper, float random,
                 int dtype, void *stream) {

    printf(" random sample do\n");
    switch (dtype) {
        case 0:
            return STATUS_SUCCESS;
        case 1: {
            // auto softmaxOut = workspace;
            // auto softmaxWorkspace = (void *) ((uint8_t *) workspace + topk * sizeof(half));
            // softmax_with_t_fp16<<<32, nullptr, stream>>>(
            //     p, topkAddr, softmaxOut, softmaxWorkspace, topk, voc, temper);
            // random_sample_kernel_f16<<<1, nullptr, stream>>>(
            //     softmaxOut, topkIdxAddr, res, topk, topp, random);
            random_sample_kernel_f16<<<8, nullptr, stream>>>(
                p, res, topkAddr, topkIdxAddr, workspace, topk, voc, topp, temper, random);
            return STATUS_SUCCESS;
        }
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
