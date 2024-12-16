#include "../../../../include/status.h"
#include "kernel_operator.h"

using namespace AscendC;

template<typename T>
class KernelRandomSample {
public:
    __aicore__ inline KernelRandomSample() {}
    __aicore__ inline void Init(GM_ADDR p, GM_ADDR res, GM_ADDR topkAddr,
                                GM_ADDR topkIdxAddr, int32_t topk_, int32_t voc_,
                                float topp_, float temper_, float random_) {

        topk = topk_;
        voc = voc_;
        topp = topp_;
        temperature = temper_;
        random = random_;
        blockSize = 256 * 2;

        // CumSumInfo
        if (sizeof(T) == sizeof(float)) {
            topkAligned = (topk + 7) / 8 * 8;
            vocAligned = (voc + 7) / 8 * 8;
        } else {
            topkAligned = (topk + 15) / 16 * 16;
            vocAligned = (voc + 15) / 16 * 16;
        }
        topkIdxAligned = (topk + 3) / 4 * 4;

        // Set Gm
        pGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(p), voc);
        topkGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(topkAddr), topk);
        topkIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(topkIdxAddr), topk);
        resGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(res), 1);

        // Global input and output
        pipe.InitBuffer(pQue, 1, vocAligned * sizeof(T));
        pipe.InitBuffer(topkQue, 1, topkAligned * sizeof(T));
        pipe.InitBuffer(topkIdxQue, 1, topkIdxAligned * sizeof(int64_t));
        pipe.InitBuffer(resQue, 1, 32);// 32 bytes for aligned

        pipe.InitBuffer(softMaxBuf1, blockSize);
        pipe.InitBuffer(softMaxBuf2, blockSize);
        pipe.InitBuffer(softMaxBuf3, blockSize);
        pipe.InitBuffer(softMaxOutBuf, topkAligned * sizeof(T));

        pipe.InitBuffer(inclusiveSumOutBuf, topkAligned * sizeof(T));
    }
    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    // Softmax
    __aicore__ inline void SoftMax(LocalTensor<T> &valIn,
                                   LocalTensor<T> &topkValIn,
                                   LocalTensor<T> &softMaxOut) {
        int32_t repeatTimes = vocAligned * sizeof(T) / blockSize;
        int32_t remainder = vocAligned * sizeof(T) % blockSize / sizeof(T);
        int32_t tileLength = blockSize / sizeof(T);
        float negMax = -static_cast<float>(topkValIn(0));
        float invTemperature = 1.0f / temperature;
        float sum = 0.f;
        float sum_s = 0.f;
        LocalTensor<T> tmpBuffer = softMaxBuf1.Get<T>();
        LocalTensor<T> tmpBuffer2 = softMaxBuf2.Get<T>();
        LocalTensor<T> tmpBuffer3 = softMaxBuf3.Get<T>();
        for (int32_t i = 0; i < repeatTimes; i++) {
            Adds(tmpBuffer, valIn[i * tileLength], static_cast<T>(negMax), tileLength);
            Muls(tmpBuffer2, tmpBuffer, static_cast<T>(invTemperature), tileLength);
            Exp(tmpBuffer3, tmpBuffer2, tileLength);
            sum_s = 0.f;
            for (int j = 0; j < tileLength; ++j) {
                sum_s += static_cast<float>(tmpBuffer3(j));
            }
            sum += sum_s;
        }
        if (remainder != 0) {
            Adds(tmpBuffer, valIn[repeatTimes * tileLength], static_cast<T>(negMax), remainder);
            Muls(tmpBuffer2, tmpBuffer, static_cast<T>(invTemperature), remainder);
            Exp(tmpBuffer3, tmpBuffer2, remainder);
            sum_s = 0.f;
            for (int i = 0; i < remainder; ++i) {
                sum_s += static_cast<float>(tmpBuffer3(i));
            }
            sum += sum_s;
        }
        float invSum = 1.0f / sum;
        Adds(tmpBuffer, topkValIn, static_cast<T>(negMax), topk);
        Muls(tmpBuffer2, tmpBuffer, static_cast<T>(invTemperature), topk);
        Exp(tmpBuffer3, tmpBuffer2, topk);
        Muls(softMaxOut, tmpBuffer3, static_cast<T>(invSum), topk);
    }

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
        LocalTensor<T> pLocal = pQue.AllocTensor<T>();
        LocalTensor<T> topkValLocal = topkQue.AllocTensor<T>();
        LocalTensor<int64_t> topkIdxLocal = topkIdxQue.AllocTensor<int64_t>();

        DataCopy(pLocal, pGm, vocAligned);
        DataCopy(topkValLocal, topkGm, topkAligned);
        DataCopy(topkIdxLocal, topkIdxGm, topkIdxAligned);

        pQue.EnQue(pLocal);
        topkQue.EnQue(topkValLocal);
        topkIdxQue.EnQue(topkIdxLocal);
    }

    __aicore__ inline void Compute() {
        // Get input data
        LocalTensor<T> pLocal = pQue.DeQue<T>();
        LocalTensor<T> topkValLocal = topkQue.DeQue<T>();

        // SoftMax
        LocalTensor<T> softMaxOutLocal = softMaxOutBuf.Get<T>();
        SoftMax(pLocal, topkValLocal, softMaxOutLocal);

        // InclusiveSum
        LocalTensor<T> inclusiveOutLocal = inclusiveSumOutBuf.Get<T>();
        InclusiveSum(softMaxOutLocal, inclusiveOutLocal);

        // randomSample
        LocalTensor<int64_t> topkIdxLocal = topkIdxQue.DeQue<int64_t>();
        LocalTensor<int64_t> resultLocal = resQue.AllocTensor<int64_t>();
        RandomSample(inclusiveOutLocal, topkIdxLocal, resultLocal);

        pQue.FreeTensor(pLocal);
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
    GlobalTensor<T> pGm;
    GlobalTensor<T> topkGm;
    GlobalTensor<int64_t> topkIdxGm;
    GlobalTensor<int64_t> resGm;

    TPipe pipe;

    TQue<QuePosition::VECIN, 1> pQue;
    TQue<QuePosition::VECIN, 1> topkQue;
    TQue<QuePosition::VECIN, 1> topkIdxQue;
    TQue<QuePosition::VECOUT, 1> resQue;

    TBuf<TPosition::VECCALC> softMaxBuf1;
    TBuf<TPosition::VECCALC> softMaxBuf2;
    TBuf<TPosition::VECCALC> softMaxBuf3;
    TBuf<TPosition::VECCALC> softMaxOutBuf;

    TBuf<TPosition::VECCALC> inclusiveSumOutBuf;

    // Kernel params
    int32_t topk;
    int32_t voc;
    float topp;
    float temperature;
    float random;

    int32_t topkAligned;
    int32_t topkIdxAligned;
    int32_t vocAligned;
    int32_t blockSize;
};

extern "C" __global__ __aicore__ void
random_sample_kernel_f16(GM_ADDR p, GM_ADDR res, GM_ADDR topkAddr,
                         GM_ADDR topkIdxAddr, int32_t topk_, int32_t voc_,
                         float topp_, float temper_, float random_) {
    KernelRandomSample<half> op;
    op.Init(p, res, topkAddr, topkIdxAddr, topk_, voc_, topp_, temper_, random_);
    op.Process();
}

extern "C" infiniopStatus_t
random_sample_do(void *p, void *res, void *topkAddr, void *topkIdxAddr,
                 int32_t topk, int32_t voc, float topp, float temper,
                 float random, int dtype, void *stream) {

    switch (dtype) {
        case 0:
            return STATUS_SUCCESS;
        case 1:
            random_sample_kernel_f16<<<1, nullptr, stream>>>(
                p, res, topkAddr, topkIdxAddr, topk, voc, topp, temper, random);
            return STATUS_SUCCESS;
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
