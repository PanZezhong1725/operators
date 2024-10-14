#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t MAX_TILE_SIZE = 32 * 8;// 8 datablocks

template<typename T> class RoPE {
public:
    __aicore__ inline RoPE() {}
    // Init op
    // pos position vector
    // t input tensor
    // input tensor shape [nt, nh, dh]
    // make block_num = nh, tile_len = dh
    __aicore__ inline void Init(GM_ADDR t, GM_ADDR pos, GM_ADDR sin,
                                GM_ADDR cos, int32_t nt,
                                int32_t nh, int32_t dh);
    __aicore__ inline void Process();

private:
    // Copy a tile into UB
    __aicore__ inline void CopyIn(int32_t i, int32_t tno);
    __aicore__ inline void Compute(int32_t i, int32_t tno);
    __aicore__ inline void CopyOut(int32_t i, int32_t tno);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> sinQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> cosQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    TBuf<TPosition::VECCALC> tmpOddBuf;
    TBuf<TPosition::VECCALC> tmpEvenBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> tmp2Buf;
    TBuf<TPosition::VECCALC> tmp3Buf;
    TBuf<TPosition::VECCALC> tmp4Buf;
    // TBuf<TPosition::VECCALC> tmpOffsetBuf;
    // if (sizeof(float) != sizeof(T)) {
    TBuf<TPosition::VECCALC> tmpSinBuf;
    TBuf<TPosition::VECCALC> tmpCosBuf;
    // }

    GlobalTensor<T> xGm;
    GlobalTensor<uint64_t> pGm;
    GlobalTensor<float> sinGm;
    GlobalTensor<float> cosGm;
    GlobalTensor<T> oGm;

    // TODO: Change to uint64_t
    uint32_t _block_idx;
    uint32_t block_len;
    uint32_t tile_len;
    uint32_t tile_num_i;
    uint32_t tile_num_j;

    // t[nt, nh, dh]
    // nt num of tokens
    // nh num of heads
    // dh dimension of each head
    int32_t nt;
    int32_t nh;
    int32_t dh;
};

template<typename T>
__aicore__ inline void RoPE<T>::Init(GM_ADDR t, GM_ADDR pos, GM_ADDR sin,
                                     GM_ADDR cos, int32_t nt,
                                     int32_t nh, int32_t dh) {
    this->nt = nt;
    this->nh = nh;
    this->dh = dh;

    _block_idx = GetBlockIdx();
    block_len = static_cast<uint32_t>(nt * dh);
    tile_len = MAX_TILE_SIZE / sizeof(T);// 128 of fp16, 64 of fp32
    tile_num_i = nt;
    // TODO: deal remaining segmentation
    tile_num_j = dh / tile_len;

    // Init global buffer
    xGm.SetGlobalBuffer((__gm__ T *) t);
    pGm.SetGlobalBuffer((__gm__ uint64_t *) pos);
    sinGm.SetGlobalBuffer((__gm__ float *) sin);
    cosGm.SetGlobalBuffer((__gm__ float *) cos);
    oGm.SetGlobalBuffer((__gm__ T *) t);

    // Init Queue buffer
    pipe.InitBuffer(inQue, BUFFER_NUM, tile_len * sizeof(T));
    pipe.InitBuffer(outQue, BUFFER_NUM, tile_len * sizeof(T));
    pipe.InitBuffer(sinQue, BUFFER_NUM, tile_len * sizeof(float));
    pipe.InitBuffer(cosQue, BUFFER_NUM, tile_len * sizeof(float));
    pipe.InitBuffer(tmpOddBuf, tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmpEvenBuf, tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmpBuf, tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmp2Buf, tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmp3Buf, tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmp4Buf, tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmpSinBuf, tile_len * sizeof(T));
    pipe.InitBuffer(tmpCosBuf, tile_len * sizeof(T));
}

template<typename T>
__aicore__ inline void RoPE<T>::CopyIn(int32_t i, int32_t tno) {
    LocalTensor<T> inputUb = inQue.AllocTensor<T>();
    LocalTensor<float> sinUb = sinQue.AllocTensor<float>();
    LocalTensor<float> cosUb = cosQue.AllocTensor<float>();
    // Get idx of current tile in total input
    auto idx = i * nh * dh + _block_idx * dh;
    // Copy tile current tile into UB
    DataCopy(inputUb, xGm[idx + tno * tile_len], tile_len);
    // Copy sin cos tile
    auto pos_idx = pGm(i);
    // Cast sin cos to T type
    DataCopy(sinUb, sinGm[pos_idx * dh + tno * tile_len], tile_len);
    DataCopy(cosUb, cosGm[pos_idx * dh + tno * tile_len], tile_len);

    // Push in operands
    inQue.EnQue(inputUb);
    sinQue.EnQue(sinUb);
    cosQue.EnQue(cosUb);
}

template<typename T>
__aicore__ inline void RoPE<T>::Compute(int32_t i, int32_t tno) {
    LocalTensor<T> inputUb = inQue.DeQue<T>();
    LocalTensor<float> sinUb = sinQue.DeQue<float>();
    LocalTensor<float> cosUb = cosQue.DeQue<float>();
    LocalTensor<T> outUb = outQue.AllocTensor<T>();

    // Choose odd and even position
    LocalTensor<T> tmpOdd = tmpOddBuf.Get<T>();
    LocalTensor<T> tmpEven = tmpEvenBuf.Get<T>();
    LocalTensor<T> tmpUb = tmpBuf.Get<T>();
    LocalTensor<T> tmp2Ub = tmp2Buf.Get<T>();
    LocalTensor<T> tmp3Ub = tmp3Buf.Get<T>();
    LocalTensor<T> tmp4Ub = tmp4Buf.Get<T>();
    LocalTensor<T> tmpSinUb = tmpSinBuf.Get<T>();
    LocalTensor<T> tmpCosUb = tmpCosBuf.Get<T>();

    Cast<T, float>(tmpSinUb, sinUb, RoundMode::CAST_FLOOR, tile_len);
    Cast<T, float>(tmpCosUb, cosUb, RoundMode::CAST_FLOOR, tile_len);

    uint64_t rsvdCnt = 0;
    // Select odd & even numbers
    GatherMask<T>(tmpOdd, inputUb, 1, false, 0, {1, 1, 0, 0}, rsvdCnt);
    GatherMask<T>(tmpEven, inputUb, 2, false, 0, {1, 1, 0, 0}, rsvdCnt);

    // Calc odd position
    GatherMask<T>(tmpUb, tmpCosUb, 1, false, 0, {1, 1, 0, 0}, rsvdCnt);
    GatherMask<T>(tmp2Ub, tmpSinUb, 2, false, 0, {1, 1, 0, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    tmpUb = tmpOdd * tmpUb;
    tmp2Ub = tmpEven * tmp2Ub;
    PipeBarrier<PIPE_V>();
    tmpUb = tmpUb - tmp2Ub;

    // Calc even position
    GatherMask<T>(tmp3Ub, tmpSinUb, 1, false, 0, {1, 1, 0, 0}, rsvdCnt);
    GatherMask<T>(tmp4Ub, tmpCosUb, 2, false, 0, {1, 1, 0, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    tmp3Ub = tmpOdd * tmp3Ub;
    tmp4Ub = tmpEven * tmp4Ub;
    PipeBarrier<PIPE_V>();
    tmp3Ub = tmp3Ub + tmp4Ub;

    // // Scatter
    // Scatter<T>(outUb, tmpUb, tmpOffsetUb, (uint32_t)sizeof(T), tile_len / 2);
    for (uint32_t i = 0; i < tile_len / 2; i += 1) {
        outUb(i * 2 + 1) = tmp3Ub(i);
        outUb(i * 2) = tmpUb(i);
    }

    outQue.EnQue<T>(outUb);
    inQue.FreeTensor(inputUb);
    sinQue.FreeTensor(sinUb);
    cosQue.FreeTensor(cosUb);
}

template<typename T>
__aicore__ inline void RoPE<T>::CopyOut(int32_t i, int32_t tno) {
    LocalTensor<T> outUb = outQue.DeQue<T>();
    auto idx = i * nh * dh + _block_idx * dh;
    DataCopy(oGm[idx + tno * tile_len], outUb, tile_len);
    outQue.FreeTensor(outUb);
}

template<typename T> __aicore__ inline void RoPE<T>::Process() {
    
    for (int32_t i = 0; i < nt; ++i) {
        for (int32_t j = 0; j < dh / tile_len; j++) {
            CopyIn(i, j);
            Compute(i, j);
            CopyOut(i, j);
        }
    }
}

// Kernel func
extern "C" __global__ __aicore__ void rope_kernel_fp16(GM_ADDR t, GM_ADDR pos,
                                                       GM_ADDR sin, GM_ADDR cos,
                                                       int32_t nt, int32_t nh, int32_t dh) {
    RoPE<half> op;
    op.Init(t, pos, sin, cos, nt, nh, dh);
    op.Process();
}

extern "C" void rope_kernel_do(void *t, void *pos, void *sin, void *cos,
                               int32_t nt, int32_t nh, int32_t dh,
                               int dtype, void *stream) {
    switch (dtype) {
        case 1:// ACL_FLOAT16
            rope_kernel_fp16<<<nh, nullptr, stream>>>(t, pos, sin, cos, nt, nh, dh);
            break;
        default:
            break;
    }
}
