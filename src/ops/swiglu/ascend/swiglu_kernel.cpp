#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t BLOCK_NUM = 8;
constexpr int32_t MAX_TILE_SIZE = 32 * 8;// 8 * 32 bytes


template<typename T> class KernelSwiGLU {
public:
    __aicore__ inline KernelSwiGLU() {}
    // Init SwiGLU
    // c output tensor, support only 2 dim
    // a up tensor
    // b gate tensor
    // formular: b = a x silu(b)
    // a, b, c has same tensor shape 
    __aicore__ inline void Init(GM_ADDR c, GM_ADDR a, GM_ADDR b,
                                float beta, int32_t nt, int32_t dh,
                                int32_t sta, int32_t stb, int32_t stc);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t i, int32_t tno);
    __aicore__ inline void Compute(int32_t i, int32_t tno);
    __aicore__ inline void CopyOut(int32_t i, int32_t tno);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> aQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> bQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> cQue;

    GlobalTensor<T> aGm;
    GlobalTensor<T> bGm;
    GlobalTensor<T> cGm;

    uint32_t _block_idx;
    uint32_t block_len;
    uint32_t tile_len;
    uint32_t tile_num_i;
    uint32_t tile_num_j;

    // c[nt, dh]
    // strides = [st, 1]

    int32_t nt;
    int32_t dh;
    float beta;
    int32_t sta;
    int32_t stb;
    int32_t stc;
};


template<typename T>
__aicore__ inline void KernelSwiGLU<T>::Init(GM_ADDR c, GM_ADDR a, GM_ADDR b,
                                             float beta, int32_t nt, int32_t dh,
                                             int32_t sta, int32_t stb, int32_t stc) {

    this->nt = nt;
    this->dh = dh;
    this->beta = beta;
    this->sta = sta;
    this->stb = stb;
    this->stc = stc;

    _block_idx = GetBlockIdx();
    block_len = static_cast<uint32_t>(dh / BLOCK_NUM * nt);
    tile_len = MAX_TILE_SIZE / sizeof(T);// 128 of fp16 and 64 of fp32
    tile_num_i = nt;
    tile_num_j = dh / BLOCK_NUM / tile_len;

    // Set global tensor
    aGm.SetGlobalBuffer((__gm__ T *) a);
    bGm.SetGlobalBuffer((__gm__ T *) b);
    cGm.SetGlobalBuffer((__gm__ T *) c);

    // Pipe alloc memory to queue, the unit is bytes
    pipe.InitBuffer(aQue, BUFFER_NUM, tile_len * sizeof(T));
    pipe.InitBuffer(bQue, BUFFER_NUM, tile_len * sizeof(T));
    pipe.InitBuffer(cQue, BUFFER_NUM, tile_len * sizeof(T));
}

template<typename T>
__aicore__ inline void KernelSwiGLU<T>::CopyIn(int32_t i, int32_t tno) {
    // Alloc tensor from queue memory
    LocalTensor<T> aUb = aQue.AllocTensor<T>();
    LocalTensor<T> bUb = bQue.AllocTensor<T>();
    // Get idx of current tile
    auto idxa = i * sta + _block_idx * dh / BLOCK_NUM;
    auto idxb = i * stb + _block_idx * dh / BLOCK_NUM;
    // Copy process_th tile from global tensor to local tensor
    DataCopy(aUb, aGm[idxa + tno * tile_len], tile_len);
    DataCopy(bUb, bGm[idxb + tno * tile_len], tile_len);

    // if (i == 0 && _block_idx == 0) {
    //     DumpTensor(aUb, 1, tile_len);
    //     DumpTensor(bUb, 2, tile_len);
    // }

    // Enque input tensor to VECIN queue
    aQue.EnQue(aUb);
    bQue.EnQue(bUb);
}

template<typename T>
__aicore__ inline void KernelSwiGLU<T>::Compute(int32_t i, int32_t tno) {
    // Deque input tensors from VECIN queue
    LocalTensor<T> aUb = aQue.DeQue<T>();
    LocalTensor<T> bUb = bQue.DeQue<T>();
    LocalTensor<T> cUb = cQue.AllocTensor<T>();

    SwiGLU<T, false>(cUb, aUb, bUb, beta);

    cQue.EnQue<T>(cUb);
    aQue.FreeTensor(aUb);
    bQue.FreeTensor(bUb);
}

template<typename T>
__aicore__ inline void KernelSwiGLU<T>::CopyOut(int32_t i, int32_t tno) {
    // Deque output tensor from VECOUT queue
    LocalTensor<T> cUb = cQue.DeQue<T>();
    auto idxc = i * stc + _block_idx * dh / BLOCK_NUM;
    // Copy progress_th tile from local tensor to global tensor
    DataCopy(cGm[idxc + tno * tile_len], cUb, tile_len);
    // Free output Local tensor
    cQue.FreeTensor(cUb);
}

template<typename T>
__aicore__ inline void KernelSwiGLU<T>::Process() {
    for (int32_t i = 0; i < tile_num_i; ++i) {
        for (int32_t j = 0; j < tile_num_j; ++j) {
            CopyIn(i, j);
            Compute(i, j);
            CopyOut(i, j);
        }
    }
}

extern "C" __global__ __aicore__ void swiglu_kernel_f16(GM_ADDR c, GM_ADDR a, GM_ADDR b,
                                                        float beta, int32_t nt, int32_t dh,
                                                        int32_t sta, int32_t stb, int32_t stc) {
    KernelSwiGLU<half> op;
    op.Init(c, a, b, beta, nt, dh, sta, stb, stc);
    op.Process();
}

extern "C" __global__ __aicore__ void swiglu_kernel_f32(GM_ADDR c, GM_ADDR a, GM_ADDR b,
                                                        float beta, int32_t nt, int32_t dh,
                                                        int32_t sta, int32_t stb, int32_t stc) {
    KernelSwiGLU<float> op;
    op.Init(c, a, b, beta, nt, dh, sta, stb, stc);
    op.Process();
}

extern "C" void swiglu_kernel_do(void *c, void *a, void *b,
                                 float beta, int32_t nt, int32_t dh,
                                 int32_t sta, int32_t stb, int32_t stc,
                                 int dtype, void *stream) {


    switch (dtype) {
        case 0:
            swiglu_kernel_f32<<<BLOCK_NUM, nullptr, stream>>>(
                c, a, b, beta, nt, dh, sta, stb, stc);
        case 1:
            swiglu_kernel_f16<<<BLOCK_NUM, nullptr, stream>>>(
                c, a, b, beta, nt, dh, sta, stb, stc);
    }
    return;
}
