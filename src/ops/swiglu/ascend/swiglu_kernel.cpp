#include "../../../../include/status.h"
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t BLOCK_NUM = 8;

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
                                int32_t sta, int32_t stb, int32_t stc,
                                uint32_t remainder, uint32_t base);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t i);
    __aicore__ inline void Compute(int32_t i);
    __aicore__ inline void CopyOut(int32_t i);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> aQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> bQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> cQue;
    // Used in GatherMask
    // TBuf<TPosition::VECCALC> outBuf;

    GlobalTensor<T> aGm;
    GlobalTensor<T> bGm;
    GlobalTensor<T> cGm;

    uint32_t _block_idx;
    uint32_t _tile_len;
    uint32_t _copy_len;

    // c[nt, dh]
    // strides = [stx, 1]
    int32_t nt;
    int32_t dh;
    int32_t sta;
    int32_t stb;
    int32_t stc;
    float beta;
};


template<typename T>
__aicore__ inline void KernelSwiGLU<T>::Init(GM_ADDR c, GM_ADDR a, GM_ADDR b,
                                             float beta, int32_t nt, int32_t dh,
                                             int32_t sta, int32_t stb, int32_t stc,
                                             uint32_t remainder, uint32_t base) {

    this->nt = nt;
    this->dh = dh;
    this->beta = beta;
    this->sta = sta;
    this->stb = stb;
    this->stc = stc;

    _block_idx = GetBlockIdx();
    _tile_len = _block_idx < remainder ? base + 1 : base;
    _copy_len = _tile_len * sizeof(T) % 32 == 0
                    ? _tile_len
                    : (_tile_len * sizeof(T) + 31) / 32 * 32 / sizeof(T);

    // Set global tensor
    aGm.SetGlobalBuffer((__gm__ T *) a);
    bGm.SetGlobalBuffer((__gm__ T *) b);
    cGm.SetGlobalBuffer((__gm__ T *) c);

    // Pipe alloc memory to queue, the unit is bytes
    pipe.InitBuffer(aQue, BUFFER_NUM, _copy_len * sizeof(T));
    pipe.InitBuffer(bQue, BUFFER_NUM, _copy_len * sizeof(T));
    pipe.InitBuffer(cQue, BUFFER_NUM, _copy_len * sizeof(T));
}

template<typename T>
__aicore__ inline void KernelSwiGLU<T>::CopyIn(int32_t i) {
    // Alloc tensor from queue memory
    LocalTensor<T> aUb = aQue.AllocTensor<T>();
    LocalTensor<T> bUb = bQue.AllocTensor<T>();
    // Get idx of current tile
    auto idxa = i * sta + _block_idx * _tile_len;
    auto idxb = i * stb + _block_idx * _tile_len;
    // Copy process_th tile from global tensor to local tensor
    // See https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/opdevgapi/atlasascendc_api_07_0105.html
    // DataCopy cut down if _tile_len * sizeof(T) / 32 != 0
    DataCopy(aUb, aGm[idxa], _copy_len);
    DataCopy(bUb, bGm[idxb], _copy_len);

    // Enque input tensor to VECIN queue
    aQue.EnQue(aUb);
    bQue.EnQue(bUb);
}

template<typename T>
__aicore__ inline void KernelSwiGLU<T>::Compute(int32_t i) {
    // Deque input tensors from VECIN queue
    LocalTensor<T> aUb = aQue.DeQue<T>();
    LocalTensor<T> bUb = bQue.DeQue<T>();
    LocalTensor<T> cUb = cQue.AllocTensor<T>();
    // Call SwiGLU ascend api
    SwiGLU<T, false>(cUb, aUb, bUb, beta);
    // Enque result and free input
    cQue.EnQue<T>(cUb);
    aQue.FreeTensor(aUb);
    bQue.FreeTensor(bUb);
}

template<typename T>
__aicore__ inline void KernelSwiGLU<T>::CopyOut(int32_t i) {
    // Deque output tensor from VECOUT queue
    LocalTensor<T> cUb = cQue.DeQue<T>();
    auto idxc = i * stc + _block_idx * _tile_len;
    // Copy progress_th tile from local tensor to global tensor
    // Use Gather mask if _tile_len * sizeof(T) % 32 != 0
    if (_tile_len * sizeof(T) % 32 != 0) {
        DataCopyExtParams dcep = {1, static_cast<uint32_t>(_tile_len * sizeof(T)), 0, 0, 0};
        DataCopyPad(cGm[idxc], cUb, dcep);
    }
    DataCopy(cGm[idxc], cUb, _tile_len);
    // Free output Local tensor
    cQue.FreeTensor(cUb);
}

template<typename T>
__aicore__ inline void KernelSwiGLU<T>::Process() {
    for (int32_t i = 0; i < nt; ++i) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

__global__ __aicore__ void swiglu_kernel_f16(GM_ADDR c, GM_ADDR a, GM_ADDR b,
                                             float beta, int32_t nt, int32_t dh,
                                             int32_t sta, int32_t stb, int32_t stc,
                                             uint32_t remainder, uint32_t base) {
    KernelSwiGLU<half> op;
    op.Init(c, a, b, beta, nt, dh, sta, stb, stc, remainder, base);
    op.Process();
}

__global__ __aicore__ void swiglu_kernel_f32(GM_ADDR c, GM_ADDR a, GM_ADDR b,
                                             float beta, int32_t nt, int32_t dh,
                                             int32_t sta, int32_t stb, int32_t stc,
                                             uint32_t remainder, uint32_t base) {
    KernelSwiGLU<float> op;
    op.Init(c, a, b, beta, nt, dh, sta, stb, stc, remainder, base);
    op.Process();
}

extern "C" infiniopStatus_t swiglu_kernel_do(void *c, void *a, void *b,
                                             float beta, int32_t nt, int32_t dh,
                                             int32_t sta, int32_t stb, int32_t stc,
                                             int dtype, void *stream) {

    // Tiling params
    auto base = static_cast<uint32_t>(dh / BLOCK_NUM);
    auto remainder = static_cast<uint32_t>(dh % BLOCK_NUM);

    switch (dtype) {
        case 0:
            swiglu_kernel_f32<<<BLOCK_NUM, nullptr, stream>>>(
                c, a, b, beta, nt, dh, sta, stb, stc, remainder, base);
            return STATUS_SUCCESS;
        case 1:
            swiglu_kernel_f16<<<BLOCK_NUM, nullptr, stream>>>(
                c, a, b, beta, nt, dh, sta, stb, stc, remainder, base);
            return STATUS_SUCCESS;
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
