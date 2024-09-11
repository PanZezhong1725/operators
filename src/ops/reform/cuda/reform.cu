#include "../../utils.h"
#include "reform.cuh"
#include <numeric>

template<class Tmem>
static __global__ void reform(
    void *__restrict__ dst,
    unsigned int const rsa,
    unsigned int const csa,
    void const *__restrict__ src,
    unsigned int const rsb,
    unsigned int const csb,
    unsigned int const ncols) {

    auto row = blockIdx.y,
         col = blockIdx.x * blockDim.y + threadIdx.y;
    if (col >= ncols) return;

    auto thread = threadIdx.x,
         warp_size = blockDim.x;
    auto i = (row * rsa + col * csa) * warp_size + thread;
    auto j = (row * rsb + col * csb) * warp_size + thread;

    reinterpret_cast<Tmem *>(dst)[i] = reinterpret_cast<Tmem const *>(src)[j];
}

union DataLayout_ {
    DataLayout i;
    unsigned short u;
};

void reform_nv_gpu(Tensor y, Tensor x, void *stream) {
    DataLayout_ dl_y, dl_x;
    dl_y.i = y.layout->dt;
    dl_x.i = x.layout->dt;
    ASSERT_EQ(dl_y.u, dl_x.u);
    ASSERT_EQ(y.layout->ndim, x.layout->ndim);
    auto ndim = y.layout->ndim;
    ASSERT(ndim >= 2);
    for (int i = 0; i < ndim; ++i) {
        ASSERT_EQ(y.layout->shape[i], x.layout->shape[i]);
    }
    ASSERT_EQ(y.layout->strides[ndim - 1], y.layout->dt.size);
    ASSERT_EQ(x.layout->strides[ndim - 1], x.layout->dt.size);
    unsigned int r = 0, c = 0, b = 0;
    unsigned int rsa = 0, csa = 0, rsb = 0, csb = 0;
    if (ndim == 2) {
        c = y.layout->shape[0];
        b = y.layout->shape[1];
        csa = y.layout->strides[0] / y.layout->dt.size;
        csb = x.layout->strides[0] / x.layout->dt.size;
    } else if (ndim == 3) {
        r = y.layout->shape[0];
        c = y.layout->shape[1];
        b = y.layout->shape[2];
        csa = y.layout->strides[1] / y.layout->dt.size;
        csb = x.layout->strides[1] / x.layout->dt.size;
        rsa = y.layout->strides[0] / y.layout->dt.size;
        rsb = x.layout->strides[0] / x.layout->dt.size;
    } else {
        for (int i = ndim - 3; i >= 1; --i) {
            ASSERT_EQ(y.layout->shape[i] * y.layout->strides[i], y.layout->strides[i - 1]);
            ASSERT_EQ(x.layout->shape[i] * x.layout->strides[i], x.layout->strides[i - 1]);
        }
        r = std::accumulate(y.layout->shape, y.layout->shape + ndim - 2, 1, std::multiplies<unsigned int>());
        c = y.layout->shape[ndim - 2];
        b = y.layout->shape[ndim - 1];
        csa = y.layout->strides[ndim - 2] / y.layout->dt.size;
        csb = x.layout->strides[ndim - 2] / x.layout->dt.size;
        rsa = y.layout->strides[ndim - 3] / y.layout->dt.size;
        rsb = x.layout->strides[ndim - 3] / x.layout->dt.size;
    }
    auto contiguous_bytes = b * y.layout->dt.size;
    ASSERT_EQ(contiguous_bytes % WARP_SIZE, 0);
    auto bytes_per_thread = contiguous_bytes / WARP_SIZE;
    ASSERT(bytes_per_thread > 0 && bytes_per_thread <= 32 && (bytes_per_thread & (bytes_per_thread - 1)) == 0);

    auto dst_ptr = static_cast<void *>(reinterpret_cast<uint8_t *>(y.data));
    rsa /= b;
    csa /= b;
    auto src_ptr = static_cast<void const *>(reinterpret_cast<uint8_t const *>(x.data));
    rsb /= b;
    csb /= b;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid_dims = dim3((c + MAX_WARP_PER_BLOCK - 1) / MAX_WARP_PER_BLOCK, r);
    dim3 block_dims = dim3(WARP_SIZE, (c + grid_dims.x - 1) / grid_dims.x);
    switch (bytes_per_thread) {
        case 1:
            reform<uchar1><<<grid_dims, block_dims, 0, cuda_stream>>>(dst_ptr, rsa, csa, src_ptr, rsb, csb, c);
            break;
        case 2:
            reform<uchar2><<<grid_dims, block_dims, 0, cuda_stream>>>(dst_ptr, rsa, csa, src_ptr, rsb, csb, c);
            break;
        case 4:
            reform<float1><<<grid_dims, block_dims, 0, cuda_stream>>>(dst_ptr, rsa, csa, src_ptr, rsb, csb, c);
            break;
        case 8:
            reform<float2><<<grid_dims, block_dims, 0, cuda_stream>>>(dst_ptr, rsa, csa, src_ptr, rsb, csb, c);
            break;
        case 16:
            reform<float4><<<grid_dims, block_dims, 0, cuda_stream>>>(dst_ptr, rsa, csa, src_ptr, rsb, csb, c);
            break;
        case 32:
            reform<double4><<<grid_dims, block_dims, 0, cuda_stream>>>(dst_ptr, rsa, csa, src_ptr, rsb, csb, c);
            break;
    }
}
