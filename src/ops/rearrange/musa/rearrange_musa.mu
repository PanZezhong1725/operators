#include "../../../devices/musa/common_musa.h"
#include "rearrange_musa.h"

template<class Tmem>
static __global__ void rearrange(
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


void rearrange_mt_gpu(RearrangeMusaDescriptor_t desc, void *y, void *x, void *stream) {
    unsigned long int rsa = desc->rsa, csa = desc->csa, rsb = desc->rsb, csb = desc->csb;
    unsigned int r = desc->r, c = desc->c, b = desc->b, bytes_per_thread = desc->bytes_per_thread;
    auto dst_ptr = static_cast<void *>(reinterpret_cast<uint8_t *>(y));
    rsa /= b;
    csa /= b;
    auto src_ptr = static_cast<void const *>(reinterpret_cast<uint8_t const *>(x));
    rsb /= b;
    csb /= b;
    auto musa_stream = reinterpret_cast<musaStream_t>(stream);
    dim3 grid_dims = dim3((c + MAX_WARP_PER_BLOCK - 1) / MAX_WARP_PER_BLOCK, r);
    dim3 block_dims = dim3(WARP_SIZE, (c + grid_dims.x - 1) / grid_dims.x);
    switch (bytes_per_thread) {
        case 1:
            rearrange<uchar1><<<grid_dims, block_dims, 0, musa_stream>>>(dst_ptr, rsa, csa, src_ptr, rsb, csb, c);
            break;
        case 2:
            rearrange<uchar2><<<grid_dims, block_dims, 0, musa_stream>>>(dst_ptr, rsa, csa, src_ptr, rsb, csb, c);
            break;
        case 4:
            rearrange<float1><<<grid_dims, block_dims, 0, musa_stream>>>(dst_ptr, rsa, csa, src_ptr, rsb, csb, c);
            break;
        case 8:
            rearrange<float2><<<grid_dims, block_dims, 0, musa_stream>>>(dst_ptr, rsa, csa, src_ptr, rsb, csb, c);
            break;
        case 16:
            rearrange<float4><<<grid_dims, block_dims, 0, musa_stream>>>(dst_ptr, rsa, csa, src_ptr, rsb, csb, c);
            break;
        case 32:
            rearrange<double4><<<grid_dims, block_dims, 0, musa_stream>>>(dst_ptr, rsa, csa, src_ptr, rsb, csb, c);
            break;
        default:
            break;
    }
}
infiniopStatus_t musaRearrange(RearrangeMusaDescriptor_t desc,
                               void *dst, void *src, void *stream) {
	if(musaSetDevice(desc->device_id) != musaSuccess){
        return STATUS_BAD_DEVICE;
    }	
    rearrange_mt_gpu(desc, dst, src, stream);
    return STATUS_SUCCESS;
}
