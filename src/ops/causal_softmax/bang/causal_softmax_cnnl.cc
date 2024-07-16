#include "causal_softmax_cnnl.h"
#include "../../../devices/bang/common_bang.h"
#include "../../../devices/bang/handle_pool.h"
#include "../../utils.h"
#include "cnrt.h"

#define checkCnnlError(call)                                                   \
    {                                                                          \
        cnnlStatus_t err = call;                                               \
        if (CNNL_STATUS_SUCCESS != err) {                                      \
            fprintf(stderr, "cnnl error in %s:%i : %s.\n", __FILE__, __LINE__, \
                    cnnlGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

CausalSoftmaxBangDescriptor::CausalSoftmaxBangDescriptor(Device device) {
    this->device = device;
    get_cnnl_pool();
}

void causal_softmax_cnnl_f16(CausalSoftmaxBangDescriptor *descriptor, Tensor t, void *stream) {
    ASSERT(t.layout->ndim >= 2);
    ASSERT(t.layout->shape[t.layout->ndim - 1] >= t.layout->shape[t.layout->ndim - 2]);

    int ndim_ = std::max(int(t.layout->ndim), 4);
    std::vector<int> dims(ndim_, 1);
    for (uint64_t i = 0; i < t.layout->ndim; i++) {
        dims[ndim_ - 1 - i] = static_cast<int>(t.layout->shape[t.layout->ndim - i - 1]);
    }

    // 创建 mask
    bool mask_matrix[dims[0]][dims[1]][dims[2]][dims[3]];

    // 填充上三角矩阵（右上角为 false）
    for (int i = 0; i < dims[0]; ++i) {
        for (int j = 0; j < dims[1]; ++j) {
            for (int m = 0; m < dims[2]; ++m) {
                for (int n = 0; n < dims[3]; ++n) {
                    if (n - m > dims[3] - dims[2]) {
                        mask_matrix[i][j][m][n] = true;
                    } else {
                        mask_matrix[i][j][m][n] = false;
                    }
                }
            }
        }
    }

    void *mask;
    cnrtMalloc((void **) &mask, sizeof(bool) * dims[0] * dims[1] * dims[2] * dims[3]);
    cnrtMemcpy(mask, mask_matrix, sizeof(bool) * dims[0] * dims[1] * dims[2] * dims[3], cnrtMemcpyHostToDev);

    // 不支持 stride
    cnnlSetTensorDescriptor(descriptor->tDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
                            dims.size(), dims.data());
    cnnlSetTensorDescriptor(descriptor->maskDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_BOOL,
                            dims.size(), dims.data());

    use_cnnl((cnrtQueue_t) stream,
             [&](cnnlHandle_t handle) {
                 checkCnnlError(cnnlMaskedSoftmax(handle, CNNL_MASKED_SOFTMAX_MASKED_FILL,
                                                  -1, 1.0, descriptor->tDesc, t.data, descriptor->maskDesc, mask,
                                                  descriptor->tDesc, t.data));
             });
}
