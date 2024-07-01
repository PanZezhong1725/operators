#include "causal_softmax_cnnl.h"
#include "../../../devices/bang/common_bang.h"
#include "../../../devices/bang/handle_pool.h"
#include "../../utils.h"
#include "cnrt.h"

CausalSoftmaxBangDescriptor::CausalSoftmaxBangDescriptor(Device device) {
    this->device = device;
    get_cnnl_pool();
}

void causal_softmax_cnnl_f16(Tensor t, void *stream) {
    ASSERT(t.layout->ndim >= 2);
    cnnlTensorDescriptor_t tDesc;
    cnnlCreateTensorDescriptor(&tDesc);

    std::vector<int> dims(std::max(int(t.layout->ndim), 4), 1);
    for (uint64_t i = 1; i <= t.layout->ndim; i++) {
        dims[t.layout->ndim - i] = static_cast<int>(t.layout->shape[t.layout->ndim - i]);
    }
    cnnlSetTensorDescriptor(tDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
                            dims.size(), dims.data());

    use_cnnl((cnrtQueue_t) stream,
             [&](cnnlHandle_t handle) {
                 cnnlMaskedSoftmax(handle, CNNL_MASKED_SOFTMAX_UPPER_TRIANGLE_MASK_NEG_INF,
                                   -1, 1.0, tDesc, t.data, nullptr, nullptr, 
                                   tDesc, t.data);
             });

    cnnlDestroyTensorDescriptor(tDesc);
}
