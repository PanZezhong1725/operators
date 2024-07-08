#include "matmul_cnnl.h"
#include "../../../devices/bang/common_bang.h"
#include "../../../devices/bang/handle_pool.h"
#include "../../utils.h"
#include "cnrt.h"

MatmulBangDescriptor::MatmulBangDescriptor(Device device) {
    this->device = device;
    get_cnnl_pool();
}

void matmul_cnnl_f16(MatmulBangDescriptor *descriptor, Tensor c, float beta, Tensor a, Tensor b, float alpha, void *stream) {
    auto info = MatmulInfo(c, a, b, false);

    int32_t use_stride = true;

    setMatrixTensorEx(descriptor->aDesc, info.a_matrix);
    setMatrixTensorEx(descriptor->bDesc, info.b_matrix);
    setMatrixTensorEx(descriptor->cDesc, info.c_matrix);

    cnnlSetMatMulDescAttr(descriptor->opDesc, CNNL_MATMUL_USE_STRIDE, &use_stride,
                          sizeof(int32_t));

    void *workspace;

    use_cnnl((cnrtQueue_t) stream,
             [&](cnnlHandle_t handle) {
                 int count = 0;
                 cnnlGetBatchMatMulAlgoHeuristic(handle, descriptor->opDesc, descriptor->aDesc,
                                                 descriptor->bDesc, descriptor->cDesc,
                                                 NULL, 1, &(descriptor->algoResult), &count);
                 size_t wsSize;
                 cnnlGetBatchMatMulHeuristicResult(descriptor->algoResult, descriptor->algo, &wsSize);
                 cnrtMalloc(&workspace, wsSize);
                 cnnlBatchMatMulBCast_v2(handle, descriptor->opDesc, descriptor->algo,
                                         &alpha, descriptor->aDesc, info.a_ptr,
                                         descriptor->bDesc, info.b_ptr,
                                         &beta, descriptor->cDesc, info.c_ptr,
                                         workspace, wsSize);
             });

    cnrtFree(workspace);
}
