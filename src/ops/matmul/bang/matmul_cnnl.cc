#include "matmul_cnnl.h"
#include "../../../devices/bang/common_bang.h"
#include "../../../devices/bang/handle_pool.h"
#include "../../utils.h"
#include "../blas.h"
#include "cnrt.h"

MatmulBangDescriptor::MatmulBangDescriptor(Device device) {
    this->device = device;
    get_cnnl_pool();
}

void matmul_cnnl_f16(Tensor c, float beta, Tensor a, Tensor b, float alpha, void *stream) {
    auto info = MatmulInfo(c, a, b);

    cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
    cnnlCreateTensorDescriptor(&aDesc);
    cnnlCreateTensorDescriptor(&bDesc);
    cnnlCreateTensorDescriptor(&cDesc);

    setCnnlTensor(aDesc, a.layout);
    setCnnlTensor(bDesc, b.layout);
    setCnnlTensor(cDesc, c.layout);

    cnnlMatMulDescriptor_t opDesc;
    cnnlMatMulAlgo_t algo;
    cnnlMatMulHeuristicResult_t algoResult;
    cnnlMatMulDescCreate(&opDesc);
    cnnlMatMulAlgoCreate(&algo);
    cnnlCreateMatMulHeuristicResult(&algoResult);

    int32_t transA = info.a_matrix.row_stride == 1 ? false : true;
    int32_t transB = info.b_matrix.row_stride == 1 ? false : true;
    cnnlSetMatMulDescAttr(opDesc, CNNL_MATMUL_DESC_TRANSA, &transA,
                          sizeof(int32_t));
    cnnlSetMatMulDescAttr(opDesc, CNNL_MATMUL_DESC_TRANSB, &transB,
                          sizeof(int32_t));


    void *workspace;

    use_cnnl((cnrtQueue_t) stream,
             [&](cnnlHandle_t handle) {
                 int count = 0;
                 cnnlGetBatchMatMulAlgoHeuristic(handle, opDesc, aDesc,
                                                 bDesc, cDesc,
                                                 NULL, 1, &algoResult, &count);
                 size_t wsSize;
                 cnnlGetBatchMatMulHeuristicResult(algoResult, algo, &wsSize);
                 cnrtMalloc(&workspace, wsSize);
                 cnnlBatchMatMulBCast_v2(handle, opDesc, algo,
                                         &alpha, aDesc, a.data,
                                         bDesc, b.data,
                                         &beta, cDesc, c.data,
                                         workspace, wsSize);
             });

    cnrtFree(workspace);

    cnnlDestroyTensorDescriptor(aDesc);
    cnnlDestroyTensorDescriptor(bDesc);
    cnnlDestroyTensorDescriptor(cDesc);
    cnnlMatMulDescDestroy(opDesc);
    cnnlMatMulAlgoDestroy(algo);
    cnnlDestroyMatMulHeuristicResult(algoResult);
}
