#include "matmul_cnnl.h"
#include "../../../devices/bang/common_bang.h"
#include "../../../devices/bang/handle_pool.h"
#include "../../utils.h"
#include "cnrt.h"

MatmulBangDescriptor::MatmulBangDescriptor(Device device) {
    this->device = device;
    get_cnnl_pool();
}

void matmul_cnnl_f16(Tensor c, float beta, Tensor a, Tensor b, float alpha, void *stream) {
    // auto info = MatmulInfo(c, a, b, false);

    // int32_t use_stride = true;

    // cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
    // cnnlCreateTensorDescriptor(&aDesc);
    // cnnlCreateTensorDescriptor(&bDesc);
    // cnnlCreateTensorDescriptor(&cDesc);

    // setMatrixTensorEx(aDesc, info.a_matrix);
    // setMatrixTensorEx(bDesc, info.b_matrix);
    // setMatrixTensorEx(cDesc, info.c_matrix);

    // cnnlMatMulDescriptor_t opDesc;
    // cnnlMatMulAlgo_t algo;
    // cnnlMatMulHeuristicResult_t algoResult;
    // cnnlMatMulDescCreate(&opDesc);
    // cnnlMatMulAlgoCreate(&algo);
    // cnnlCreateMatMulHeuristicResult(&algoResult);

    // cnnlSetMatMulDescAttr(opDesc, CNNL_MATMUL_USE_STRIDE, &use_stride,
    //                       sizeof(int32_t));


    // void *workspace;

    // use_cnnl((cnrtQueue_t) stream,
    //          [&](cnnlHandle_t handle) {
    //              int count = 0;
    //              cnnlGetBatchMatMulAlgoHeuristic(handle, opDesc, aDesc,
    //                                              bDesc, cDesc,
    //                                              NULL, 1, &algoResult, &count);
    //              size_t wsSize;
    //              cnnlGetBatchMatMulHeuristicResult(algoResult, algo, &wsSize);
    //              cnrtMalloc(&workspace, wsSize);
    //              cnnlBatchMatMulBCast_v2(handle, opDesc, algo,
    //                                      &alpha, aDesc, info.a_ptr,
    //                                      bDesc, info.b_ptr,
    //                                      &beta, cDesc, info.c_ptr,
    //                                      workspace, wsSize);
    //          });

    // cnrtFree(workspace);

    // cnnlDestroyTensorDescriptor(aDesc);
    // cnnlDestroyTensorDescriptor(bDesc);
    // cnnlDestroyTensorDescriptor(cDesc);
    // cnnlMatMulDescDestroy(opDesc);
    // cnnlMatMulAlgoDestroy(algo);
    // cnnlDestroyMatMulHeuristicResult(algoResult);
}
