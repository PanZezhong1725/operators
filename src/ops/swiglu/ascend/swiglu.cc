#include "swiglu.h"
#include "../../../utils.h"

extern "C" void swiglu_kernel_do(void *c, void *a, void *b,
                                 float beta, int32_t nt, int32_t dh,
                                 int32_t sta, int32_t stb, int32_t stc,
                                 int dtype, void *stream);

infiniopStatus_t ascendCreateSwiGLUDescriptor(infiniopHandle_t handle,
                                              SwiGLUAscendDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t c_desc,
                                              infiniopTensorDescriptor_t a_desc,
                                              infiniopTensorDescriptor_t b_desc) {
    uint64_t ndim = c_desc->ndim;
    DT dtype = c_desc->dt;

    aclDataType dt;
    if (dtype_eq(dtype, F16)) {
        dt = aclDataType::ACL_FLOAT16;
    } else if (dtype_eq(dtype, F32)) {
        dt = aclDataType::ACL_FLOAT;
    } else {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    if (ndim != 2 || a_desc->ndim != 2 || b_desc->ndim != 2) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    if (c_desc->strides[1] != 1 || a_desc->strides[1] != 1 || b_desc->strides[1] != 1) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    int32_t seq_len = static_cast<int32_t>(c_desc->shape[0]),
            di = static_cast<int32_t>(c_desc->shape[1]);

    int32_t sta = static_cast<int32_t>(a_desc->strides[0]);
    int32_t stb = static_cast<int32_t>(b_desc->strides[0]);
    int32_t stc = static_cast<int32_t>(c_desc->strides[0]);

    if (di % 1024 != 0) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    *desc_ptr = new SwiGLUAscendDescriptor{
        handle->device,
        dt,
        seq_len,
        di,
        sta,
        stb,
        stc};
    return STATUS_SUCCESS;
}

infiniopStatus_t ascendSwiGLU(SwiGLUAscendDescriptor_t desc,
                              void *c,
                              void const *a,
                              void const *b,
                              void *stream) {
    auto seq_len = desc->seq_len,
         di = desc->di;

    auto sta = desc->sta,
         stb = desc->stb,
         stc = desc->stc;

    auto dt = desc->dtype;

    swiglu_kernel_do(c, (void *) a, (void *) b, 1.0, seq_len, di, sta, stb, stc, dt, stream);
    return STATUS_SUCCESS;
}

infiniopStatus_t ascendDestroySwiGLUDescriptor(SwiGLUAscendDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}


// void swiglu_aclnn_f16(SwigluAscendCDescriptor *descriptor, Tensor gate,
//                       Tensor up, void *stream) {
//     // Rename in/out tensor descriptor
//     auto gateDesc = descriptor->out;
//     auto upDesc = descriptor->in;
//     aclnnTensorDesc_t catDesc;

//     // Copy tensor layout to descriptor
//     aclnnSetTensorDescriptorFromTensorLayout(gateDesc, gate.layout);
//     aclnnSetTensorDescriptorFromTensorLayout(upDesc, up.layout);

//     // Create aclnnCat out tensor
//     aclnnCreateTensorDescriptor(&catDesc);
//     std::vector<int64_t> catShape(gateDesc->ndim, 1);
//     for (auto i = 0; i < gateDesc->ndim; ++i) {
//         if (i != 1) {
//             ASSERT_EQ(gateDesc->shape[i], upDesc->shape[i]);
//             catShape[i] = gateDesc->shape[i];
//         }
//     }
//     catShape[1] = gateDesc->shape[1] + upDesc->shape[1];
//     aclnnSetTensorDescriptor(catDesc, catShape.data(), nullptr, catShape.size(),
//                              0, gateDesc->dataType, gateDesc->format);

//     // Malloc catDesc data
//     void *cat_data = nullptr;
//     auto ret =
//         aclrtMalloc(&cat_data, numElements(catDesc->shape, catDesc->ndim),
//                     ACL_MEM_MALLOC_HUGE_FIRST);
//     CHECK_RET(ret == ACL_SUCCESS,
//               LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret));

//     // aclnnCreateTensor
//     aclTensor *tgate;
//     aclTensor *tup;
//     aclTensor *tcat;

//     aclnnCreateTensor(gateDesc, gate.data, &tgate);
//     aclnnCreateTensor(upDesc, up.data, &tup);
//     aclnnCreateTensor(catDesc, cat_data, &tcat);

//     // Prepare aclnnCat
//     std::vector<aclTensor *> tmp{tgate, tup};
//     aclTensorList *tensorList = aclCreateTensorList(tmp.data(), tmp.size());
//     // aclnnCat
//     uint64_t workspaceSize = 0;
//     aclOpExecutor *executor;
//     ret = aclnnCatGetWorkspaceSize(tensorList, -1, tcat, &workspaceSize,
//                                    &executor);
//     CHECK_RET(ret == ACL_SUCCESS,
//               LOG_PRINT("aclnnCatGetWorkspaceSize failed. ERROR: %d\n", ret));
//     // Malloc workspace on device
//     void *workspaceAddr = mallocWorkspace(workspaceSize);
//     ret = aclnnCat(workspaceAddr, workspaceSize, executor, stream);
//     CHECK_RET(ret == ACL_SUCCESS,
//               LOG_PRINT("aclnnCat failed. ERROR: %d\n", ret));
//     freeWorkspace(workspaceAddr);

//     // Pretare aclnnSwiGlu
//     workspaceSize = 0;
//     executor = nullptr;
//     ret =
//         aclnnSwiGluGetWorkspaceSize(tcat, -1, tgate, &workspaceSize, &executor);
//     CHECK_RET(ret == ACL_SUCCESS,
//               LOG_PRINT("aclnnSwiGlu failed. ERROR: %d\n", ret));
//     workspaceAddr = mallocWorkspace(workspaceSize);
//     ret = aclnnSwiGlu(workspaceAddr, workspaceSize, executor, stream);
//     CHECK_RET(ret == ACL_SUCCESS,
//               LOG_PRINT("aclnnSwiGlu failed. ERROR: %d\n", ret));

//     // Wait device work
//     ret = aclrtSynchronizeStream(stream);
//     CHECK_RET(ret == ACL_SUCCESS,
//               LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret));

//     aclDestroyTensorList(tensorList);
//     aclDestroyTensor(tcat);

//     aclrtFree(cat_data);
//     aclnnDestoryTensorDescriptor(catDesc);
// }
