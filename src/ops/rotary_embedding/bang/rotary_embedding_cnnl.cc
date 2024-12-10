#include "rotary_embedding_cnnl.h"
#include "../../../devices/bang/common_bang.h"
#include "../../utils.h"

infiniopStatus_t cnnlCreateRoPEDescriptor(BangHandle_t handle,
                                          RoPECnnlDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t t,
                                          infiniopTensorDescriptor_t pos_ids,
                                          infiniopTensorDescriptor_t sin_table,
                                          infiniopTensorDescriptor_t cos_table) {

    if (desc_ptr == nullptr)
        return STATUS_MEMORY_NOT_ALLOCATED;

    if (t->ndim != 3 ||
        pos_ids->ndim != 1 ||
        sin_table->ndim != 2 ||
        cos_table->ndim != 2)
        return STATUS_BAD_TENSOR_SHAPE;

    auto seq_len = static_cast<int>(t->shape[0]);
    auto nhead = static_cast<int>(t->shape[1]);
    auto dim = static_cast<int>(t->shape[2]);
    auto total_seq_len = static_cast<int>(sin_table->shape[0]);

    if (dim % 2 != 0)
        return STATUS_BAD_TENSOR_SHAPE;

    if (pos_ids->shape[0] != t->shape[0] ||
        sin_table->shape[1] != t->shape[2] ||
        cos_table->shape[1] != t->shape[2] ||
        sin_table->shape[0] != cos_table->shape[0])
        return STATUS_BAD_TENSOR_SHAPE;

    if (t->strides[2] != 1 ||
        pos_ids->strides[0] != 1 ||
        sin_table->strides[1] != 1 ||
        cos_table->strides[1] != 1)
        return STATUS_BAD_TENSOR_STRIDES;

    if (!dtype_eq(t->dt, F16))
        return STATUS_BAD_TENSOR_DTYPE;

    if (!dtype_eq(sin_table->dt, F32) || !dtype_eq(cos_table->dt, F32))
        return STATUS_BAD_TENSOR_DTYPE;

    if (!dtype_eq(pos_ids->dt, U64))
        return STATUS_BAD_TENSOR_DTYPE;

    cnnlRotaryEmbeddingDescriptor_t ropeDesc;

    cnnlCreateRotaryEmbeddingDescriptor(&ropeDesc);
    cnnlSetRotaryEmbeddingDescriptor_v2(ropeDesc, false, true,
                                        false, false, CNNL_SEQDATA_TNBC);

    cnnlTensorDescriptor_t inDesc, posDesc, sinFullDesc, sinSelectedDesc, sinSelectedFP16Desc;
    cnnlCreateTensorDescriptor(&inDesc);
    cnnlCreateTensorDescriptor(&posDesc);
    cnnlCreateTensorDescriptor(&sinFullDesc);
    cnnlCreateTensorDescriptor(&sinSelectedDesc);
    cnnlCreateTensorDescriptor(&sinSelectedFP16Desc);

    int inShape[4] = {seq_len, 1, nhead, dim};
    int inStrides[4] = {
        static_cast<int>(t->strides[0]),
        static_cast<int>(t->strides[0]),
        static_cast<int>(t->strides[1]),
        static_cast<int>(t->strides[2]),
    };
    int sinFullShape[2] = {total_seq_len, dim};
    int sinSelectedShape[2] = {seq_len, dim};
    cnnlSetTensorDescriptorEx(inDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(t->dt), 4, inShape, inStrides);
    cnnlSetTensorDescriptor(posDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT64, 1, &seq_len);
    cnnlSetTensorDescriptor(sinFullDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, sinFullShape);
    cnnlSetTensorDescriptor(sinSelectedDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, sinSelectedShape);
    cnnlSetTensorDescriptor(sinSelectedFP16Desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF, 2, sinSelectedShape);

    *desc_ptr = new RoPECnnlDescriptor{
        handle->device,
        handle->device_id,
        handle->cnnl_handles,
        t->dt,
        std::move(ropeDesc),
        std::move(inDesc),
        std::move(posDesc),
        std::move(sinFullDesc),
        std::move(sinSelectedDesc),
        std::move(sinSelectedFP16Desc),
    };

    return STATUS_SUCCESS;
}


infiniopStatus_t cnnlGetRoPEWorkspaceSize(RoPECnnlDescriptor_t desc, uint64_t *size) {
    *size = cnnlGetTensorElementNum(desc->sinSelectedDesc) * sizeof(float) * 2;
    return STATUS_SUCCESS;
}


infiniopStatus_t cnnlDestroyRoPEDescriptor(RoPECnnlDescriptor_t desc) {
    cnnlDestroyRotaryEmbeddingDescriptor(desc->ropeDesc);
    cnnlDestroyTensorDescriptor(desc->inDesc);
    cnnlDestroyTensorDescriptor(desc->posDesc);
    cnnlDestroyTensorDescriptor(desc->sinFullDesc);
    cnnlDestroyTensorDescriptor(desc->sinSelectedDesc);
    delete desc;
    return STATUS_SUCCESS;
}

infiniopStatus_t cnnlRoPE(RoPECnnlDescriptor_t desc,
                          void *workspace,
                          uint64_t workspace_size,
                          void *t,
                          void const *pos_ids,
                          void const *sin_table,
                          void const *cos_table,
                          void *stream) {

    if (cnrtSetDevice(desc->device_id) != cnrtSuccess) {
        return STATUS_BAD_DEVICE;
    }

    use_cnnl(desc->pool, desc->device_id, (cnrtQueue_t) stream,
             [&](cnnlHandle_t handle) {
                 void *sinSelectedTable = workspace;
                 void *cosSelectedTable = reinterpret_cast<char *>(workspace) + workspace_size / 2;
                 // Use Gather to select sin/cos entries by pos_id
                 cnnlBatchGatherV2_v2(handle, 0, 0, 1,
                                      desc->sinFullDesc, sin_table,
                                      desc->posDesc, pos_ids,
                                      desc->sinSelectedDesc, sinSelectedTable);
                 cnnlCastDataType(handle, desc->sinSelectedDesc, sinSelectedTable,
                                  CNNL_CAST_FLOAT_TO_HALF, 
                                  desc->sinSelectedFP16Desc, sinSelectedTable);
                 cnnlBatchGatherV2_v2(handle, 0, 0, 1,
                                      desc->sinFullDesc, cos_table,
                                      desc->posDesc, pos_ids,
                                      desc->sinSelectedDesc, cosSelectedTable);
                 cnnlCastDataType(handle, desc->sinSelectedDesc, cosSelectedTable,
                                  CNNL_CAST_FLOAT_TO_HALF,
                                  desc->sinSelectedFP16Desc, cosSelectedTable);

                 // Do RoPE
                 cnnlRotaryEmbedding_v2(handle, desc->ropeDesc,
                                        desc->inDesc, t, nullptr, nullptr,
                                        desc->sinSelectedFP16Desc, cosSelectedTable,
                                        desc->sinSelectedFP16Desc, sinSelectedTable,
                                        nullptr, nullptr, nullptr, nullptr, nullptr, 0,
                                        desc->inDesc, t, nullptr, nullptr);
             });

    cnrtQueueSync((cnrtQueue_t) stream);

    return STATUS_SUCCESS;
}