#include "../utils.h"
#include "ops/attention/attention.h"
#include "ops/causal_softmax/causal_softmax.h"
#include "ops/matmul/matmul.h"
#include "ops/rearrange/rearrange.h"
#include "tensor/tensor_descriptor.h"
#include <cmath>

struct _AttentionDescriptor {
    Device device;
    infiniopRearrangeDescriptor_t rearrange_desc_k;
    infiniopRearrangeDescriptor_t rearrange_desc_v;
    infiniopRearrangeDescriptor_t rearrange_desc_out;
    infiniopMatmulDescriptor_t matmul_desc1;
    infiniopMatmulDescriptor_t matmul_desc2;
    infiniopCausalSoftmaxDescriptor_t softmax_desc;
    uint64_t workspace_size;
    uint64_t matmul1_workspace_size;
    uint64_t matmul1_tensor_size;
    uint64_t matmul2_workspace_size;
    uint64_t matmul2_tensor_size;
    uint64_t softmax_workspace_size;
    uint64_t k_cache_offset;
    uint64_t v_cache_offset;
    float qk_alpha;
};

typedef struct _AttentionDescriptor *_AttentionDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAttentionDescriptor(infiniopHandle_t handle,
                                                                infiniopAttentionDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t out_desc,
                                                                infiniopTensorDescriptor_t q_desc,
                                                                infiniopTensorDescriptor_t k_desc,
                                                                infiniopTensorDescriptor_t v_desc,
                                                                infiniopTensorDescriptor_t k_cache_desc,
                                                                infiniopTensorDescriptor_t v_cache_desc,
                                                                uint64_t pos) {
    if (out_desc->ndim != 3 || q_desc->ndim != 3 || k_desc->ndim != 3 ||
        v_desc->ndim != 3 || k_cache_desc->ndim != 3 || v_cache_desc->ndim != 3) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    uint64_t n_q_head = q_desc->shape[0];
    uint64_t seq_len = q_desc->shape[1];
    uint64_t head_dim = q_desc->shape[2];
    uint64_t hidden_size = n_q_head * head_dim;
    uint64_t n_kv_head = k_desc->shape[0];
    uint64_t total_seq_len = seq_len + pos;
    uint64_t n_group = n_q_head / n_kv_head;

    // out: [n_q_head, seq_len, head_dim]
    if (out_desc->shape[0] != n_q_head || out_desc->shape[1] != seq_len || out_desc->shape[2] != head_dim) {
        return STATUS_BAD_PARAM;
    }

    // k: [n_kv_head, seq_len, head_dim]
    if (k_desc->shape[0] != n_kv_head || k_desc->shape[1] != seq_len || k_desc->shape[2] != head_dim) {
        return STATUS_BAD_PARAM;
    }

    // v: [n_kv_head, seq_len, head_dim]
    if (v_desc->shape[0] != n_kv_head || v_desc->shape[1] != seq_len || v_desc->shape[2] != head_dim) {
        return STATUS_BAD_PARAM;
    }

    // k_cache: [n_kv_head, _, head_dim]
    if (k_cache_desc->shape[0] != n_kv_head || k_cache_desc->shape[1] < total_seq_len || k_cache_desc->shape[2] != head_dim) {
        return STATUS_BAD_PARAM;
    }

    // v_cache: [n_kv_head, _, head_dim]
    if (v_cache_desc->shape[1] != n_kv_head || v_cache_desc->shape[1] < total_seq_len || v_cache_desc->shape[2] != head_dim) {
        return STATUS_BAD_PARAM;
    }

    // Rearrange k into k_cache
    infiniopTensorDescriptor_t dst_k_desc = new TensorDescriptor;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&dst_k_desc, 3, k_desc->shape, k_cache_desc->strides, k_cache_desc->dt), STATUS_SUCCESS);
    infiniopRearrangeDescriptor_t rearrange_desc_k = new RearrangeDescriptor;
    CHECK_STATUS(infiniopCreateRearrangeDescriptor(handle, &rearrange_desc_k, dst_k_desc, k_desc), STATUS_SUCCESS);

    // Rearrange v into v_cache
    infiniopTensorDescriptor_t dst_v_desc = new TensorDescriptor;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&dst_v_desc, 3, v_desc->shape, v_cache_desc->strides, v_cache_desc->dt), STATUS_SUCCESS);
    infiniopRearrangeDescriptor_t rearrange_desc_v = new RearrangeDescriptor;
    CHECK_STATUS(infiniopCreateRearrangeDescriptor(handle, &rearrange_desc_v, dst_v_desc, v_desc), STATUS_SUCCESS);

    // Matmul1: q * full_k
    //      q: [n_q_head, seq_len, head_dim] -> [n_kv_head, n_group *seq_len, head_dim]
    infiniopTensorDescriptor_t reshaped_q_desc = new TensorDescriptor;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&reshaped_q_desc, 3, q_desc->shape, q_desc->strides, q_desc->dt), STATUS_SUCCESS);
    dim_split(reshaped_q_desc, 0, {n_kv_head, n_group});
    dim_merge(reshaped_q_desc, 1, 2);
    //      full_k: [n_kv_head, head_dim, total_seq_len]
    infiniopTensorDescriptor_t full_k_desc = new TensorDescriptor;
    uint64_t full_k_shape[3] = {n_kv_head, total_seq_len, head_dim};
    CHECK_STATUS(infiniopCreateTensorDescriptor(&full_k_desc, 3, full_k_shape, k_cache_desc->strides, k_cache_desc->dt), STATUS_SUCCESS);
    permute(full_k_desc, {0, 2, 1});
    //      qk: [n_kv_head, n_group * seq_len, total_seq_len]
    infiniopTensorDescriptor_t qk_desc = new TensorDescriptor;
    uint64_t qk_shape[3] = {n_kv_head, n_group * seq_len, total_seq_len};
    CHECK_STATUS(infiniopCreateTensorDescriptor(&qk_desc, 3, qk_shape, nullptr, q_desc->dt), STATUS_SUCCESS);
    //      matmul1_desc
    infiniopMatmulDescriptor_t matmul1_desc = new MatmulDescriptor;
    CHECK_STATUS(infiniopCreateMatmulDescriptor(handle, &matmul1_desc, qk_desc, q_desc, full_k_desc), STATUS_SUCCESS);
    //      matmul1 workspace size
    uint64_t matmul1_workspace_size;
    CHECK_STATUS(infiniopGetMatmulWorkspaceSize(matmul1_desc, &matmul1_workspace_size), STATUS_SUCCESS);
    //      matmul1 tensor size
    uint64_t matmul1_tensor_size = get_byte_size(qk_desc);

    // CausalSoftmax: softmax(qk)
    infiniopCausalSoftmaxDescriptor_t softmax_desc = new CausalSoftmaxDescriptor;
    CHECK_STATUS(infiniopCreateCausalSoftmaxDescriptor(handle, &softmax_desc, qk_desc), STATUS_SUCCESS);
    //      softmax workspace size
    uint64_t softmax_workspace_size;
    CHECK_STATUS(infiniopGetCausalSoftmaxWorkspaceSize(softmax_desc, &softmax_workspace_size), STATUS_SUCCESS);

    // Matmul2: softmax(qk) * full_v
    //      softmax(qk): [n_kv_head, n_group * seq_len, total_seq_len]
    //      full_v: [n_kv_head, total_seq_len, head_dim]
    infiniopTensorDescriptor_t full_v_desc = new TensorDescriptor;
    uint64_t full_v_shape[3] = {n_kv_head, total_seq_len, head_dim};
    CHECK_STATUS(infiniopCreateTensorDescriptor(&full_v_desc, 3, full_v_shape, v_cache_desc->strides, v_cache_desc->dt), STATUS_SUCCESS);
    //      temp_out: [n_kv_head, n_group * seq_len, head_dim]
    infiniopTensorDescriptor_t temp_out_desc = new TensorDescriptor;
    uint64_t temp_out_shape[3] = {n_kv_head, n_group * seq_len, head_dim};
    CHECK_STATUS(infiniopCreateTensorDescriptor(&temp_out_desc, 3, temp_out_shape, nullptr, q_desc->dt), STATUS_SUCCESS);
    //      matmul2_desc
    infiniopMatmulDescriptor_t matmul2_desc = new MatmulDescriptor;
    CHECK_STATUS(infiniopCreateMatmulDescriptor(handle, &matmul2_desc, temp_out_desc, qk_desc, full_v_desc), STATUS_SUCCESS);
    //      matmul2 workspace size
    uint64_t matmul2_workspace_size;
    CHECK_STATUS(infiniopGetMatmulWorkspaceSize(matmul2_desc, &matmul2_workspace_size), STATUS_SUCCESS);
    //      matmul2 tensor size
    uint64_t matmul2_tensor_size = get_byte_size(temp_out_desc);

    // Rearrange temp_out into out
    //      out: [n_q_head, seq_len, head_dim]
    //      temp_out: [n_kv_head, n_group * seq_len, head_dim]
    dim_split(temp_out_desc, 0, {n_kv_head, n_group});
    dim_merge(temp_out_desc, 1, 2);
    infiniopRearrangeDescriptor_t rearrange_desc_out = new RearrangeDescriptor;
    CHECK_STATUS(infiniopCreateRearrangeDescriptor(handle, &rearrange_desc_out, out_desc, temp_out_desc), STATUS_SUCCESS);

    // workspace size
    uint64_t workspace_size = std::max(std::max(matmul1_workspace_size + matmul1_tensor_size,
                                                matmul1_tensor_size + softmax_workspace_size),
                                       matmul1_tensor_size + matmul2_workspace_size + matmul2_tensor_size);

    // k_cache_offset
    uint64_t k_cache_offset = 0;
    if (pos > 0) {
        k_cache_offset = pos * k_cache_desc->strides[0] * k_cache_desc->strides[1];
    }

    // v_cache_offset
    uint64_t v_cache_offset = 0;
    if (pos > 0) {
        v_cache_offset = pos * v_cache_desc->strides[0] * v_cache_desc->strides[1];
    }

    // qk_alpha
    float qk_alpha = 1 / sqrt(head_dim);

    // create attention descriptor
    *(_AttentionDescriptor_t *) desc_ptr = new _AttentionDescriptor{
        handle->device,
        rearrange_desc_k,
        rearrange_desc_v,
        rearrange_desc_out,
        matmul1_desc,
        matmul2_desc,
        softmax_desc,
        workspace_size,
        matmul1_workspace_size,
        matmul1_tensor_size,
        matmul2_workspace_size,
        matmul2_tensor_size,
        softmax_workspace_size,
        k_cache_offset,
        v_cache_offset,
        qk_alpha,
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopGetAttentionWorkspaceSize(infiniopAttentionDescriptor_t desc, uint64_t *size) {
    *size = ((_AttentionDescriptor_t) desc)->workspace_size;
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopAttention(infiniopAttentionDescriptor_t desc,
                                                void *workspace,
                                                uint64_t workspace_size,
                                                void *out,
                                                void *q,
                                                void *k,
                                                void *v,
                                                void *k_cache,
                                                void *v_cache,
                                                void *stream) {
    auto _desc = (_AttentionDescriptor_t) desc;
    if (workspace_size < _desc->workspace_size) {
        return STATUS_MEMORY_NOT_ALLOCATED;
    }

    // concat k and v to k_cache and v_cache
    CHECK_STATUS(infiniopRearrange(_desc->rearrange_desc_k,
                                   (char *) k_cache + _desc->k_cache_offset, k, stream),
                 STATUS_SUCCESS);
    CHECK_STATUS(infiniopRearrange(_desc->rearrange_desc_v,
                                   (char *) v_cache + _desc->v_cache_offset, v, stream),
                 STATUS_SUCCESS);
    // matmul1: q * full_k
    CHECK_STATUS(infiniopMatmul(_desc->matmul_desc1,
                                (char *) workspace + _desc->matmul1_tensor_size, workspace_size - _desc->matmul1_tensor_size,
                                workspace, q, k_cache, _desc->qk_alpha, 0, stream),
                 STATUS_SUCCESS);
    // softmax(qk)
    CHECK_STATUS(infiniopCausalSoftmax(_desc->softmax_desc,
                                       (char *) workspace + _desc->matmul1_tensor_size, workspace_size - _desc->matmul1_tensor_size,
                                       workspace, stream),
                 STATUS_SUCCESS);
    // matmul2: softmax(qk) * full_v
    CHECK_STATUS(infiniopMatmul(_desc->matmul_desc2,
                                (char *) workspace + _desc->matmul1_tensor_size + _desc->matmul2_tensor_size,
                                workspace_size - _desc->matmul1_tensor_size - _desc->matmul2_tensor_size,
                                out, workspace, v_cache, 1, 0, stream),
                 STATUS_SUCCESS);
    // rearrange out
    CHECK_STATUS(infiniopRearrange(_desc->rearrange_desc_out, out, (char *) workspace + _desc->matmul1_tensor_size, stream), STATUS_SUCCESS);

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyAttentionDescriptor(infiniopAttentionDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyRearrangeDescriptor(((_AttentionDescriptor_t) desc)->rearrange_desc_k), STATUS_SUCCESS);
    CHECK_STATUS(infiniopDestroyRearrangeDescriptor(((_AttentionDescriptor_t) desc)->rearrange_desc_v), STATUS_SUCCESS);
    CHECK_STATUS(infiniopDestroyRearrangeDescriptor(((_AttentionDescriptor_t) desc)->rearrange_desc_out), STATUS_SUCCESS);
    CHECK_STATUS(infiniopDestroyMatmulDescriptor(((_AttentionDescriptor_t) desc)->matmul_desc1), STATUS_SUCCESS);
    CHECK_STATUS(infiniopDestroyMatmulDescriptor(((_AttentionDescriptor_t) desc)->matmul_desc2), STATUS_SUCCESS);
    CHECK_STATUS(infiniopDestroyCausalSoftmaxDescriptor(((_AttentionDescriptor_t) desc)->softmax_desc), STATUS_SUCCESS);
    delete (_AttentionDescriptor_t) desc;

    return STATUS_SUCCESS;
}
