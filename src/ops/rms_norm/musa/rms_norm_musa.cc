#include "rms_norm_musa.h"
#include "../../utils.h"
#include "../../../devices/musa/common_musa.h"

infiniopStatus_t musaCreateRMSNormDescriptor(MusaHandle_t handle, RMSNormMusaDescriptor_t *desc_ptr,
                                    infiniopTensorDescriptor_t y_desc,
                                    infiniopTensorDescriptor_t x_desc,
                                    infiniopTensorDescriptor_t w_desc,
                                    float epsilon) {
    if (y_desc->ndim != 2 || x_desc->ndim != 2 || w_desc->ndim != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    auto n = y_desc->shape[0],
         d = y_desc->shape[1];

    if (x_desc->shape[0] != n || x_desc->shape[1] != d || w_desc->shape[0] != d) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    unsigned long int stride_y = y_desc->strides[0];
    unsigned long int stride_x = x_desc->strides[0];
    auto w_datatype = w_desc->dt;
    *desc_ptr = new RMSNormMusaDescriptor{
        handle->device,
        handle->device_id,
        y_desc->dt,
        n,
        d,
        stride_y,
        stride_x,
        w_datatype,
        epsilon,
        handle->mudnn_handles_t};

    return STATUS_SUCCESS;
}

infiniopStatus_t musaGetRMSNormWorkspaceSize(RMSNormMusaDescriptor_t desc, unsigned long int *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t musaDestroyRMSNormDescriptor(RMSNormMusaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

infiniopStatus_t musaRMSNorm(RMSNormMusaDescriptor_t desc,
                                   void *workspace,
                                   unsigned long int workspace_size,
                                   void *y, void *x, void *w,
                                   void *stream){
    if(musaSetDevice(desc->device_id) != musaSuccess){
        return STATUS_BAD_DEVICE;
    }

    if (dtype_eq(desc->dtype, F16)){
        if (!dtype_eq(desc->w_datatype, F16)) {
            return STATUS_BAD_TENSOR_DTYPE;
        }
        rms_norm_mt_gpu_f16(desc, y, x, w, stream);
        return STATUS_SUCCESS;
    }

    return STATUS_BAD_TENSOR_DTYPE;
}

void rms_norm_mt_gpu_f16(RMSNormMusaDescriptor_t desc, void *y, void *x, void *w, void *stream) {
    musa::dnn::Tensor *out = createMudnnTensor(y, desc->n, desc->d, desc->dtype);
    musa::dnn::Tensor *in = createMudnnTensor(x, desc->n, desc->d, desc->dtype);
    musa::dnn::Tensor *gamma = createMudnnTensor(w, desc->d, desc->w_datatype);
    musa::dnn::Tensor *mean = new musa::dnn::Tensor();

    auto rms_norm_operator = createRMSNormOperator(desc->epsilon);

    use_mudnn(desc->mudnn_handles_t, desc->device_id, (musaStream_t) stream,
              [&](musa::dnn::Handle* handle) {
        rms_norm_operator->Run(*handle, *out, *mean, *in, *gamma);
    });
}

musa::dnn::Tensor* createMudnnTensor(void const *data, uint64_t batch, uint64_t dim, DT dtype) {
    musa::dnn::Tensor* tensor = new musa::dnn::Tensor();
    int64_t* dims = new int64_t[2]{(int64_t) batch, (int64_t) dim};

    tensor->SetAddr(data);
    tensor->SetFormat(musa::dnn::Tensor::Format::NCHW);
    tensor->SetNdInfo(2, dims);

    if (dtype_eq(dtype, F16)) {
        tensor->SetType(musa::dnn::Tensor::Type::HALF);
    }
    delete dims;
    return tensor;
}

musa::dnn::Tensor* createMudnnTensor(void const *data, uint64_t dim, DT dtype) {
    musa::dnn::Tensor* tensor = new musa::dnn::Tensor();

    int64_t* dims = new int64_t[1]{(int64_t) dim};

    tensor->SetAddr(data);
    tensor->SetFormat(musa::dnn::Tensor::Format::NCHW);
    tensor->SetNdInfo(1, dims);

    if (dtype_eq(dtype, F16)) {
        tensor->SetType(musa::dnn::Tensor::Type::HALF);
    }
    delete dims;
    return tensor;
}

musa::dnn::RMSNorm* createRMSNormOperator(float epsilon) {
    int axes[1] = {-1};
    musa::dnn::RMSNorm* rms_norm_operator = new musa::dnn::RMSNorm();
    rms_norm_operator->SetEpsilon(epsilon);
    rms_norm_operator->SetAxis(1, axes);
    return rms_norm_operator;
}
