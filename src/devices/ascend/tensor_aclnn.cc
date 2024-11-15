#include "tensor_aclnn.h"
#include "../../ops/utils.h"
#include <algorithm>

/// @brief Set aclnnTensorDescriptor from infiniopTensorDescriptor
/// @param y infiniopTensorDescriptor
/// @return infiniopStatus_t
infiniopStatus_t aclnnTensorDescriptor::fromInfiniOpTensorDescriptor(infiniopTensorDescriptor_t y) {
    uint64_t ndim = y->ndim;
    // Cast shape type
    auto shape = new std::vector<int64_t>(ndim);
    auto strides = new std::vector<int64_t>(ndim);
    for (uint64_t i = 0; i < ndim; ++i) {
        (*shape)[i] = static_cast<int64_t>(y->shape[i]);
        (*strides)[i] = y->strides[i];
    }
    aclDataType dt;
    if (dtype_eq(y->dt, F16)) {
        dt = aclDataType::ACL_FLOAT16;
    } else if (dtype_eq(y->dt, F32)) {
        dt = aclDataType::ACL_FLOAT;
    } else {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    // Set format
    // TODO: Support other format
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    this->ndim = ndim;
    this->shape = (*shape).data();
    this->strides = (*strides).data();
    // TODO: Support other offset
    this->offset = 0;
    this->dataType = dt;
    this->format = format;

    infiniopTensorDescriptor_t yOri;
    CHECK_STATUS(inferOriginInfiniOpTensorDescriptor(y, &yOri), STATUS_SUCCESS);

    // Infer continuous storageShape
    auto storageShape = new std::vector<int64_t>(ndim);
    for (uint64_t i = 0; i < ndim - 1; ++i) {
        (*storageShape)[i] = ((yOri->shape)[i] * (yOri->strides)[i]) /
                             ((yOri->shape)[i + 1] * (yOri->strides)[i + 1]);
    }
    (*storageShape)[ndim - 1] = (yOri->shape)[ndim - 1] * (yOri->strides)[ndim - 1];
    this->storageShape = (*storageShape).data();
    this->storageNdim = ndim;

    CHECK_STATUS(infiniopDestroyTensorDescriptor(yOri), STATUS_SUCCESS);

    return STATUS_SUCCESS;
}

/// @brief Wrapper of aclCreateTensor. Create aclTensor.
/// See https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha001/apiref/appdevgapi/aclcppdevg_03_0168.html
/// @param desc Alias of aclnnTensorDescriptor*.
/// @param data Data ptr on device global mem.
/// @param tensor Pointer of pointer of aclTensor.
/// @return
infiniopStatus_t aclnnTensorDescriptor::createTensor() {
    if (this->t) {
        return STATUS_SUCCESS;
    }
    this->t = aclCreateTensor(this->shape,
                              this->ndim,
                              this->dataType,
                              this->strides,
                              this->offset,
                              this->format,
                              this->storageShape,
                              this->storageNdim,
                              nullptr);
    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnTensorDescriptor::destroyTensor() {
    auto ret = aclDestroyTensor(this->t);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclDesctroyTensor failed, ERROR: %d\n", ret);
              return STATUS_EXECUTION_FAILED);
    t = nullptr;
    shape = nullptr;
    strides = nullptr;
    storageShape = nullptr;

    return STATUS_SUCCESS;
}

infiniopStatus_t
aclnnTensorDescriptor::inferOriginInfiniOpTensorDescriptor(infiniopTensorDescriptor_t y,
                                                           infiniopTensorDescriptor_t *ori_ptr) {
    auto shape = y->shape;
    auto strides = y->strides;
    auto ndim = y->ndim;

    std::vector<uint64_t> indices(ndim);
    for (uint64_t i = 0; i < ndim; ++i) {
        indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(), [&](uint64_t a, uint64_t b) {
        return strides[a] > strides[b];
    });

    auto oriShape = new std::vector<uint64_t>(ndim);
    auto oriStrides = new std::vector<int64_t>(ndim);
    for (uint64_t i = 0; i < ndim; ++i) {
        (*oriShape)[i] = shape[indices[i]];
        (*oriStrides)[i] = strides[indices[i]];
    }

    auto status = infiniopCreateTensorDescriptor(
        ori_ptr,
        ndim,
        (*oriShape).data(),
        (*oriStrides).data(),
        y->dt);

    return status;
}

aclnnTensorDescriptor::~aclnnTensorDescriptor() {
    if (this->t) {
        destroyTensor();
    } else {
        delete shape;
        delete strides;
        delete storageShape;
    }
}

/// @brief TensorDescriptor's string info
/// @param desc Alias of aclnnTensorDescriptor*.
/// @return String of aclnnTensorDescriptor.
char *aclnnTensorDescriptor::toString() {

    // Assume bufferSize
    size_t bufferSize = 1024 + this->ndim * 40 + this->storageNdim * 40;
    char *buffer = (char *) malloc(bufferSize);
    if (!buffer) return NULL;

    // Write info into buffer
    char *ptr = buffer;
    ptr += sprintf(ptr, "ndim: %" PRId64 "\n", this->ndim);

    ptr += sprintf(ptr, "shape: [");
    for (uint64_t i = 0; i < this->ndim; ++i) {
        ptr += sprintf(ptr, "%" PRId64, this->shape[i]);
        if (i < this->ndim - 1) {
            ptr += sprintf(ptr, ", ");
        }
    }
    ptr += sprintf(ptr, "]\n");

    ptr += sprintf(ptr, "stride: [");
    for (uint64_t i = 0; i < this->ndim; ++i) {
        ptr += sprintf(ptr, "%" PRId64, this->strides[i]);
        if (i < this->ndim - 1) {
            ptr += sprintf(ptr, ", ");
        }
    }
    ptr += sprintf(ptr, "]\n");

    ptr += sprintf(ptr, "offset: %" PRId64 "\n", this->offset);
    ptr += sprintf(ptr, "dataType: %s\n", dataTypeToString(this->dataType));
    ptr += sprintf(ptr, "format: %s\n", formatToString(this->format));

    ptr += sprintf(ptr, "storageShape: [");
    for (int64_t i = 0; i < this->storageNdim; ++i) {
        ptr += sprintf(ptr, "%" PRId64, this->storageShape[i]);
        if (i < this->storageNdim - 1) {
            ptr += sprintf(ptr, ", ");
        }
    }
    ptr += sprintf(ptr, "]\n");

    ptr += sprintf(ptr, "storageNdim: %" PRId64 "\n", this->storageNdim);

    return buffer;
}
