#include "tensor_aclnn.h"
#include "../../utils.h"

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

    // Infer continuous storageShape
    auto storageShape = new std::vector<int64_t>(ndim);
    for (uint64_t i = 0; i < ndim - 1; ++i) {
        (*storageShape)[i] = ((*shape)[i] * (*strides)[i]) /
                             ((*shape)[i + 1] * (*strides)[i + 1]);
    }
    (*storageShape)[ndim - 1] = (*shape)[ndim - 1];
    this->storageShape = (*storageShape).data();
    this->storageNdim = ndim;

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
    auto status = aclDestroyTensor(this->t);
    if (status != 0) {
        return STATUS_EXECUTION_FAILED;
    }
    t = nullptr;
    shape = nullptr;
    strides = nullptr;
    storageShape = nullptr;

    return STATUS_SUCCESS;
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
