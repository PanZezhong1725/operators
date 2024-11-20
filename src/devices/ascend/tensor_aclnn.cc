#include "tensor_aclnn.h"
#include "../../ops/utils.h"
#include <algorithm>

infiniopStatus_t aclnnTensorDescriptor::setDescriptor(DT dtype, const std::vector<int64_t> &shape, const std::vector<int64_t> &strides){
    if (shape.size()!= strides.size()) {
        return STATUS_BAD_PARAM;
    }
    this->ndim = shape.size();
    this->shape = std::vector<int64_t>(shape);
    this->strides = std::vector<int64_t>(strides);

    if (dtype_eq(dtype, F16)) {
        this->dataType = aclDataType::ACL_FLOAT16;
    } else if (dtype_eq(dtype, F32)) {
        this->dataType = aclDataType::ACL_FLOAT;
    } else {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    // Set format
    // TODO: Support other format
    aclFormat format = aclFormat::ACL_FORMAT_ND;
    this->format = format;

    CHECK_STATUS(this->inferStorageShape(), STATUS_SUCCESS);

    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnTensorDescriptor::inferStorageShape(){
    this->storageNdim = this->ndim;
    this->storageShape = std::vector<int64_t>(this->storageNdim, 1);
    auto shape = std::vector<int64_t>(this->shape);
    auto strides = std::vector<int64_t>(this->strides);
    std::vector<uint64_t> indices(ndim);
    for (uint64_t i = 0; i < ndim; ++i) {
        indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(), [&](uint64_t a, uint64_t b) {
        return strides[a] > strides[b];
    });
    auto bound = 0; // upper bound of non-zero-strided dimension
    for (uint64_t i = 0; i < ndim; ++i) {
        // sort shape and strides by strides
        shape[i] = this->shape[indices[i]];
        strides[i] = this->strides[indices[i]];
        if (strides[i] >= 1){
            bound++;
        }else if (strides[i] < 0){
            // negative stride not supported
            return STATUS_BAD_TENSOR_STRIDES;
        }
    }
    // Treat the last non-zero-strided dimension as continuous 
    // All trilling zero-strided dimensions are treated as 1
    shape[bound - 1] = shape[bound - 1] * strides[bound - 1];
    strides[bound - 1] = 1;
    int64_t carry = 1;
    printf("In inferStorageShape\n");
    for (size_t i = 0; i < this->shape.size(); i++){
        printf("%lu ", this->shape[i]);
    }
    printf("\n");
    for (size_t i = 0; i < this->strides.size(); i++){
        printf("%ld ", this->strides[i]);
    }
    printf("\n");

    for (int64_t i = bound - 1; i > 0; --i) {
        // Each non-cummulative stride should be no smaller than corresponding dim
        // and storage shape is the bigger one
        this->storageShape[i] = strides[i-1] / carry;
        if (shape[i] > this->storageShape[i]){
                return STATUS_BAD_TENSOR_STRIDES;
        }
        carry *= this->storageShape[i];  
    }
    this->storageShape[0] = shape[0];
    
    return STATUS_SUCCESS;
}

/// @brief Set aclnnTensorDescriptor from infiniopTensorDescriptor
/// @param y infiniopTensorDescriptor
/// @return infiniopStatus_t
infiniopStatus_t aclnnTensorDescriptor::fromInfiniOpTensorDescriptor(infiniopTensorDescriptor_t y) {
    uint64_t ndim = y->ndim;
    // Cast shape type
    auto shape = std::vector<int64_t>(ndim);
    auto strides =std::vector<int64_t>(ndim);
    for (uint64_t i = 0; i < ndim; ++i) {
        shape[i] = static_cast<int64_t>(y->shape[i]);
        strides[i] = y->strides[i];
    }
    return setDescriptor(y->dt, shape, strides);
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
    this->t = aclCreateTensor(this->shape.data(),
                              this->ndim,
                              this->dataType,
                              this->strides.data(),
                              this->offset,
                              this->format,
                              this->storageShape.data(),
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

    return STATUS_SUCCESS;
}

aclnnTensorDescriptor::~aclnnTensorDescriptor() {
    if (this->t) {
        destroyTensor();
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
