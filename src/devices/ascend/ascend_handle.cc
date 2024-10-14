#include "ascend_handle.h"

infiniopStatus_t createAscendHandle(AscendHandle_t *handle_ptr, int device_id) {
    uint32_t device_count;
    aclrtGetDeviceCount(&device_count);
    if (device_id >= static_cast<int>(device_count)) {
        return STATUS_BAD_DEVICE;
    }

    auto pool = std::make_shared<Pool<aclOpExecutor *>>();

    auto ret = aclrtSetDevice(device_id);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret));

    aclOpExecutor *executor = nullptr;
    pool->push(std::move(executor));

    *handle_ptr = new AscendContext{DevAscendNpu, device_id, std::move(pool)};

    return STATUS_SUCCESS;
}

infiniopStatus_t deleteAscendHandle(AscendHandle_t handle_ptr) {
    handle_ptr->aclnn_handles = nullptr;
    delete handle_ptr;

    return STATUS_SUCCESS;
}
