#include "musa_handle.h"

infiniopStatus_t createMusaHandle(MusaHandle_t* handle_ptr, int device_id) {
    printf("aaa");
    int device_count;
    musaGetDeviceCount(&device_count);
    if (device_id >= device_count) {
        return STATUS_BAD_DEVICE;
    }

    auto pool = std::make_shared<Pool<musa::dnn::Handle>>();

    musa::dnn::Handle *handle = new musa::dnn::Handle;
    pool->push(handle);

    *handle_ptr = new MusaContext{DevMtGpu, device_id, std::move(pool)};

    return STATUS_SUCCESS;
}

infiniopStatus_t deleteMusaHandle(MusaHandle_t handle_ptr) {
    // handle_ptr->mudnn_handles_t = nullptr;
    delete handle_ptr;

    return STATUS_SUCCESS;
}