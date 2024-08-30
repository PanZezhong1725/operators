#include "musa_handle.h"

infiniopStatus_t createMusaHandle(MusaHandle_t* handle_ptr, int device_id) {
    int device_count;
    musaGetDeviceCount(&device_count);
    if (device_id >= device_count) {
        return STATUS_BAD_DEVICE;
    }

    auto pool = Pool<musa::dnn::Handle>();
    musaSetDevice(device_id);
    musa::dnn::Handle *handle = new musa::dnn::Handle(device_id);
    pool.push(handle);

    *handle_ptr = new MusaContext{DevMtGpu, device_id, std::move(pool)};

    return STATUS_SUCCESS;
}