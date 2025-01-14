#include "kunlun_handle.h"

infiniopStatus_t createKunlunHandle(KunlunHandle_t *handle_ptr, int device_id) {
    int device_count;
    xpu_device_count(&device_count);
    if (device_id >= device_count) {
        return STATUS_BAD_DEVICE;
    }

    auto pool = std::make_shared<Pool<xdnnHandle_t>>();
    if (xpu_set_device(device_id) != XPU_SUCCESS) {
        return STATUS_BAD_DEVICE;
    }
    xdnnHandle_t handle = xdnn::create_context();
    pool->push(std::move(handle));

    *handle_ptr = new KunlunContext {
        DevKunlunXpu,
        device_id,
        std::move(pool),
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t deleteKunlunHandle(KunlunHandle_t handle_ptr) {
    handle_ptr->xdnn_handles_t = nullptr;
    delete handle_ptr;

    return STATUS_SUCCESS;
}
