#ifndef __KUNLUN_HANDLE_H__
#define __KUNLUN_HANDLE_H__

#include "../pool.h"
#include "common_kunlun.h"
#include "device.h"
#include "status.h"

typedef xdnn::Context *xdnnHandle_t;

struct KunlunContext {
    Device device;
    int device_id;
    std::shared_ptr<Pool<xdnnHandle_t>> xdnn_handles_t;
};
typedef struct KunlunContext *KunlunHandle_t;

infiniopStatus_t createKunlunHandle(KunlunHandle_t *handle_ptr, int device_id);

infiniopStatus_t deleteKunlunHandle(KunlunHandle_t handle_ptr);

template<typename T>
void use_xdnn(std::shared_ptr<Pool<xdnnHandle_t>> xdnn_handles_t,
              int device_id,
              XPUStream stream,
              T const &f) {
    auto handle = xdnn_handles_t->pop();
    if (!handle) {
        xpu_set_device(device_id);
        *handle = xdnn::create_context();
    }
    (*handle)->set_stream(stream);
    f(*handle);
    xdnn_handles_t->push(std::move(*handle));
}

#endif
