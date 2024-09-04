#ifndef BANG_HANDLE_H
#define BANG_HANDLE_H

#include "../pool.h"
#include "cnnl.h"
#include "cnrt.h"
#include "device.h"
#include "status.h"
#include <memory>

struct BangContext {
    Device device;
    int device_id;
    std::shared_ptr<Pool<cnnlHandle_t>> cnnl_handles;
};
typedef struct BangContext *BangHandle_t;

infiniopStatus_t createBangHandle(BangHandle_t *handle_ptr, int device_id);

template<typename T>
void use_cnnl(std::shared_ptr<Pool<cnnlHandle_t>> &pool, int device_id, cnrtQueue_t queue, T const &f) {
    auto handle = pool->pop();
    if (!handle) {
        cnrtSetDevice(device_id);
        cnnlCreate(&(*handle));
    }
    cnnlSetQueue(*handle, (cnrtQueue_t) queue);
    f(*handle);
    pool->push(std::move(*handle));
}

#endif
