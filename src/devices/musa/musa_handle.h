#ifndef __MUSA_HANDLE_POOL_H__
#define __MUSA_HANDLE_POOL_H__

#include "pool.h"
#include "device.h"
#include "status.h"
#include <musa.h>
#include <musa_runtime_api.h>
#include <mudnn.h>
#include <mudnn_base.h>

using namespace musa::dnn;

struct MusaContext {
    Device device;
    int device_id;
    Pool<Handle> mudnn_handles;
};
typedef struct MusaContext *MusaHandle_t;

infiniopStatus_t createMusaHandle(Handle *handle_ptr, int device_id);
// const Pool<Handle> &get_musa_pool();

template<typename T>
void use_mudnn(MusaHandle_t musa_handle, musaStream_t stream, T const &f) {
    auto &pool = musa_handle->mudnn_handles;
    auto handle = pool.pop();
    if (!handle) {
        musaSetDevice(musa_handle->device_id);
        handle = new Handle(musa_handle->device_id);
    }
    handle->SetStream(stream);
    f(handle);
    pool.push(handle);
}

#endif