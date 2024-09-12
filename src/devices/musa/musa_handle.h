#ifndef __MUSA_HANDLE_H__
#define __MUSA_HANDLE_H__

#include "pool.h"
#include "device.h"
#include "status.h"
#include "ops/matmul/matmul.h"
#include <memory>
#include <musa.h>
#include <musa_runtime_api.h>
#include <mudnn.h>
#include <mudnn_base.h>

// using namespace musa::dnn;

struct MusaContext {
    Device device;
    int device_id;
    std::shared_ptr<Pool<musa::dnn::Handle>> mudnn_handles_t;
};
typedef struct MusaContext *MusaHandle_t;

infiniopStatus_t createMusaHandle(MusaHandle_t *handle_ptr, int device_id);

infiniopStatus_t deleteMusaHandle(MusaHandle_t handle_ptr);

template<typename T>
void use_mudnn(std::shared_ptr<Pool<musa::dnn::Handle>> mudnn_handles_t, int device_id, musaStream_t stream, T const &f) {
    auto handle = mudnn_handles_t->pop();
    if (!handle) {
        // musaSetDevice(device_id);
        handle = new musa::dnn::Handle;
    }
    handle->SetStream(stream);
    f(handle);
    mudnn_handles_t->push(handle);
}

#endif // __MUSA_HANDLE_H__