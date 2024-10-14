#ifndef ASCEND_HANDLE_H
#define ASCEND_HANDLE_H

#include "../pool.h"
#include "common_ascend.h"
#include "device.h"
#include "status.h"
#include <acl/acl.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <aclnn/acl_meta.h>
#include <memory>

struct AscendContext {
    Device device;
    int device_id;
    std::shared_ptr<Pool<aclOpExecutor *>> aclnn_handles;
};
typedef struct AscendContext *AscendHandle_t;

infiniopStatus_t createAscendHandle(AscendHandle_t *handle_ptr, int device_id);

infiniopStatus_t deleteAscendHandle(AscendHandle_t handle_ptr);

template<typename T>
void use_aclnn(AscendHandle_t handle, T const &f) {
    auto &pool = handle->aclnn_handles;
    auto executor = pool->pop();
    if (!executor.has_value()) {
        aclrtSetDevice(handle->device_id);
        *executor = nullptr;
    }
    f(*executor);
    pool->push(std::move(*executor));
}

#endif
