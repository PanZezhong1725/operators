#ifndef INFINIOP_HANDLE_EXPORT_H
#define INFINIOP_HANDLE_EXPORT_H
#include "../status.h"
#include "../handle.h"
#include "../export.h"
#include "../device.h"

__C __export infiniopStatus_t infiniopCreateHandle(infiniopHandle_t *handle_ptr, Device device, int device_id);

__C __export infiniopStatus_t infiniopDestroyHandle(infiniopHandle_t handle);

#endif // INFINIOP_HANDLE_EXPORT_H
