#ifndef INFINIOP_HANDLE_H
#define INFINIOP_HANDLE_H

#include "device.h"

typedef struct HandleStruct {
    Device device;
} HandleStruct;

typedef HandleStruct *infiniopHandle_t;

#endif
