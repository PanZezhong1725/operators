#ifndef __COMMON_KUNLUN_H__
#define __COMMON_KUNLUN_H__

#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"
#include "xpu/xdnn.h"

namespace xdnn = baidu::xpu::api;

#define checkKUNLUNError(call)                                         \
    {                                                                  \
        auto err = call;                                               \
        if (XPU_SUCCESS != err) {                                      \
            fprintf(stderr, "KUNLUN error in %s:%i : %s.\n", __FILE__, \
                    __LINE__, xpu_strerror(err));                      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    }

#endif
