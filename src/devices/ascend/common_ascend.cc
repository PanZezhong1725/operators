#include "common_ascend.h"

int64_t numElements(const int64_t *shape, int64_t num) {
    int64_t numEle = 1;
    for (int i = 0; i < num; i++) {
        numEle *= shape[i];
    }
    return numEle;
}

void *mallocWorkspace(uint64_t workspaceSize) {
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspaceAddr, workspaceSize,
                          ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret));
    }
    return workspaceAddr;
}

void freeWorkspace(void *workspaceAddr) {
    aclrtFree(workspaceAddr);
}

const char *dataTypeToString(aclDataType dtype) {
    switch (dtype) {
    case ACL_DT_UNDEFINED:
        return "ACL_DT_UNDEFINED";
    case ACL_FLOAT:
        return "ACL_FLOAT";
    case ACL_FLOAT16:
        return "ACL_FLOAT16";
    case ACL_INT8:
        return "ACL_INT8";
    case ACL_INT32:
        return "ACL_INT32";
    case ACL_UINT8:
        return "ACL_UINT8";
    case ACL_INT16:
        return "ACL_INT16";
    case ACL_UINT16:
        return "ACL_UINT16";
    case ACL_UINT32:
        return "ACL_UINT32";
    case ACL_INT64:
        return "ACL_INT64";
    case ACL_UINT64:
        return "ACL_UINT64";
    case ACL_DOUBLE:
        return "ACL_DOUBLE";
    case ACL_BOOL:
        return "ACL_BOOL";
    case ACL_STRING:
        return "ACL_STRING";
    case ACL_COMPLEX64:
        return "ACL_COMPLEX64";
    case ACL_COMPLEX128:
        return "ACL_COMPLEX128";
    case ACL_BF16:
        return "ACL_BF16";
    case ACL_INT4:
        return "ACL_INT4";
    case ACL_UINT1:
        return "ACL_UINT1";
    case ACL_COMPLEX32:
        return "ACL_COMPLEX32";
    default:
        return "UNKNOWN";
    }
}

const char *formatToString(aclFormat format) {
    switch (format) {
    case ACL_FORMAT_UNDEFINED:
        return "ACL_FORMAT_UNDEFINED";
    case ACL_FORMAT_NCHW:
        return "ACL_FORMAT_NCHW";
    case ACL_FORMAT_NHWC:
        return "ACL_FORMAT_NHWC";
    case ACL_FORMAT_ND:
        return "ACL_FORMAT_ND";
    case ACL_FORMAT_NC1HWC0:
        return "ACL_FORMAT_NC1HWC0";
    case ACL_FORMAT_FRACTAL_Z:
        return "ACL_FORMAT_FRACTAL_Z";
    case ACL_FORMAT_NC1HWC0_C04:
        return "ACL_FORMAT_NC1HWC0_C04";
    case ACL_FORMAT_HWCN:
        return "ACL_FORMAT_HWCN";
    case ACL_FORMAT_NDHWC:
        return "ACL_FORMAT_NDHWC";
    case ACL_FORMAT_FRACTAL_NZ:
        return "ACL_FORMAT_FRACTAL_NZ";
    case ACL_FORMAT_NCDHW:
        return "ACL_FORMAT_NCDHW";
    case ACL_FORMAT_NDC1HWC0:
        return "ACL_FORMAT_NDC1HWC0";
    case ACL_FRACTAL_Z_3D:
        return "ACL_FRACTAL_Z_3D";
    case ACL_FORMAT_NC:
        return "ACL_FORMAT_NC";
    case ACL_FORMAT_NCL:
        return "ACL_FORMAT_NCL";
    default:
        return "UNKNOWN";
    }
}
