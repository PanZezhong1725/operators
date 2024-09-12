#ifndef __COMMON_MUSA_H__
#define __COMMON_MUSA_H__

enum class Type {
    QINT4,
    QINT8,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    HALF,
    BFLOAT16,
    FLOAT,
    DOUBLE,
    BOOL,
};

enum class Format {
    UNKNOWN,
    NCW,
    NWC,
    NCHW,
    NHWC,
    HWCN,
    NCDHW,
    NDHWC,
    DHWCN,
};

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_WARP_PER_BLOCK 128
#define WARP_SIZE 128

#endif // __COMMON_MUSA_H__