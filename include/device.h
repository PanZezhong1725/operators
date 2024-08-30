#ifndef __DEVICE_H__
#define __DEVICE_H__

enum DeviceEnum {
    DevCpu,
    DevNvGpu,
    DevCambriconMlu,
    DevMtGpu,
};

typedef enum DeviceEnum Device;

#endif// __DEVICE_H__
