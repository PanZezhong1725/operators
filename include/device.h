﻿#ifndef __DEVICE_H__
#define __DEVICE_H__

enum DeviceEnum {
    DevCpu,
    DevNvGpu,
    DevCambriconMlu,
    DevAscendNpu,
    DevIluvatarBi,
};

typedef enum DeviceEnum Device;

#endif// __DEVICE_H__
