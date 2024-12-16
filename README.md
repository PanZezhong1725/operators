# InfiniOperators 算子库

跨平台高性能统一算子库。形式为 C 接口动态库。

## 简介

### 算子接口设计

采用3+1段式算子设计，每个算子都实现并对外暴露以下的 C 接口:

- 第一阶段：构造硬件控柄（Handle）。用户提供控柄地址、硬件类型以及硬件序号。控柄所在的内存空间由用户管理。

  ```C
  infiniopStatus_t infiniopCreateHandle(infiniopHandle_t *handle_ptr, int device, int device_id);
  ```

- 第二阶段：构造算子描述（Descriptor）。用户提供描述符地址、硬件控柄、以及算子涉及的张量描述（含张量数据类型、形状和步长）。这一步会完成算子所需的与张量数据无关的预计算。

  ```C
  infiniopStatus_t infiniopCreateOpDescriptor(infiniopHandle_t handle, infiniopOpDescriptor_t *desc_ptr, infiniopTensorDescriptor_t t, ...);
  ```

- 第三阶段（可选）：计算额外工作空间。根据算子描述，计算算子所需的额外工作空间大小，并存储于用户提供的位置。具体空间分配由用户负责。

  ```C
  infiniopStatus_t infiniopGetOpWorkspaceSize(infiniopOpDescriptor_t desc, uint64_t *size);
  ```

- 第四阶段：计算。根据算子描述符，在指定的硬件上执行相应计算，用户需要提供输入输出的数据，以及硬件计算流（CPU 为 NULL）。

  ```C
  infiniopStatus_t infiniopGetOp(infiniopOpDescriptor_t desc, [void *workspace, uint64_t workspace_size,] void *output_data, void *input_data, ..., void *stream);
  ```

- 销毁描述和硬件控柄。

  ```C
  infiniopStatus_t infiniopDestroyOpDescriptor(infiniopOpDescriptor_t desc);
  infiniopStatus_t infiniopDestroyHandle(infiniopHandle_t handle);
  ```

### 张量（Tensor）描述设计

张量描述由以下几个部分组成：

1.数据类型，由打包大小（即一个元素代表几个数据）、符号位、元素大小、尾数位数、指数位数共4字节表示。定义如下：

```C
typedef struct DataLayout {
    unsigned short
        packed : 8,
        sign : 1,
        size : 7,
        mantissa : 8,
        exponent : 8;
} DataLayout;
```

2.维度信息。张量有多少个维度。类型为uint64_t。

3.张量形状。张量每个维度的大小。类型为uint64_t*。

4.张量步长。张量每个维度的步长。类型为uint64_t*。

创建和销毁张量描述符的接口：

```C
infiniopStatus_t infiniopCreateTensorDescriptor(infiniopTensorDescriptor_t *desc_ptr, DataLayout layout, uint64_t ndim, uint64_t *shape, uint64_t *strides);
infiniopStatus_t infiniopDestroyTensorDescriptor(infiniopTensorDescriptor_t desc);
```

## 一、使用说明

### 1. 配置

#### 查看当前配置

```xmake
xmake f -v
```

#### 配置 CPU （默认配置）

```xmake
xmake f --cpu=true -cv
```

#### 配置 GPU

需要指定 CUDA 路径， 一般为 `CUDA_HOME` 或者 `CUDA_ROOT`。

```xmake
xmake f --nv-gpu=true --cuda=$CUDA_HOME -cv
```

#### 配置 MLU

```xmake
xmake f --cambricon-mlu=true -cv
```

#### 配置 NPU

````xmake
xmake f --ascend-npu=true -cv
````

### 2. 编译安装

```xmake
xmake build && xmake install
```

### 3. 设置环境变量

按输出提示设置 `INFINI_ROOT` 和 `LD_LIBRARY_PATH` 环境变量。

### 4. 运行算子测试

```bash
cd operatorspy/tests
python operator_name.py [--cpu | --cuda | --cambricon | --ascend]
```

## 二、开发说明

### 目录结构

```bash
├── xmake.lua  # xmake 构建脚本
├── include
│   ├── ops
│   │   ├── [operator_name].h  # 对外暴露的算子 C 接口定义，descriptor 定义
│   ├── tensor
│   │   ├── tensor_descriptor.h  # 对外暴露的张量 descriptor 定义
│   ├── handle
│   │   ├── handle_export.h  # 对外暴露的硬件 handle 定义
│   ├── *.h  # 对外暴露的核心结构体定义
├── src
│   ├── devices
│   │   ├── [device_name]
│   │       ├── *.cc/.h # 特定硬件（如 cpu、英伟达）通用代码
│   ├── ops
│   │   ├── utils.h  # 全算子通用代码 (如 assert)
│   │   ├── [operator_name]  # 算子实现目录
│   │       ├── operator.cc # 算子 C 接口实现 (根据 descriptor 调用不同的算子实现)
│   │       ├── [device_name]
│   │       │   ├── *.cc/.h/... # 特定硬件的算子实现代码
│   ├── *.h  # 核心结构体定义
│  
├── operatorspy  # Python 封装以及测试脚本
    ├── tests
    │   ├── operator_name.py  # 测试脚本
    ├── *.py     # Python 封装代码
```

### 增加新的硬件

- 在 `src/device.h` 和 `operatorspy/devices.py` 中增加新的硬件类型，注意两者需要一一对应；
- 在 `xmake.lua` 中增加新硬件的编译选项以及编译方式；
- 在 `src/ops/devices/[device_name]` 下编写特定硬件的handle实现和通用代码；
- 实现该硬件的算子；

### 增加新的算子

- 在 `src/ops/[operator_name]` 增加创建/销毁算子描述符、算子计算的C接口，注意C接口header使用`__C __export`前缀；
- 在 `src/ops/[operator_name]/[device_name]` 增加算子在各硬件的实现代码；
- 在 `operatorspy/tests/[operator_name].py` 增加算子测试；
