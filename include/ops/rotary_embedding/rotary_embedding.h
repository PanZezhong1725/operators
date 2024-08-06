#ifndef ROTARY_EMBEDDING_H
#define ROTARY_EMBEDDING_H

#include "../../export.h"
#include "../../operators.h"

typedef struct RotaryEmbeddingDescriptor RotaryEmbeddingDescriptor;

__C __export void *createRotaryEmbeddingDescriptor(Device, void *config);
__C __export void destroyRotaryEmbeddingDescriptor(RotaryEmbeddingDescriptor *descriptor);
__C __export void rotaryEmbedding(RotaryEmbeddingDescriptor *descriptor, Tensor t, Tensor pos, float theta, void *stream);

#endif
