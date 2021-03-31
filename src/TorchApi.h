#pragma once

//#ifndef TORCH_API_H
//#define TORCH_API_H

#ifdef __cplusplus
namespace Scripting {
#endif

typedef struct Conv2D { int handle; } Conv2D;
typedef struct Tensor { int handle; } Tensor;
typedef struct LinearLayer { int handle; } LinearLayer;
typedef struct Object { int handle; } Object;
typedef struct Value { int handle; } Value;
typedef struct Optimizer { int handle; } Optimizer;
typedef struct TrainResult { int handle; } TrainResult;

Tensor tensor_new();
Tensor tensor_new_4d(int x, int y, int z, int w);
Tensor tensor_clone(Tensor input);
void tensor_squeeze(Tensor tensor);
void tensor_backward(Tensor tensor);
float tensor_item_float(Tensor tensor);
void tensor_print(Tensor tensor);

Conv2D conv2d_new(const char* name, int a, int b, int c);
Conv2D conv2d_get(const char* name);
void conv2d_forward(Conv2D conv, Tensor input, Tensor output);

LinearLayer linearlayer_new(const char* name, int totalFeatures, int hiddenNodesCount);
LinearLayer linearlayer_get(const char* name);
void linearlayer_forward(LinearLayer layer, Tensor input);

void relu(Tensor input);
void smooth_l1_loss(Tensor input, Tensor target, Tensor result);

Tensor pack_into_tensor(void (*selector)(Object input, Value output));

void value_set_float(Value value, float v);
void value_set_float_array(Value value, float* array, int count);

float object_get_float(Object input, const char* name);

Optimizer optimizer_new_adam(const char* name, float learningRate);
void optimizer_zero_grad(Optimizer optimizer);
void optimizer_step(Optimizer optimizer);
Optimizer optimizer_get(const char* name);

void trainresult_set_loss(TrainResult result, float loss);

#ifdef __cplusplus
};
#endif

//#endif
