#include "ScriptNetwork.hpp"
#include "TorchApi.h"

using namespace Scripting;

void RegisterScriptFunctions()
{
    SCRIPT_REGISTER(tensor_new);
    SCRIPT_REGISTER(tensor_new_4d);
    SCRIPT_REGISTER(tensor_clone);
    SCRIPT_REGISTER(tensor_squeeze);
    SCRIPT_REGISTER(tensor_backward);
    SCRIPT_REGISTER(tensor_print);
    SCRIPT_REGISTER(tensor_item_float);
    SCRIPT_REGISTER(tensor_item_int64);
    SCRIPT_REGISTER(tensor_max);

    SCRIPT_REGISTER(conv2d_new)
    SCRIPT_REGISTER(conv2d_get)
    SCRIPT_REGISTER(conv2d_forward);

    SCRIPT_REGISTER(optimizer_new_adam);
    SCRIPT_REGISTER(optimizer_get);
    SCRIPT_REGISTER(optimizer_zero_grad);
    SCRIPT_REGISTER(optimizer_step);

    SCRIPT_REGISTER(relu);

    SCRIPT_REGISTER(linearlayer_new);
    SCRIPT_REGISTER(linearlayer_get);
    SCRIPT_REGISTER(linearlayer_forward);

    SCRIPT_REGISTER(object_get_float);

    SCRIPT_REGISTER(value_set_float);
    SCRIPT_REGISTER(value_set_float_array);
    SCRIPT_REGISTER(value_set_int32);

    SCRIPT_REGISTER(pack_into_tensor);

    SCRIPT_REGISTER(smooth_l1_loss);
};