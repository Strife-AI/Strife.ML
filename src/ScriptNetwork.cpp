#include "ScriptNetwork.hpp"
#include "TorchApi.h"

using namespace Scripting;

void RegisterScriptFunctions()
{
    SCRIPT_REGISTER(tensor_new);
    SCRIPT_REGISTER(tensor_new_4d);
    SCRIPT_REGISTER(tensor_clone);
    SCRIPT_REGISTER(tensor_squeeze);

    SCRIPT_REGISTER(conv2d_new)
    SCRIPT_REGISTER(conv2d_get)
    SCRIPT_REGISTER(conv2d_forward);

    SCRIPT_REGISTER(relu);

    SCRIPT_REGISTER(linearlayer_new);
    SCRIPT_REGISTER(linearlayer_get);
    SCRIPT_REGISTER(linearlayer_forward);

    SCRIPT_REGISTER(object_get_float);

    SCRIPT_REGISTER(optimizer_new_adam);

    SCRIPT_REGISTER(value_set_float);
    SCRIPT_REGISTER(value_set_float_array);

    SCRIPT_REGISTER(pack_into_tensor);
};